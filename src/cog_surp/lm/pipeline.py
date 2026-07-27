"""Artifact-producing language-model scoring pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cog_surp.lm.backends import SurprisalBackend
from cog_surp.lm.domain import ScoringRequest, ScoringResult


def score_stimulus_artifact(
    *,
    stimuli_path: Path,
    backend: SurprisalBackend,
    output_path: Path,
    batch_size: int,
) -> int:
    """Score every validated stimulus row and write a token-auditable Parquet."""
    import pandas as pd

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    stimuli = pd.read_parquet(stimuli_path)
    if "scoreable" in stimuli.columns:
        stimuli = stimuli.loc[stimuli["scoreable"].astype(bool)].reset_index(drop=True)
    required = {"condition", "context_text", "target_text"}
    missing = required - set(stimuli.columns)
    if missing:
        raise ValueError(f"stimulus artifact missing columns: {sorted(missing)}")
    if stimuli.empty:
        raise ValueError("stimulus artifact has no scoreable rows")

    def request_id(row: Any) -> str:
        if hasattr(row, "item"):
            return f"{getattr(row, 'dataset_id', 'dataset')}:{row.item}:{row.condition}"
        return f"{row.source_file}:{row.source_row}:{row.condition}:{row.target_text}"

    requests = [
        ScoringRequest(
            request_id=request_id(row),
            context=str(row.context_text),
            target=str(row.target_text),
        )
        for row in stimuli.itertuples(index=False)
    ]
    results: list[ScoringResult] = []
    for start in range(0, len(requests), batch_size):
        results.extend(backend.score(requests[start : start + batch_size]))
    by_id = {result.request_id: result for result in results}
    if len(by_id) != len(requests):
        raise RuntimeError("backend returned missing or duplicate scoring results")
    records: list[dict[str, Any]] = []
    for row, request in zip(stimuli.to_dict("records"), requests, strict=True):
        result = by_id[request.request_id]
        records.append(
            {
                **row,
                "request_id": request.request_id,
                "full_text": result.text,
                "target_start": result.target_start,
                "target_end": result.target_end,
                "token_count": len(result.tokens),
                "target_token_count": len(result.target_token_positions),
                "target_token_positions": json.dumps(result.target_token_positions),
                "target_log_probability": result.target_log_probability,
                "target_surprisal_nats": result.target_surprisal_nats,
                "target_surprisal_bits": result.target_surprisal_bits,
                "probability_strategy": result.probability_strategy,
                "model_id": result.model_id,
                "model_revision": result.model_revision,
                "tokenizer_revision": result.tokenizer_revision,
                "dtype": result.dtype,
                "quantization": result.quantization,
                "token_observations": json.dumps(
                    [asdict(token) for token in result.tokens],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(output_path, index=False)
    return len(records)
