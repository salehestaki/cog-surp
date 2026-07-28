"""Deterministic, redistributable synthetic dashboard demo."""

from __future__ import annotations

import hashlib
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from cog_surp import __version__
from cog_surp.provenance.checksums import sha256_file
from cog_surp.provenance.manifests import canonical_json_bytes
from cog_surp.provenance.runtime import collect_runtime_provenance
from cog_surp.release.manifest import (
    ArtifactType,
    DatasetReference,
    DataStatus,
    ModelReference,
    ReleaseArtifact,
    ReleaseManifest,
    RunLineage,
    load_release_manifest,
)

DEMO_NOTICE = "SYNTHETIC TEST/DEMO DATA — NOT HUMAN EVIDENCE"


def _svg(title: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420"
viewBox="0 0 960 420" role="img" aria-label="{title}">
<rect width="960" height="420" fill="#fbfbfd"/>
<text x="48" y="52" font-family="sans-serif" font-size="25"
 fill="#202938">{title}</text>
<rect x="48" y="72" width="864" height="38" rx="8" fill="#fff1c2"/>
<text x="64" y="98" font-family="sans-serif" font-size="16"
 fill="#6b4f00">{DEMO_NOTICE}</text>
{body}
</svg>
"""


def _wave_svg(*, difference: bool) -> str:
    if difference:
        title = "Synthetic unrelated-minus-related difference wave"
        related = ""
        path = (
            '<path d="M80 285 C180 280 260 270 340 245 '
            "C410 205 445 125 500 138 C565 150 610 225 690 255 "
            'C770 270 835 265 890 250" fill="none" stroke="#9d174d" '
            'stroke-width="5"/>'
        )
    else:
        title = "Synthetic N400 condition waveforms"
        related = (
            '<path d="M80 260 C190 250 300 230 390 225 '
            'C470 215 530 220 610 230 C720 240 800 238 890 235" '
            'fill="none" stroke="#2563eb" stroke-width="5"/>'
        )
        path = (
            '<path d="M80 258 C190 250 300 235 390 205 '
            "C455 160 500 128 560 155 C625 188 670 235 750 248 "
            'C805 255 850 248 890 242" fill="none" stroke="#e11d48" '
            'stroke-width="5"/>'
        )
    body = f"""
<line x1="80" y1="300" x2="890" y2="300" stroke="#6b7280"/>
<line x1="220" y1="130" x2="220" y2="330" stroke="#6b7280"/>
<rect x="450" y="125" width="145" height="180" fill="#e5e7eb" opacity=".65"/>
{related}{path}
<text x="410" y="360" font-family="sans-serif" font-size="16"
 fill="#374151">Time from synthetic target onset</text>
"""
    return _svg(title, body)


def _topomap_svg() -> str:
    body = """
<defs><radialGradient id="g"><stop offset="0" stop-color="#2563eb"/>
<stop offset=".55" stop-color="#f8fafc"/><stop offset="1" stop-color="#dc2626"/>
</radialGradient></defs>
<circle cx="480" cy="245" r="118" fill="url(#g)" stroke="#111827" stroke-width="4"/>
<path d="M450 128 L480 105 L510 128" fill="none" stroke="#111827" stroke-width="4"/>
<circle cx="440" cy="225" r="5" fill="#111827"/>
<circle cx="520" cy="225" r="5" fill="#111827"/>
<circle cx="480" cy="270" r="5" fill="#111827"/>
"""
    return _svg("Synthetic N400-window scalp pattern", body)


def _cluster_svg() -> str:
    body = """
<rect x="90" y="145" width="800" height="190" fill="#f3f4f6" stroke="#9ca3af"/>
<path d="M90 300 C180 275 250 292 330 250 C410 205 480 160 560 205
C650 250 730 230 890 280" fill="none" stroke="#7c3aed" stroke-width="6"/>
<rect x="420" y="145" width="170" height="190" fill="#fbbf24" opacity=".25"/>
"""
    return _svg("Synthetic exploratory sensor-time statistic", body)


def _causal_svg() -> str:
    body = """
<rect x="80" y="175" width="220" height="80" rx="12" fill="#dbeafe" stroke="#2563eb"/>
<text x="105" y="220" font-family="sans-serif" font-size="18">Synthetic condition</text>
<rect x="380" y="135" width="220" height="80" rx="12" fill="#fee2e2" stroke="#dc2626"/>
<text x="425" y="180" font-family="sans-serif" font-size="18">Demo N400</text>
<rect x="660" y="235" width="220" height="80" rx="12" fill="#ede9fe" stroke="#7c3aed"/>
<text x="695" y="280" font-family="sans-serif" font-size="18">Model measure</text>
<path d="M300 205 L375 180" stroke="#111827" stroke-width="4" marker-end="url(#arrow)"/>
<path d="M300 225 L655 270" stroke="#111827" stroke-width="4" marker-end="url(#arrow)"/>
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3"
orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#111827"/></marker></defs>
"""
    return _svg("Synthetic demo causal graph — no model-to-EEG edge", body)


def _demo_tables() -> dict[ArtifactType, pd.DataFrame]:
    import pandas as pd

    item_rows = [
        ("item-01", "bread", 0.70, 1.5, 1.9, 1),
        ("item-02", "river", 0.48, 2.0, 2.8, 1),
        ("item-03", "cloud", 0.28, 2.4, 3.7, 1),
        ("item-04", "piano", 0.62, 1.8, 2.2, 2),
        ("item-05", "garden", 0.35, 2.2, 3.2, 1),
        ("item-06", "window", 0.18, 2.6, 4.1, 1),
    ]
    feature_rows = []
    for participant_index, participant in enumerate(("demo-01", "demo-02")):
        for item_index, (
            item,
            word,
            cloze,
            entropy,
            surprisal,
            tokens,
        ) in enumerate(item_rows):
            feature_rows.append(
                {
                    "participant": participant,
                    "item": item,
                    "target_word": word,
                    "human_cloze_probability": cloze,
                    "human_cloze_surprisal_nats": -math.log(cloze),
                    "human_response_entropy_nats": entropy,
                    "target_surprisal_nats": surprisal,
                    "target_token_count": tokens,
                    "n400_mean_voltage_uv": (
                        -0.35 * surprisal + 0.08 * item_index + 0.12 * participant_index
                    ),
                    "model_id": "cog-surp/synthetic-demo-lm",
                    "model_revision": "demo-revision-v1",
                    "probability_strategy": "boundary-aware",
                }
            )
    features = pd.DataFrame.from_records(feature_rows)
    predictive = pd.DataFrame.from_records(
        [
            {
                "split": split,
                "model": model,
                "mean_rmse_uv": rmse,
                "sd_rmse_uv": 0.04,
                "mean_r2": r2,
                "sd_r2": 0.02,
                "folds": 2,
            }
            for split in ("leave-items-out", "leave-participants-out")
            for model, rmse, r2 in (
                ("lexical-controls", 1.20, 0.01),
                ("lm-surprisal", 1.12, 0.05),
                ("combined", 1.08, 0.08),
            )
        ]
    )
    posterior = pd.DataFrame.from_records(
        [
            {
                "parameter": parameter,
                "mean": mean,
                "hdi_3%_lb": mean - 0.18,
                "hdi_97%_ub": mean + 0.18,
            }
            for parameter, mean in (
                ("target_surprisal_nats_z", -0.30),
                ("human_cloze_surprisal_nats_z", -0.22),
                ("human_response_entropy_nats_z", -0.08),
                ("word_frequency_z", 0.10),
                ("number_of_letters_z", -0.04),
                ("word_position_z", 0.02),
            )
        ]
    )
    robustness = pd.DataFrame.from_records(
        [
            {
                "model_role": "demo-reference",
                "model_id": "cog-surp/synthetic-demo-lm",
                "model_revision": "demo-revision-v1",
                "surprisal_coefficient_uv_per_sd": -0.30,
                "coefficient_hdi95_lb": -0.48,
                "coefficient_hdi95_ub": -0.12,
                "leave_items_out_rmse_uv": 1.12,
            },
            {
                "model_role": "demo-comparison",
                "model_id": "cog-surp/synthetic-demo-lm-b",
                "model_revision": "demo-revision-v1",
                "surprisal_coefficient_uv_per_sd": -0.25,
                "coefficient_hdi95_lb": -0.44,
                "coefficient_hdi95_ub": -0.07,
                "leave_items_out_rmse_uv": 1.15,
            },
        ]
    )
    qc = pd.DataFrame.from_records(
        [
            {
                "participant": participant,
                "preprocessing_run_id": "demo-preprocessing-v1",
                "participant_included": True,
                "accepted_trials": 36,
                "rejected_trials": 4,
                "rejection_fraction": 0.10,
                "behavior_accuracy": 0.92,
                "related_accepted_trials": 18,
                "unrelated_accepted_trials": 18,
            }
            for participant in ("demo-01", "demo-02")
        ]
    )
    cluster = pd.DataFrame.from_records(
        [
            {
                "cluster": 1,
                "p_value": 0.12,
                "time_start_s": 0.32,
                "time_end_s": 0.48,
                "channels": "CPz,Pz",
            }
        ]
    )
    surprisal = features.drop_duplicates("item")[
        ["item", "target_word", "target_surprisal_nats", "model_id", "model_revision"]
    ].copy()
    return {
        ArtifactType.FEATURES: features,
        ArtifactType.LM_SURPRISAL: surprisal,
        ArtifactType.PREDICTIVE_SUMMARY: predictive,
        ArtifactType.POSTERIOR_SUMMARY: posterior,
        ArtifactType.ROBUSTNESS: robustness,
        ArtifactType.H1_PARTICIPANT_QC: qc,
        ArtifactType.CLUSTER_SUMMARY: cluster,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value))


def build_synthetic_demo(
    *,
    output_dir: Path,
    project_root: Path,
) -> tuple[Path, ReleaseManifest]:
    """Build or validate a deterministic, clearly synthetic public demo."""
    manifest_path = output_dir / "release-manifest.json"
    if manifest_path.is_file():
        return manifest_path, load_release_manifest(manifest_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            f"demo output exists without a valid manifest; refusing overwrite: "
            f"{output_dir}"
        )
    temporary = output_dir.parent / f".{output_dir.name}.tmp"
    if temporary.exists():
        raise ValueError(f"temporary demo output already exists: {temporary}")
    artifacts_dir = temporary / "artifacts"
    artifacts_dir.mkdir(parents=True)
    try:
        tables = _demo_tables()
        written: dict[ArtifactType, Path] = {}
        for artifact_type, frame in tables.items():
            path = artifacts_dir / f"{artifact_type.value}.parquet"
            frame.to_parquet(path, index=False)
            written[artifact_type] = path

        diagnostics = {
            "schema_version": 1,
            "data_status": "synthetic",
            "max_rhat": 1.001,
            "min_bulk_ess": 950,
            "divergences": 0,
            "posterior_predictive_rmse_uv": 1.08,
            "notice": DEMO_NOTICE,
        }
        h1 = {
            "schema_version": 1,
            "run_id": "demo-preprocessing-v1",
            "data_status": "synthetic",
            "participant_counts": {
                "public_available": 2,
                "primary_included": 2,
                "primary_excluded": 0,
            },
            "primary_rule_based_cohort": {
                "estimate_uv": -1.25,
                "ci95_low_uv": -1.70,
                "ci95_high_uv": -0.80,
                "sign_convention": (
                    "Synthetic negative values imitate a larger demo N400; "
                    "they are not human evidence."
                ),
            },
            "all_public_participants_sensitivity": {
                "estimate_uv": -1.25,
                "ci95_low_uv": -1.70,
                "ci95_high_uv": -0.80,
            },
            "notice": DEMO_NOTICE,
        }
        h2_models = [
            {
                "model_id": "cog-surp/synthetic-demo-lm",
                "estimate_nats": 1.40,
                "ci95_low_nats": 0.90,
                "ci95_high_nats": 1.90,
                "n_matched_targets": 6,
            },
            {
                "model_id": "cog-surp/synthetic-demo-lm-b",
                "estimate_nats": 1.10,
                "ci95_low_nats": 0.60,
                "ci95_high_nats": 1.60,
                "n_matched_targets": 6,
            },
        ]
        h2 = {
            "schema_version": 2,
            "run_id": "demo-h2-v1",
            "data_status": "synthetic",
            "models": h2_models,
            "notice": DEMO_NOTICE,
        }
        causal = {
            "schema_version": 1,
            "run_id": "demo-causal-v1",
            "data_status": "synthetic",
            "treatment": "synthetic_condition",
            "outcome": "synthetic_n400",
            "adjustment_set": [],
            "estimate_uv": -1.25,
            "refuters": {
                "placebo_treatment": 0.01,
                "random_common_cause": -1.24,
                "data_subset": -1.20,
            },
            "claim_boundary": (
                "Demonstration output only. No human or neuroscience conclusion."
            ),
            "notice": DEMO_NOTICE,
        }
        cluster_metadata = {
            "schema_version": 1,
            "run_id": "demo-cluster-v1",
            "data_status": "synthetic",
            "participants": 2,
            "n_permutations": 32,
            "clusters": 1,
            "clusters_passing_alpha": 0,
            "interpretation_boundary": (
                "Synthetic exploratory display only; no human timing, source, "
                "or anatomical inference."
            ),
            "notice": DEMO_NOTICE,
        }
        json_values = {
            ArtifactType.DIAGNOSTICS: diagnostics,
            ArtifactType.H1_EFFECT: h1,
            ArtifactType.H2_EFFECT: h2,
            ArtifactType.CAUSAL_AUDIT: causal,
            ArtifactType.CLUSTER_METADATA: cluster_metadata,
        }
        for artifact_type, json_value in json_values.items():
            path = artifacts_dir / f"{artifact_type.value}.json"
            _write_json(path, json_value)
            written[artifact_type] = path

        svg_values = {
            ArtifactType.H1_CONDITION_ERP: _wave_svg(difference=False),
            ArtifactType.H1_DIFFERENCE_WAVE: _wave_svg(difference=True),
            ArtifactType.H1_TOPOMAP: _topomap_svg(),
            ArtifactType.CAUSAL_GRAPH: _causal_svg(),
            ArtifactType.CLUSTER_FIGURE: _cluster_svg(),
        }
        for artifact_type, svg_value in svg_values.items():
            path = artifacts_dir / f"{artifact_type.value}.svg"
            path.write_text(svg_value, encoding="utf-8")
            written[artifact_type] = path

        environment = collect_runtime_provenance(project_root)
        code = environment["code"]
        identity = {
            "schema_version": 1,
            "project_version": __version__,
            "demo_version": "synthetic-demo-v1",
            "implementation": code["code_tree_sha256"],
            "artifacts": {
                artifact_type.value: sha256_file(path)
                for artifact_type, path in sorted(
                    written.items(),
                    key=lambda item: item[0].value,
                )
            },
        }
        digest = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        release_id = f"demo-{digest[:12]}"
        report = f"""# Cog-Surp synthetic demo report

**{DEMO_NOTICE}**

- Manifest ID: `{release_id}`
- Project version: `{__version__}`
- Demo observations: 12
- Demo participants: 2 simulated identifiers
- Demo items: 6

The displayed coefficients, intervals, waveforms, refuters, and held-out
metrics are deterministic demonstration outputs. They are not measurements
from a person and support no conclusion about human cognition or neuroscience.

## Provenance

The release manifest records the code revision, every artifact checksum,
synthetic status, run lineage, model fixture identity, and this report.
"""
        report_path = artifacts_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        written[ArtifactType.REPORT] = report_path

        runs = RunLineage(
            preprocessing_run_id="demo-preprocessing-v1",
            feature_run_id="demo-feature-v1",
            lm_scoring_run_ids=["demo-lm-v1"],
            statistical_analysis_run_ids=[
                "demo-predictive-v1",
                "demo-bayesian-v1",
                "demo-robustness-v1",
                "demo-h2-v1",
                "demo-cluster-v1",
            ],
            causal_analysis_run_ids=["demo-causal-v1"],
            report_run_id="demo-report-v1",
        )
        release_parents = [
            "demo-features",
            "demo-h1-effect",
            "demo-h2-effect",
            "demo-causal-audit",
        ]
        artifacts: list[ReleaseArtifact] = []
        for artifact_type, path in sorted(
            written.items(),
            key=lambda item: item[0].value,
        ):
            artifact_id = f"demo-{artifact_type.value}"
            source_run_id, parent_run_ids = _demo_lineage(artifact_type)
            parent_artifact_ids: list[str] = []
            if artifact_type is ArtifactType.FEATURES:
                parent_artifact_ids = ["demo-lm-surprisal"]
            if artifact_type in {
                ArtifactType.PREDICTIVE_SUMMARY,
                ArtifactType.POSTERIOR_SUMMARY,
                ArtifactType.DIAGNOSTICS,
                ArtifactType.ROBUSTNESS,
            }:
                parent_artifact_ids = ["demo-features"]
            if artifact_type is ArtifactType.REPORT:
                parent_artifact_ids = release_parents
            artifacts.append(
                ReleaseArtifact(
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    path=f"artifacts/{path.name}",
                    sha256=sha256_file(path),
                    schema_version=1,
                    data_status=DataStatus.SYNTHETIC,
                    source_run_id=source_run_id,
                    parent_artifact_ids=parent_artifact_ids,
                    parent_run_ids=parent_run_ids,
                    dataset_id="synthetic-demo-v1",
                    model_id=(
                        "cog-surp/synthetic-demo-lm"
                        if artifact_type
                        in {ArtifactType.FEATURES, ArtifactType.LM_SURPRISAL}
                        else None
                    ),
                    label=f"Synthetic demo · {artifact_type.value}",
                )
            )
        manifest = ReleaseManifest(
            manifest_schema_version=1,
            project_version=__version__,
            release_id=release_id,
            label="Cog-Surp deterministic synthetic public demo",
            created_at_utc=datetime(2026, 7, 28, tzinfo=UTC),
            git_commit=str(code["git_revision"] or "uncommitted"),
            git_dirty=bool(code["git_dirty"]),
            source_implementation_sha256=str(code["code_tree_sha256"]),
            resolved_configuration_hashes={
                "synthetic-demo-v1": hashlib.sha256(
                    b"cog-surp-synthetic-demo-v1"
                ).hexdigest()
            },
            data_status=DataStatus.SYNTHETIC,
            datasets=[
                DatasetReference(
                    dataset_id="synthetic-demo-v1",
                    version="1",
                    sha256=hashlib.sha256(
                        b"cog-surp-synthetic-demo-dataset-v1"
                    ).hexdigest(),
                    data_status=DataStatus.SYNTHETIC,
                    citation="Cog-Surp deterministic synthetic demo fixture.",
                )
            ],
            models=[
                ModelReference(
                    model_id="cog-surp/synthetic-demo-lm",
                    revision="demo-revision-v1",
                    tokenizer_revision="demo-tokenizer-v1",
                    scoring_run_ids=["demo-lm-v1"],
                )
            ],
            runs=runs,
            parent_artifact_ids=release_parents,
            artifacts=artifacts,
            citations=[
                "Cog-Surp software; synthetic fixture requires no dataset citation."
            ],
            known_limitations=[
                "All values are simulated and are not human evidence.",
                (
                    "The demo exercises presentation and integrity paths, "
                    "not model inference."
                ),
            ],
        )
        (temporary / "release-manifest.json").write_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json"))
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest_path, load_release_manifest(manifest_path)


def _demo_lineage(artifact_type: ArtifactType) -> tuple[str, list[str]]:
    if artifact_type is ArtifactType.FEATURES:
        return "demo-feature-v1", ["demo-lm-v1"]
    if artifact_type is ArtifactType.LM_SURPRISAL:
        return "demo-lm-v1", []
    if artifact_type in {
        ArtifactType.PREDICTIVE_SUMMARY,
        ArtifactType.POSTERIOR_SUMMARY,
        ArtifactType.DIAGNOSTICS,
    }:
        return "demo-bayesian-v1", ["demo-feature-v1"]
    if artifact_type is ArtifactType.ROBUSTNESS:
        return "demo-robustness-v1", ["demo-feature-v1"]
    if artifact_type in {
        ArtifactType.H1_EFFECT,
        ArtifactType.H1_CONDITION_ERP,
        ArtifactType.H1_DIFFERENCE_WAVE,
        ArtifactType.H1_TOPOMAP,
        ArtifactType.H1_PARTICIPANT_QC,
    }:
        return "demo-preprocessing-v1", []
    if artifact_type is ArtifactType.H2_EFFECT:
        return "demo-h2-v1", ["demo-lm-v1"]
    if artifact_type in {ArtifactType.CAUSAL_AUDIT, ArtifactType.CAUSAL_GRAPH}:
        return "demo-causal-v1", ["demo-preprocessing-v1", "demo-lm-v1"]
    if artifact_type in {
        ArtifactType.CLUSTER_METADATA,
        ArtifactType.CLUSTER_SUMMARY,
        ArtifactType.CLUSTER_FIGURE,
    }:
        return "demo-cluster-v1", ["demo-preprocessing-v1"]
    if artifact_type is ArtifactType.REPORT:
        return "demo-report-v1", []
    raise AssertionError(f"unhandled demo artifact type: {artifact_type}")
