from __future__ import annotations

from pathlib import Path

import pytest

from cog_surp.stimuli.erp_core import load_erp_core_stimuli


def _write_balanced_lists(root: Path) -> None:
    stimuli = root / "stimuli"
    stimuli.mkdir()
    list_one = []
    list_two = []
    for index in range(50):
        target_a = f"A{index}"
        target_b = f"B{index}"
        list_one.append(f"RA{index}\t{target_a}\tUA{index}\t{target_b}")
        list_two.append(f"RB{index}\t{target_b}\tUB{index}\t{target_a}")
    (stimuli / "N400_stimuli_list1_English.txt").write_text(
        "\n".join(list_one), encoding="utf-8"
    )
    (stimuli / "N400_stimuli_list2_English.txt").write_text(
        "\n".join(list_two), encoding="utf-8"
    )


def test_publisher_lists_are_structured_and_counterbalanced(tmp_path: Path) -> None:
    _write_balanced_lists(tmp_path)

    frame = load_erp_core_stimuli(tmp_path)

    assert len(frame) == 200
    assert frame["target_word"].nunique() == 100
    assert set(frame["condition"]) == {"related", "unrelated"}
    assert set(frame["validation_status"]) == {"publisher-validated"}


def test_malformed_stimulus_row_is_rejected(tmp_path: Path) -> None:
    _write_balanced_lists(tmp_path)
    path = tmp_path / "stimuli" / "N400_stimuli_list1_English.txt"
    path.write_text("ONLY\tTHREE\tFIELDS\n", encoding="utf-8")

    with pytest.raises(ValueError, match="four nonempty fields"):
        load_erp_core_stimuli(tmp_path)
