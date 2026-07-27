"""Deterministic synthetic N400 epochs for recovery tests and demos only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SyntheticN400Config(BaseModel):
    """Fully specified simulator configuration with a mandatory seed."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int
    participants: int = Field(default=12, ge=2)
    items: int = Field(default=40, ge=2)
    sampling_frequency_hz: float = Field(default=256, gt=0)
    epoch_start_s: float = -0.2
    epoch_end_s: float = 0.8
    baseline_end_s: float = 0.0
    channels: tuple[str, ...] = ("Fz", "Cz", "CPz", "Pz")
    roi_channels: tuple[str, ...] = ("Cz", "CPz")
    n400_latency_s: float = 0.4
    n400_width_s: float = Field(default=0.075, gt=0)
    treatment_effect_uv: float = -3.0
    participant_intercept_sd_uv: float = Field(default=1.0, ge=0)
    participant_slope_sd_uv: float = Field(default=0.4, ge=0)
    item_intercept_sd_uv: float = Field(default=0.8, ge=0)
    item_slope_sd_uv: float = Field(default=0.3, ge=0)
    noise_sd_uv: float = Field(default=2.0, gt=0)
    anomaly_noise_multiplier: float = Field(default=1.2, gt=0)
    autocorrelation: float = Field(default=0.8, ge=0, lt=1)
    line_noise_uv: float = Field(default=0.0, ge=0)
    blink_probability: float = Field(default=0.0, ge=0, le=1)
    blink_amplitude_uv: float = Field(default=20.0, ge=0)
    missing_trial_probability: float = Field(default=0.0, ge=0, lt=1)

    @model_validator(mode="after")
    def validate_geometry(self) -> SyntheticN400Config:
        if not self.epoch_start_s < self.baseline_end_s <= 0 < self.epoch_end_s:
            raise ValueError("epoch and baseline bounds are inconsistent")
        if not set(self.roi_channels) <= set(self.channels):
            raise ValueError("ROI channels must be present in the montage")
        if not self.epoch_start_s < self.n400_latency_s < self.epoch_end_s:
            raise ValueError("N400 latency must fall inside the epoch")
        return self


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """Synthetic epochs and tidy outcomes, always marked synthetic."""

    times_s: np.ndarray
    channel_names: tuple[str, ...]
    epochs_uv: np.ndarray
    trials: pd.DataFrame
    expected_window_effect_uv: float
    data_status: str = "synthetic"


class SyntheticN400Generator:
    """Generate null or non-null crossed participant/item N400 fixtures."""

    def __init__(self, config: SyntheticN400Config) -> None:
        self.config = config

    def generate(self) -> SyntheticDataset:
        """Generate deterministic epochs and outcomes from the recorded seed."""
        config = self.config
        rng = np.random.default_rng(config.seed)
        times = np.arange(
            config.epoch_start_s,
            config.epoch_end_s + 0.5 / config.sampling_frequency_hz,
            1 / config.sampling_frequency_hz,
        )
        waveform = np.exp(
            -0.5 * ((times - config.n400_latency_s) / config.n400_width_s) ** 2
        )
        topography = _centro_parietal_topography(config.channels)
        participant_intercepts = rng.normal(
            0, config.participant_intercept_sd_uv, config.participants
        )
        participant_slopes = rng.normal(
            0, config.participant_slope_sd_uv, config.participants
        )
        item_intercepts = rng.normal(0, config.item_intercept_sd_uv, config.items)
        item_slopes = rng.normal(0, config.item_slope_sd_uv, config.items)
        rows: list[dict[str, Any]] = []
        epochs: list[np.ndarray] = []
        roi_indices = [
            config.channels.index(channel) for channel in config.roi_channels
        ]
        window = (times >= 0.3) & (times <= 0.5)
        baseline = (times >= config.epoch_start_s) & (times <= config.baseline_end_s)

        trial_number = 0
        for participant in range(config.participants):
            for item in range(config.items):
                for treatment, condition in ((0, "related"), (1, "unrelated")):
                    trial_number += 1
                    if rng.random() < config.missing_trial_probability:
                        continue
                    amplitude = (
                        participant_intercepts[participant]
                        + item_intercepts[item]
                        + treatment
                        * (
                            config.treatment_effect_uv
                            + participant_slopes[participant]
                            + item_slopes[item]
                        )
                    )
                    noise_scale = config.noise_sd_uv * (
                        config.anomaly_noise_multiplier if treatment else 1.0
                    )
                    epoch = _ar1_noise(
                        rng,
                        channels=len(config.channels),
                        samples=len(times),
                        rho=config.autocorrelation,
                        scale=noise_scale,
                    )
                    epoch += amplitude * topography[:, None] * waveform[None, :]
                    if config.line_noise_uv:
                        phase = rng.uniform(0, 2 * np.pi)
                        epoch += config.line_noise_uv * np.sin(
                            2 * np.pi * 60 * times + phase
                        )
                    blink_present = rng.random() < config.blink_probability
                    if blink_present:
                        frontal = np.array(
                            [
                                1.0 if "F" in channel else 0.2
                                for channel in config.channels
                            ]
                        )
                        blink = np.exp(-0.5 * ((times - 0.15) / 0.035) ** 2)
                        epoch += (
                            config.blink_amplitude_uv
                            * frontal[:, None]
                            * blink[None, :]
                        )
                    epoch -= epoch[:, baseline].mean(axis=1, keepdims=True)
                    outcome = float(epoch[roi_indices][:, window].mean())
                    epochs.append(epoch)
                    rows.append(
                        {
                            "participant": f"syn-{participant + 1:03d}",
                            "item": f"syn-item-{item + 1:03d}",
                            "condition": condition,
                            "treatment": treatment,
                            "trial_number": trial_number,
                            "n400_mean_voltage_uv": outcome,
                            "blink_present": blink_present,
                            "missing": False,
                            "seed": config.seed,
                            "data_status": "synthetic",
                        }
                    )
        stacked = np.stack(epochs)
        expected_effect = float(
            config.treatment_effect_uv
            * topography[roi_indices].mean()
            * waveform[window].mean()
        )
        return SyntheticDataset(
            times_s=times,
            channel_names=config.channels,
            epochs_uv=stacked,
            trials=pd.DataFrame.from_records(rows),
            expected_window_effect_uv=expected_effect,
        )


def _centro_parietal_topography(channels: tuple[str, ...]) -> np.ndarray:
    weights = {
        "Fz": 0.35,
        "FCz": 0.6,
        "Cz": 1.0,
        "CPz": 1.0,
        "Pz": 0.8,
    }
    return np.array([weights.get(channel, 0.55) for channel in channels])


def _ar1_noise(
    rng: np.random.Generator,
    *,
    channels: int,
    samples: int,
    rho: float,
    scale: float,
) -> np.ndarray:
    innovations = rng.normal(0, scale * np.sqrt(1 - rho**2), (channels, samples))
    noise = np.empty_like(innovations)
    noise[:, 0] = rng.normal(0, scale, channels)
    for index in range(1, samples):
        noise[:, index] = rho * noise[:, index - 1] + innovations[:, index]
    return noise
