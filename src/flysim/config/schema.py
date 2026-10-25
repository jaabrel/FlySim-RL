from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, list


@dataclass
class AgentConfig:
    agent_radius: float = 0.5
    agent_speed: float = 2.4
    success_distance: float = 2.0
    agent_concentration_threshold: float = 1.0
    dist_to_detection: float = 5.0
    surge_length: float = 4.0
    cast_length: float = 26.0
    cast_angle: float = 75.0
    search_length: float = 26.0
    search_angle: float = 75.0
    cast_length_increase_factor: float = 0.1
    confirmation_steps: int = 3


@dataclass
class OUMeanderConfig:
    intensity: float = 0.4
    timescale: float = 0.15


@dataclass
class SinusoidMeanderConfig:
    num_harmonics: int = 10
    v_var: float = 8.0
    amplitude: float = 8.0
    frequency: float = 0.018
    integral_length_scale_L: float = 22.0
    puff_expansion_rate: float = 0.29
    turbulence_intensity: float = 0.18
    drift_correction_timescale: float = 10.0


@dataclass
class SimConfig:
    dt: float = 0.1
    max_time: float = 300.0
    save_path: str = "experiments"
    experiment_name: str = "bio_inspired_test"
    visualize: bool = True
    real_time_pacing: bool = False
    viz_width: int = 1200
    viz_height: int = 600
    world_width: float = 100.0
    world_height: float = 50.0
    source_position: list[float] = field(default_factory=lambda: [20.0, 25.0])
    emission_rate: int = 10
    initial_puff_mass: float = 400.0
    puff_decay_rate: float = 0.02
    puff_expansion_rate: float = 0.25
    concentration_model: str = "gaussian"
    num_experiments: int = 5
    plume_stabilization_time: float = 0.0
    agent_initial_wait_time: float = 8.0
    agent_start_region_center: list[float] = field(default_factory=lambda: [90.0, 25.0])
    agent_start_region_radius: float = 2.0
    meander_model: str = "sum_of_sinusoids"
    ou_meander_config: OUMeanderConfig = field(default_factory=OUMeanderConfig)
    sinusoid_meander_config: SinusoidMeanderConfig = field(default_factory=SinusoidMeanderConfig)
    turbulence_intensity: float = 0.15
    mean_wind_velocity: list[float] = field(default_factory=lambda: [0.4, 0.0])
    agent_class: str = "QLearningAgent"
    battery_capacity: float = 100.0
    battery_base_decay: float = 0.08
    battery_wind_factor: float = 0.8
    battery_min_decay: float = 0.02
    agent_config: AgentConfig = field(default_factory=AgentConfig)


def load_config(path: str | Path = "config.json") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "_descricao_parametros_pt" in data:
        data.pop("_descricao_parametros_pt")
    return data
