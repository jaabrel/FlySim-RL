from __future__ import annotations

import numpy as np

from flysim.plume.kernels import (
    update_puffs_numba,
    update_puffs_sinusoid_numba,
)
from flysim.plume.models import (
    LinearMeanderModel,
    OUMeanderModel,
    SinusoidMeanderModel,
)


class FilamentPlume:
    """
    Modelo lagrangeano de dispersão de filamentos de pluma no meio fluido.
    Gere a emissão periódica, expansão volumétrica, decaimento químico e
    meandro transversal dos pacotes de odor (puffs).
    """

    def __init__(self, config):
        self.config = config
        self.source_pos = np.array(config["source_position"])

        # Inicialização do array estático pré-alocado para máxima performance:
        # Coluna 0: Posição X em metros
        # Coluna 1: Posição Y em metros
        # Coluna 2: Raio do pacote (r) em metros
        # Coluna 3: Massa ativa do pacote (0.0 se inativo)
        # Coluna 4: Idade do pacote em segundos
        self.max_puffs = self.config.get("max_puffs", 4000)
        self.puffs_array = np.zeros((self.max_puffs, 5), dtype=np.float32)
        self.time_since_last_puff = 0.0

        model_name = self.config.get("meander_model", "ou").lower()
        if model_name in ["sum_of_sinusoids", "sinusoid", "sinusoidal"]:
            model_config = self.config.get("sinusoid_meander_config", {})
            self.meander_generator = SinusoidMeanderModel(self.config, model_config)
        elif model_name == "linear":
            self.meander_generator = LinearMeanderModel(self.config, {})
        else:
            model_config = self.config.get("ou_meander_config", {})
            self.meander_generator = OUMeanderModel(self.config, model_config)

    def update(self, dt, current_time):
        meandering_vy = self.meander_generator.update(dt, current_time)
        self.time_since_last_puff += dt
        release_interval = 1.0 / self.config["emission_rate"]
        puffs_to_release = int(self.time_since_last_puff / release_interval)
        min_mass = self.config["initial_puff_mass"] * 0.01

        if puffs_to_release > 0:
            # Encontrar índices de slots inativos (massa <= min_mass)
            inactive_indices = np.where(self.puffs_array[:, 3] <= min_mass)[0]

            # Fallback de crescimento dinâmico se a capacidade for excedida
            if puffs_to_release > len(inactive_indices):
                print(
                    f"Warning: Plume puffs array capacity exceeded. Resizing dynamically from {self.max_puffs} to {self.max_puffs * 2}."
                )
                old_puffs = self.puffs_array.copy()
                self.max_puffs *= 2
                self.puffs_array = np.zeros((self.max_puffs, 5), dtype=np.float32)
                self.puffs_array[: old_puffs.shape[0]] = old_puffs
                inactive_indices = np.where(self.puffs_array[:, 3] <= min_mass)[0]

            slots_to_use = inactive_indices[:puffs_to_release]

            if len(slots_to_use) > 0:
                self.puffs_array[slots_to_use, 0] = self.source_pos[0]
                self.puffs_array[slots_to_use, 1] = self.source_pos[1]
                self.puffs_array[slots_to_use, 2] = 0.0
                self.puffs_array[slots_to_use, 3] = self.config["initial_puff_mass"]
                self.puffs_array[slots_to_use, 4] = 0.0

            self.time_since_last_puff = 0.0

        if self.puffs_array.shape[0] > 0:
            is_sinusoid = self.config.get("meander_model", "ou").lower() in [
                "sum_of_sinusoids",
                "sinusoid",
                "sinusoidal",
            ]
            if is_sinusoid:
                sin_cfg = self.config.get("sinusoid_meander_config", {})
                amplitude = float(sin_cfg.get("amplitude", sin_cfg.get("v_var", 8.0)))
                frequency = float(sin_cfg.get("frequency", 0.018))
                expansion_rate = float(
                    sin_cfg.get("puff_expansion_rate", self.config.get("puff_expansion_rate", 0.29))
                )
                turbulence = float(
                    sin_cfg.get(
                        "turbulence_intensity", self.config.get("turbulence_intensity", 0.18)
                    )
                )

                self.puffs_array = update_puffs_sinusoid_numba(
                    self.puffs_array,
                    dt,
                    current_time,
                    float(self.source_pos[0]),
                    float(self.source_pos[1]),
                    np.array(self.config["mean_wind_velocity"], dtype=np.float32),
                    amplitude,
                    frequency,
                    turbulence,
                    float(self.config["puff_decay_rate"]),
                    expansion_rate,
                    min_mass,
                )
            else:
                self.puffs_array = update_puffs_numba(
                    self.puffs_array,
                    dt,
                    np.array(self.config["mean_wind_velocity"]),
                    meandering_vy,
                    self.config["turbulence_intensity"],
                    self.config["puff_decay_rate"],
                    self.config.get("puff_expansion_rate", 0.25),
                    min_mass,
                )
            # Atualizar inatividade no próprio array (sem realocação de memória)
            dead_mask = (self.puffs_array[:, 3] <= min_mass) | (
                self.puffs_array[:, 4] >= self.config["max_time"]
            )
            self.puffs_array[dead_mask, 3] = 0.0
