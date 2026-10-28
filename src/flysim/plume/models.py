from __future__ import annotations

import numpy as np


class MeanderModel:
    """Classe base abstrata para modelos de meandro e flutuação transversal da pluma."""

    def __init__(self, main_config, model_config):
        self.main_config = main_config
        self.model_config = model_config

    def update(self, dt, current_time):
        raise NotImplementedError


class LinearMeanderModel(MeanderModel):
    """
    Modelo de meandro linear (sem oscilação transversal).
    A velocidade transversal vy é nula e a pluma alinha-se diretamente com o escoamento médio.
    """

    def __init__(self, main_config, model_config):
        super().__init__(main_config, model_config)
        self.meandering_vy = 0.0

    def update(self, dt, current_time):
        return 0.0


class OUMeanderModel(MeanderModel):
    """
    Modelo de meandro baseado no processo estocástico de Ornstein-Uhlenbeck (OU).
    Modela flutuações contínuas com reversão à média e dispersão browniana.
    """

    def __init__(self, main_config, model_config):
        super().__init__(main_config, model_config)
        self.meandering_vy = 0.0
        timescale = self.model_config.get("timescale", 1.0)
        intensity = self.model_config.get("intensity", 0.2)
        self.ou_theta = 1.0 / timescale if timescale > 0 else 0
        self.ou_sigma = intensity

    def update(self, dt, current_time):
        """Atualiza a velocidade transversal vy através da equação diferencial estocástica de OU."""
        if self.ou_theta > 0:
            drift = -self.ou_theta * self.meandering_vy * dt
            diffusion = self.ou_sigma * np.sqrt(dt) * np.random.randn()
            self.meandering_vy += drift + diffusion
        return self.meandering_vy


class SinusoidMeanderModel(MeanderModel):
    """
    Modelo de meandro sinusoidal harmónico.
    Produz uma oscilação ondulatória transversal periódica que simula esteiras
    e correntes sinuosas típicas de escoamentos marinhos e fluviais.
    """

    def __init__(self, main_config, model_config):
        super().__init__(main_config, model_config)

        self.num_harmonics = self.model_config.get("num_harmonics", 10)
        self.v_var = self.model_config.get("v_var", 0.25)
        self.L = self.model_config.get("integral_length_scale_L", 20.0)
        self.meandering_vy = 0.0

        drift_timescale = self.model_config.get("drift_correction_timescale", 45.0)

        if drift_timescale > 0:
            self.drift_theta = 1.0 / drift_timescale
        else:
            self.drift_theta = 0.0

        wind_speed = np.linalg.norm(self.main_config["mean_wind_velocity"])
        if wind_speed < 1e-6:
            self.amplitudes = np.zeros(self.num_harmonics)
            self.freqs = np.ones(self.num_harmonics)
            self.phases = np.zeros(self.num_harmonics)
            return

        low_freq = wind_speed / self.L
        high_freq = 100 * low_freq
        self.freqs = np.logspace(np.log10(low_freq), np.log10(high_freq), self.num_harmonics)

        wavenumbers = 2 * np.pi * self.freqs / wind_speed
        energy_spectrum = (self.v_var * (2 * self.L / np.pi)) / (
            1 + (wavenumbers * self.L) ** 2
        ) ** (5 / 6)

        freq_spectrum = energy_spectrum * (2 * np.pi / wind_speed)
        df = np.gradient(self.freqs)
        self.amplitudes = np.sqrt(2 * freq_spectrum * df)
        self.phases = np.random.rand(self.num_harmonics) * 2 * np.pi

    def update(self, dt, current_time):
        """
        Calcula a velocidade transversal instantânea física do fluido (dy/dt da esteira sinusoidal).
        A derivada temporal de y(t) = A * sin(2*pi*f*t) é dy/dt = A * 2*pi*f * cos(2*pi*f*t).
        """
        amplitude = self.model_config.get("amplitude", self.model_config.get("v_var", 8.0))
        wind_speed = np.linalg.norm(self.main_config.get("mean_wind_velocity", [0.4, 0.0]))
        length_scale = self.model_config.get("integral_length_scale_L", 22.0)
        default_freq = (wind_speed / length_scale) if length_scale > 0 else 0.018
        frequency = self.model_config.get("frequency", default_freq)
        two_pi_f = 2.0 * np.pi * frequency
        self.meandering_vy = amplitude * two_pi_f * np.cos(two_pi_f * current_time)
        return self.meandering_vy
