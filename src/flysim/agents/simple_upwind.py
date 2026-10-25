from __future__ import annotations

import numpy as np

from flysim.agents.base import Agent


class SimpleUpwindAgent(Agent):
    """A simple agent for basic testing."""

    def __init__(self, config):
        self.internal_state, self.cast_direction, self.last_hit_pos = None, None, None
        super().__init__(config)

    def reset(self):
        super().reset()
        self.internal_state, self.cast_direction, self.last_hit_pos = "casting", 1, None

    def run_search_algorithm(self, dt):
        concentration_threshold = self.config.get("agent_concentration_threshold", 1.0)
        surge_len = self.config.get("surge_length", 5.0)
        cast_len = self.config.get("cast_length", 8.0)

        custom_data = {"internal_state": self.internal_state}
        if self.current_concentration > concentration_threshold:
            self.internal_state = "surging"
            self.last_hit_pos = self.pos.copy()
            upwind_vec = -np.array(self.config["mean_wind_velocity"])
            target_pos = self.pos + upwind_vec * surge_len
        else:
            if self.internal_state == "surging":
                self.internal_state, self.cast_direction = "casting", self.cast_direction * -1
            crosswind_vec = np.array(
                [
                    -self.config["mean_wind_velocity"][1],
                    self.config["mean_wind_velocity"][0],
                ]
            )
            target_pos = self.pos + (crosswind_vec * self.cast_direction * cast_len)
        return target_pos, custom_data
