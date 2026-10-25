from __future__ import annotations

import numpy as np

from flysim.agents.base import Agent


class BioInspiredAgent(Agent):
    """
    Agente com estratégia de busca bio-inspirada (Cast-and-Surge com movimento em traça).
    Implementa histerese de decisão sensorial, busca transversal progressiva ao perder o sinal
    e retorno de emergência à posição de partida se a bateria descer abaixo de 15%.
    """

    def __init__(self, config):
        self.main_config = config
        super().__init__(config)

    def reset(self):
        """Reinicializa as variáveis de navegação e controlo comportamental do agente."""
        super().reset()
        self.state, self.behavior, self.internal_state = "SEARCHING", "SEARCHING", "DECIDING"
        self.target_pos, self.cast_direction, self.search_direction = None, 1, 1
        self.consecutive_hits, self.consecutive_misses = 0, 0
        self.current_cast_length = self.config.get("cast_length", 25.0)

    def _compute_zig_zag_target(self, length, angle_deg, direction_sign):
        """Calcula o vetor de deslocamento em ziguezague relativamente à corrente contra o vento."""
        flow_norm = np.linalg.norm(self.measured_flow_direction)
        if flow_norm < 1e-6:
            return self.pos
        upwind_dir = -self.measured_flow_direction / flow_norm
        angle_rad = np.deg2rad(angle_deg * direction_sign)
        rot_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
        return self.pos + (rot_matrix @ upwind_dir) * length

    def _set_search_target(self):
        print(f"BEHAVIOR: Searching (direction: {self.search_direction})")
        self.behavior = "SEARCHING"
        potential_target = self._compute_zig_zag_target(
            self.config["search_length"],
            self.config["search_angle"],
            self.search_direction,
        )
        if not (2.0 < potential_target[1] < self.main_config["world_height"] - 2.0):
            print("  > Search target out of bounds. Inverting direction.")
            self.search_direction *= -1
            self.target_pos = self._compute_zig_zag_target(
                self.config["search_length"],
                self.config["search_angle"],
                self.search_direction,
            )
        else:
            self.target_pos = potential_target
        if self.target_pos is not None:
            self.target_pos[0] = np.clip(
                self.target_pos[0], 2.0, self.main_config["world_width"] - 2.0
            )
            self.target_pos[1] = np.clip(
                self.target_pos[1], 2.0, self.main_config["world_height"] - 2.0
            )
        self.internal_state = "MOVING_TO_TARGET"

    def _set_cast_target(self):
        print(
            f"BEHAVIOR: Casting (direction: {self.cast_direction}, length: {self.current_cast_length:.2f})"
        )
        self.behavior = "CASTING"
        potential_target = self._compute_zig_zag_target(
            self.current_cast_length, self.config["cast_angle"], self.cast_direction
        )
        if not (2.0 < potential_target[1] < self.main_config["world_height"] - 2.0):
            print("  > Cast target out of bounds. Inverting direction.")
            self.cast_direction *= -1
            self.target_pos = self._compute_zig_zag_target(
                self.current_cast_length, self.config["cast_angle"], self.cast_direction
            )
        else:
            self.target_pos = potential_target
        if self.target_pos is not None:
            self.target_pos[0] = np.clip(
                self.target_pos[0], 2.0, self.main_config["world_width"] - 2.0
            )
            self.target_pos[1] = np.clip(
                self.target_pos[1], 2.0, self.main_config["world_height"] - 2.0
            )
        self.internal_state = "MOVING_TO_TARGET"

    def _set_surge_target(self):
        print("BEHAVIOR: Surging")
        self.behavior = "SURGING"
        self.current_cast_length = self.config.get("cast_length", 25.0)
        flow_norm = np.linalg.norm(self.measured_flow_direction)
        if flow_norm > 1e-6:
            upwind_dir = -self.measured_flow_direction / flow_norm
            self.target_pos = self.pos + upwind_dir * self.config["surge_length"]
        self.internal_state = "MOVING_TO_TARGET"

    def run_search_algorithm(self, dt):
        is_hit = self.current_concentration > self.config["agent_concentration_threshold"]
        if is_hit:
            self.consecutive_hits, self.consecutive_misses = (
                self.consecutive_hits + 1,
                0,
            )
        else:
            self.consecutive_misses, self.consecutive_hits = (
                self.consecutive_misses + 1,
                0,
            )
        confirmation_steps = self.config.get("confirmation_steps", 1)

        if self.state == "SEARCHING" and self.consecutive_hits >= confirmation_steps:
            print(
                f"PLUME FOUND ({self.consecutive_hits} consecutive hits)! Switching to TRACKING state."
            )
            self.state, self.internal_state = "TRACKING", "DECIDING"
        elif (
            self.state == "TRACKING"
            and self.behavior == "SURGING"
            and self.consecutive_misses >= confirmation_steps
        ):
            print(f"PLUME LOST ({self.consecutive_misses} consecutive misses)! Switching to CAST.")
            self.internal_state = "DECIDING"
        elif (
            self.state == "TRACKING"
            and self.behavior == "CASTING"
            and self.consecutive_hits >= confirmation_steps
        ):
            print(
                f"PLUME RE-ACQUIRED ({self.consecutive_hits} consecutive hits)! Switching to SURGE."
            )
            self.internal_state = "DECIDING"

        if self.internal_state == "DECIDING":
            if self.state == "SEARCHING":
                self._set_search_target()
            elif self.state == "TRACKING":
                if self.consecutive_hits > 0:
                    self._set_surge_target()
                else:
                    if self.behavior == "CASTING":
                        self.current_cast_length *= 1.0 + self.config.get(
                            "cast_length_increase_factor", 0.1
                        )
                        print(
                            f"  > Cast failed. Increasing cast length to {self.current_cast_length:.2f}"
                        )
                        self.cast_direction *= -1
                    self._set_cast_target()
        elif self.internal_state == "MOVING_TO_TARGET":
            hit_wall = (
                self.pos[1] <= 2.2
                or self.pos[1] >= self.main_config["world_height"] - 2.2
                or self.pos[0] <= 2.2
                or self.pos[0] >= self.main_config["world_width"] - 2.2
            )
            if (
                self.target_pos is None
                or np.linalg.norm(self.pos - self.target_pos) < 0.8
                or hit_wall
            ):
                self.internal_state, self.target_pos = "DECIDING", None

        custom_data = {
            "state": self.state,
            "behavior": self.behavior,
            "hits": self.consecutive_hits,
            "misses": self.consecutive_misses,
        }
        return self.target_pos, custom_data
