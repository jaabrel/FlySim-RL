from __future__ import annotations

import numpy as np

from flysim.plume.kernels import (
    calculate_concentration_gaussian_numba,
    calculate_concentration_numba,
)
from flysim.plume.models import (
    OUMeanderModel,
    SinusoidMeanderModel,
)


class Agent:
    """
    Classe base para agentes de navegação autónoma subaquática (AUV).
    Modela as propriedades cinemáticas, consumo de bateria sob esforço de corrente
    e medições sensoriais de fluxo e concentração química.
    """

    def __init__(self, config):
        self.config = config
        self.radius = self.config["agent_radius"]
        self.speed = self.config["agent_speed"]
        self.measured_flow_direction = np.array([1.0, 0.0])
        self.battery_capacity = self.config.get("battery_capacity", 100.0)
        self.battery_base_decay = self.config.get("battery_base_decay", 0.08)
        self.battery_wind_factor = self.config.get("battery_wind_factor", 0.80)
        self.battery_min_decay = self.config.get("battery_min_decay", 0.02)
        self.reset()

    def reset(self):
        """Reinicializa as variáveis de estado do veículo para um novo ensaio."""
        start_y = float(
            np.random.uniform(
                0.30 * self.config["world_height"], 0.70 * self.config["world_height"]
            )
        )
        self.config["agent_start_region_center"] = (90.0, start_y)
        self.pos = np.array([90.0, start_y], dtype=np.float64)
        self.start_pos = self.pos.copy()
        self.direction = np.array([1.0, 0.0])
        self.current_concentration = 0.0
        self.measured_flow_direction = np.array([1.0, 0.0])
        if hasattr(self, "internal_state"):
            self.internal_state = "casting"

        self.state = "waiting"
        self.first_detection_pos = None
        self.mission_complete = False
        self.circling_step = 0
        self.circling_angle = 0.0
        self.circling_completed = False
        self.source_reached = False
        self.prev_dist_to_detection = None
        self.skirting_side_ida = None
        self.battery = self.battery_capacity
        self.return_waypoint_reached = False
        self.low_battery_direct_return = False
        self.low_battery_start_return = False
        self.low_battery_aborted = False
        self.incomplete_mission = False
        self.theta = 0.0
        self.return_contour_completed = False
        self.return_tip_reached = False
        self.distal_tip_entered = False
        self.return_side_ida_reached = False
        self.return_at_detection_x = False
        self.pos_history = []

    def get_circling_target(self, dt):
        """
        Calcula o ponto-alvo para manobra de contorno circular suave em redor da fonte (360°),
        antes de iniciar o estado de retorno à posição da primeira deteção.
        """
        source_pos = np.array(self.config["source_position"])
        radius = self.config.get("agent_config", {}).get("success_distance", 2.0) * 1.0
        ida_side = self.skirting_side_ida if self.skirting_side_ida is not None else 1.0

        if self.circling_step == 0:
            self.circling_angle = np.pi / 2.0 if ida_side == 1.0 else 3.0 * np.pi / 2.0

        # Update angle smoothly at physical speed (v/R * dt)
        angular_speed = self.speed / radius
        if ida_side == 1.0:
            self.circling_angle += angular_speed * dt
            if self.circling_angle >= 3.0 * np.pi / 2.0:
                self.state = "returning"
                self.source_reached = False
                self.circling_completed = True
                self.return_contour_completed = False
                self.return_tip_reached = False
                self.return_side_ida_reached = False
                self.return_at_detection_x = False
                self.theta = None
                exit_pos = np.array(
                    [source_pos[0] + 0.8, source_pos[1] - ida_side * 0.8], dtype=np.float64
                )
                self.pos = exit_pos.copy()
                print(
                    f"\n[Agent] Circling complete! State changed to RETURNING on opposite margin ({exit_pos[0]:.1f}, {exit_pos[1]:.1f}). Goal: {self.first_detection_pos}"
                )
                return exit_pos
        else:
            self.circling_angle -= angular_speed * dt
            if self.circling_angle <= np.pi / 2.0:
                self.state = "returning"
                self.source_reached = False
                self.circling_completed = True
                self.return_contour_completed = False
                self.return_tip_reached = False
                self.return_side_ida_reached = False
                self.return_at_detection_x = False
                self.theta = None
                exit_pos = np.array(
                    [source_pos[0] + 0.8, source_pos[1] - ida_side * 0.8], dtype=np.float64
                )
                self.pos = exit_pos.copy()
                print(
                    f"\n[Agent] Circling complete! State changed to RETURNING on opposite margin ({exit_pos[0]:.1f}, {exit_pos[1]:.1f}). Goal: {self.first_detection_pos}"
                )
                return exit_pos

        target = source_pos + np.array(
            [radius * np.cos(self.circling_angle), radius * np.sin(self.circling_angle)]
        )
        self.circling_step += 1
        return target

    def measure_flow(self, plume):
        mean_wind = np.array(self.config["mean_wind_velocity"])

        if isinstance(plume.meander_generator, (OUMeanderModel, SinusoidMeanderModel)):
            vy = plume.meander_generator.meandering_vy
            self.measured_flow_direction = mean_wind + np.array([0.0, vy])
        else:
            self.measured_flow_direction = mean_wind.copy()

        return self.measured_flow_direction

    def measure_concentration(self, plume):
        model = self.config.get("concentration_model", "top-hat")
        if model == "gaussian":
            self.current_concentration = calculate_concentration_gaussian_numba(
                self.pos, plume.puffs_array
            )
        else:
            self.current_concentration = calculate_concentration_numba(
                self.pos, self.radius, plume.puffs_array
            )
        return self.current_concentration

    def move(self, target_pos, dt):
        if target_pos is None:
            return
        world_w, world_h = self.config["world_width"], self.config["world_height"]
        valid_target_pos = np.array(
            [np.clip(target_pos[0], 0, world_w), np.clip(target_pos[1], 0, world_h)]
        )
        direction_vec = valid_target_pos - self.pos
        distance = np.linalg.norm(direction_vec)
        if distance > 1e-6:
            self.direction = direction_vec / distance
            displacement = self.direction * self.speed * dt
            self.pos = (
                valid_target_pos
                if np.linalg.norm(displacement) > distance
                else self.pos + displacement
            )

    def check_success(self, source_pos):
        return np.linalg.norm(self.pos - source_pos) < self.config.get("success_distance", 1.0)

    def run_search_algorithm(self, dt):
        raise NotImplementedError("This method should be implemented by a subclass.")
