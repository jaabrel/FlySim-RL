from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from flysim.plume.filament import FilamentPlume
from flysim.plume.kernels import (
    calculate_concentration_gaussian_numba,
    get_plume_center_y,
)


class PlumeRLWrapper(gym.Env):
    """
    Ambiente de simulação compatível com a API Gymnasium para treino por Reforço.
    Encapsula a física da dispersão lagrangeana, dinâmica de bateria do AUV e
    calcula as funções de recompensa para orientação e localização da fonte.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config):
        """
        Inicializa o ambiente de treino, define os espaços de ação discreta (4 direções)
        e os espaços de observação contínua normalizados (8 dimensões).
        """
        super().__init__()
        self.config = config

        # Ações discretas: 0=Cima, 1=Baixo, 2=Esquerda, 3=Direita
        self.action_space = spaces.Discrete(4)

        # Vetor de observação com 8 variáveis normalizadas:
        # [x, y, concentração, fluxo_x, fluxo_y, estado_missão, nível_bateria, sentido_varrimento]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, -1, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 3, 1, 1], dtype=np.float32),
            shape=(8,),
            dtype=np.float32,
        )
        self.reward_config = {
            "step_penalty": -0.05,
            "concentration_improvement": 1.0,
            "concentration_decrease": -0.05,
            "revisit_threshold": 1.0,
            "revisit_penalty": -0.2,
            "plume_detection_bonus": 0.3,
            "upwind_bonus": 1.0,
            "goal_reward": 500.0,
            "battery_drain_penalty_factor": 0.5,
            "battery_depleted_penalty": -200.0,
        }

        self.conc_max = 10.0
        self.flow_max = 5.0

        self.battery_capacity = self.config.get("battery_capacity", 100.0)
        self.battery = self.battery_capacity
        self.last_drain = 0.0

        # Rastreio de variáveis de estado e máquina de estados
        self.speed = self.config["agent_config"]["agent_speed"]
        self.plume = None
        self.pos = None
        self.time = 0.0
        self.visit_map = {}
        self.current_cell = None
        self.steps_in_current_cell = 0
        self.prev_pos = None
        self.prev_conc = 0.0
        self.current_conc = 0.0
        self.state = "searching"
        self.cast_dir = 0
        self.first_detection_pos = None
        self.mission_complete = False
        self.source_reached = False
        self.circling_step = 0
        self.circling_angle = 0.0
        self.circling_completed = False
        self.return_tip_reached = False
        self.return_side_ida_reached = False
        self.return_at_detection_x = False
        self.return_contour_completed = False
        self.skirting_side_ida = None
        self.prev_dist_to_detection = None
        self.low_battery_return = False
        self.return_to_start = False
        self.start_pos = None
        self.searching_consecutive_straight_steps = 0
        self.prev_dist_to_source = None
        self.min_dist_to_source = float("inf")
        self.prev_dist_to_sweep = None
        self.recent_positions = []
        self.consecutive_loop_steps = 0

    def _validate_config(self):
        """Valida a consistência dos parâmetros físicos configurados no ambiente."""
        assert self.config["world_width"] > 0, "world_width deve ser positivo"
        assert self.config["world_height"] > 0, "world_height deve ser positivo"
        assert self.config["dt"] > 0 and self.config["dt"] < 1, (
            "dt deve situar-se no intervalo (0, 1)"
        )
        assert self.config["agent_config"]["agent_speed"] > 0, "agent_speed deve ser positivo"
        assert self.config["emission_rate"] > 0, "emission_rate must be positive"
        assert self.config["max_time"] > 0, "max_time must be positive"

    def reset(self, seed=None, options=None):
        """
        Reset environment for new episode
        """
        super().reset(seed=seed)

        # Validate config
        self._validate_config()

        start_y = float(
            np.random.uniform(
                0.30 * self.config["world_height"], 0.70 * self.config["world_height"]
            )
        )
        self.config["agent_start_region_center"] = (90.0, start_y)

        self.plume = FilamentPlume(self.config)
        self.time = 0.0

        # Initialize agent position: x = 90, y random between 10 and 40
        self.pos = np.array([90.0, start_y], dtype=np.float64)
        self.start_pos = self.pos.copy()
        self.prev_pos = self.pos.copy()

        # Reset tracking
        self.visit_map = {}
        self.current_cell = None
        self.steps_in_current_cell = 0
        self.prev_conc = 0.0
        self.current_conc = 0.0
        self.first_detection_pos = None
        self.has_moved_away = False
        self.path_length = 0.0

        # Propriedades da máquina de estados da missão
        self.state = "searching"
        self.first_detection_pos = None
        self.mission_complete = False
        self.source_reached = False
        self.circling_step = 0
        self.circling_angle = 0.0
        self.circling_completed = False
        self.return_tip_reached = False
        self.distal_tip_entered = False
        self.return_side_ida_reached = False
        self.return_at_detection_x = False
        self.return_contour_completed = False
        self.low_battery_aborted = False
        self.prev_action = None
        self.skirting_side_ida = None
        self.prev_dist_to_detection = None
        self.prev_dist_to_source = None
        self.min_dist_to_source = float("inf")
        self.prev_dist_to_sweep = None
        self.low_battery_return = False
        self.return_to_start = False
        self.battery = self.battery_capacity
        self.last_drain = 0.0
        self.searching_consecutive_straight_steps = 0
        self.cast_dir = int(np.random.choice([0, 1]))
        self.recent_positions = []
        self.consecutive_loop_steps = 0
        self.max_x_returning = 20.0
        self.min_x_returning_back = 75.0
        self.min_dist_to_det = 70.0

        # Esperar 8 segundos no início de cada episódio antes de iniciar o movimento
        initial_wait_time = self.config.get("agent_initial_wait_time", 8.0)
        for _ in range(int(initial_wait_time / self.config["dt"])):
            self.plume.update(self.config["dt"], self.time)
            self.time += self.config["dt"]

        return self._get_obs(), {"at_goal": False}

    def step(self, action):
        """
        Execute one step in environment
        """
        dt = self.config["dt"]
        self.prev_pos = self.pos.copy()

        if self.state == "searching":
            search_angle_deg = float(self.config.get("agent_config", {}).get("search_angle", 75.0))
            crab_upwind = float(np.cos(np.deg2rad(search_angle_deg)))  # ~0.2588 contra o vento (-x)
            crosswind = float(np.sin(np.deg2rad(search_angle_deg)))  # ~0.9659 no eixo vertical y
            dirs = {
                0: np.array(
                    [-crab_upwind, -crosswind]
                ),  # Baixo (-y) com avanço contra o vento (-x)
                1: np.array([-crab_upwind, crosswind]),  # Cima (+y) com avanço contra o vento (-x)
                2: np.array([-1.0, 0.0]),  # Esquerda direto contra o vento (-x)
                3: np.array([1.0, 0.0]),  # Direita a favor do vento (+x)
            }
        else:
            dirs = {
                0: np.array([0.0, -1.0]),  # Baixo (-y, Sul)
                1: np.array([0.0, 1.0]),  # Cima (+y, Norte)
                2: np.array([-1.0, 0.0]),  # Esquerda (-x, Oeste)
                3: np.array([1.0, 0.0]),  # Direita (+x, Este)
            }

        # --- High-level state transitions in training env ---
        conc_thresh = self.config["agent_config"]["agent_concentration_threshold"]
        source_pos = np.array(self.config["source_position"])
        dist_to_source = np.linalg.norm(self.pos - source_pos)
        success_distance = self.config["agent_config"]["success_distance"]

        # Definição do decay-rate da bateria dependendendo do vento
        direction = dirs[action]
        vy = self.plume.meander_generator.meandering_vy
        flow_vec = np.array(self.config["mean_wind_velocity"]) + np.array([0.0, vy])
        opposing_wind = -flow_vec
        wind_resistance = np.dot(direction, opposing_wind)

        base_decay = self.config.get("battery_base_decay", 0.08)
        wind_factor = self.config.get("battery_wind_factor", 0.80)
        min_decay = self.config.get("battery_min_decay", 0.02)

        drain_rate = max(min_decay, base_decay + wind_factor * wind_resistance)
        self.last_drain = drain_rate * dt
        self.battery = max(0.0, self.battery - self.last_drain)
        self.current_action = action

        if self.battery <= 0:
            terminated = True
            truncated = False
            reward = self.reward_config.get("battery_depleted_penalty", -200.0)
            self.prev_action = action
            return (
                self._get_obs(),
                reward,
                terminated,
                truncated,
                {"at_goal": False, "battery_depleted": True},
            )

        # Verificação universal de segurança de bateria baixa (<= 15%):
        battery_pct = (self.battery / self.battery_capacity) * 100.0
        if battery_pct <= 15.0:
            self.low_battery_return = True
            self.low_battery_aborted = True
            if self.first_detection_pos is None:
                self.return_to_start = True
                start_target = getattr(self, "start_pos", np.array([90.0, 25.0]))
                self.prev_dist_to_detection = np.linalg.norm(self.pos - start_target)
            else:
                self.prev_dist_to_detection = np.linalg.norm(self.pos - self.first_detection_pos)

        if self.state == "searching":
            if action == 2:  # Ação 2: Esquerda (avanço direto contra o vento)
                self.searching_consecutive_straight_steps += 1
            else:
                self.searching_consecutive_straight_steps = 0

            # Limites de varrimento em traça (cobre entre 30% e 70% da altura do mundo)
            y_min_sweep = 0.30 * self.config["world_height"]
            y_max_sweep = 0.70 * self.config["world_height"]

            if self.pos[1] >= y_max_sweep:
                if self.cast_dir == 0:
                    self.cast_dir = 1
                    self.prev_dist_to_sweep = None
            elif self.pos[1] <= y_min_sweep:
                if self.cast_dir == 1:
                    self.cast_dir = 0
                    self.prev_dist_to_sweep = None

            self.current_conc = calculate_concentration_gaussian_numba(
                self.pos, self.plume.puffs_array
            )
            if self.current_conc > conc_thresh:
                self.state = "following_boundary"
                self.first_detection_pos = self.pos.copy()
                local_center_y = get_plume_center_y(self.pos[0], self.time, self.plume, self.config)
                self.skirting_side_ida = 1.0 if self.pos[1] >= local_center_y else -1.0
                self.prev_dist_to_source = dist_to_source
                self.min_dist_to_source = dist_to_source
            elif battery_pct <= 15.0:
                self.state = "returning"

        elif self.state == "following_boundary":
            if (
                dist_to_source < success_distance
                or (self.pos[0] <= source_pos[0] + 1.8 and dist_to_source < success_distance * 1.8)
            ) and not self.source_reached:
                self.source_reached = True
                self.circling_step = 0

            if not self.source_reached and battery_pct <= 15.0:
                self.state = "returning"

        # --- Circling source is handled programmatically as part of following_boundary ---
        if self.state == "following_boundary" and self.source_reached:
            radius = success_distance * 1.0
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
                    self.return_tip_reached = False
                    self.return_side_ida_reached = False
                    self.return_at_detection_x = False
                    self.return_contour_completed = False
                    self.max_x_returning = self.pos[0]
                    self.min_x_returning_back = 75.0
                    if self.first_detection_pos is None:
                        self.first_detection_pos = self.pos.copy()
                    self.pos = np.array(
                        [source_pos[0] + 0.8, source_pos[1] - ida_side * 0.8], dtype=np.float64
                    )
                    self.prev_dist_to_detection = np.linalg.norm(
                        self.pos - self.first_detection_pos
                    )
                    self.min_dist_to_det = self.prev_dist_to_detection
                    self.recent_positions = []
                    self.current_cell = None
                    self.steps_in_current_cell = 0
            else:
                self.circling_angle -= angular_speed * dt
                if self.circling_angle <= np.pi / 2.0:
                    self.state = "returning"
                    self.source_reached = False
                    self.circling_completed = True
                    self.return_tip_reached = False
                    self.return_side_ida_reached = False
                    self.return_at_detection_x = False
                    self.return_contour_completed = False
                    self.max_x_returning = self.pos[0]
                    self.min_x_returning_back = 75.0
                    if self.first_detection_pos is None:
                        self.first_detection_pos = self.pos.copy()
                    self.pos = np.array(
                        [source_pos[0] + 0.8, source_pos[1] - ida_side * 0.8], dtype=np.float64
                    )
                    self.prev_dist_to_detection = np.linalg.norm(
                        self.pos - self.first_detection_pos
                    )
                    self.min_dist_to_det = self.prev_dist_to_detection
                    self.recent_positions = []
                    self.current_cell = None
                    self.steps_in_current_cell = 0

            if not self.circling_completed:
                self.pos = source_pos + np.array(
                    [radius * np.cos(self.circling_angle), radius * np.sin(self.circling_angle)]
                )
            self.circling_step += 1

            # Move plume and increment time
            self.plume.update(dt, self.time)
            self.time += dt
            self.current_conc = calculate_concentration_gaussian_numba(
                self.pos, self.plume.puffs_array
            )
            obs = self._get_obs()
            reward = 2.0  # constant small reward for circling
            reward -= self.reward_config.get("battery_drain_penalty_factor", 0.5) * self.last_drain
            terminated = False
            truncated = self.time >= self.config["max_time"]
            self.prev_action = action
            return obs, reward, terminated, truncated, {"at_goal": False}

        # --- Normal step movement (for searching, following, returning) ---
        target = self.pos + dirs[action] * self.config["agent_config"]["agent_speed"] * dt

        hit_boundary = False
        if (
            target[0] < 0.0
            or target[0] > self.config["world_width"]
            or target[1] < 0.0
            or target[1] > self.config["world_height"]
        ):
            hit_boundary = True

        self.pos[0] = np.clip(target[0], 0, self.config["world_width"])
        self.pos[1] = np.clip(target[1], 0, self.config["world_height"])

        if not hasattr(self, "recent_positions") or self.recent_positions is None:
            self.recent_positions = []
        self.recent_positions.append(self.pos.copy())
        if len(self.recent_positions) > 16:
            self.recent_positions.pop(0)

        # Update plume
        self.plume.update(dt, self.time)
        self.time += dt

        # Track cell visitation
        x_b = int(np.clip(self.pos[0] / self.config["world_width"] * 50, 0, 49))
        y_b = int(np.clip(self.pos[1] / self.config["world_height"] * 25, 0, 24))
        new_cell = (x_b, y_b)

        if new_cell != self.current_cell:
            self.visit_map[new_cell] = self.visit_map.get(new_cell, 0) + 1
            self.current_cell = new_cell
            self.steps_in_current_cell = 0
        else:
            self.steps_in_current_cell += 1

        visitas_nesta_celula = self.visit_map.get(new_cell, 1)

        self.prev_conc = self.current_conc
        self.current_conc = calculate_concentration_gaussian_numba(self.pos, self.plume.puffs_array)

        # Verificação de conclusão de objetivo / término
        if self.state == "returning":
            if getattr(self, "return_to_start", False):
                det_pos = getattr(self, "start_pos", np.array([90.0, 25.0]))
            else:
                det_pos = (
                    self.first_detection_pos
                    if self.first_detection_pos is not None
                    else getattr(self, "start_pos", self.pos)
                )
            dist_to_detection = np.linalg.norm(self.pos - det_pos)

            battery_pct = (self.battery / self.battery_capacity) * 100.0
            is_low_battery = (
                getattr(self, "low_battery_return", False)
                or getattr(self, "low_battery_aborted", False)
                or (battery_pct <= 15.0)
            )

            # Contorno total e preciso da pluma no regresso:
            if not is_low_battery:
                det_x = det_pos[0]
                side_ida = getattr(self, "skirting_side_ida", 1.0)
                if side_ida is None:
                    side_ida = 1.0
                local_center_y = get_plume_center_y(self.pos[0], self.time, self.plume, self.config)
                is_sinusoid = self.config.get("meander_model", "ou").lower() in [
                    "sum_of_sinusoids",
                    "sinusoid",
                    "sinusoidal",
                ]

                if is_sinusoid:
                    # Na pluma sinusoidal: o agente contorna a margem oposta até alcançar o mesmo X da deteção.
                    # Ao alcançar o mesmo x (self.pos[0] >= det_x - 0.5), conclui o contorno da pluma e desloca-se diretamente para o ponto de deteção!
                    opposite_side = -side_ida
                    is_on_opposite_side = (
                        (self.pos[1] >= local_center_y)
                        if opposite_side == 1.0
                        else (self.pos[1] <= local_center_y)
                    )
                    if not getattr(self, "return_tip_reached", False):
                        if self.pos[0] >= det_x - 0.5 and (
                            is_on_opposite_side or self.current_conc >= 0.15 * conc_thresh
                        ):
                            self.return_tip_reached = True
                            self.return_contour_completed = True
                    else:
                        if dist_to_detection < success_distance:
                            self.return_contour_completed = True
                else:
                    # Na pluma linear: contorno completo de toda a pluma até à ponta distal (x_turn ~80m),
                    # virando depois para a margem de ida a montante (-x) até alcançar det_pos:
                    x_turn = min(80.0, max(det_x + 16.0, 78.0))
                    if not getattr(self, "return_tip_reached", False):
                        is_on_ida_side = (
                            (self.pos[1] >= local_center_y)
                            if side_ida == 1.0
                            else (self.pos[1] <= local_center_y)
                        )
                        if self.pos[0] >= x_turn:
                            self.distal_tip_entered = True
                        if getattr(self, "distal_tip_entered", False) and is_on_ida_side:
                            self.return_tip_reached = True
                            self.min_x_returning_back = self.pos[0]
                            self.min_dist_to_det = dist_to_detection

                    if getattr(self, "return_tip_reached", False):
                        if dist_to_detection < success_distance:
                            self.return_contour_completed = True

            contour_done = getattr(self, "return_contour_completed", False)

            if is_low_battery:
                reached_detection = dist_to_detection < success_distance
                at_goal = False
                terminated = reached_detection
            else:
                reached_detection = dist_to_detection < success_distance
                at_goal = (
                    reached_detection
                    and contour_done
                    and getattr(self, "circling_completed", False)
                    and (battery_pct > 15.0)
                )
                terminated = at_goal

            truncated = self.time >= self.config["max_time"]
            reward = self._calculate_reward(
                self.current_conc,
                self.prev_conc,
                visitas_nesta_celula,
                at_goal,
                hit_boundary,
                action,
            )
            if truncated and not at_goal:
                reward -= 100.0
        else:
            at_goal = False
            terminated = (
                False  # A simulação não termina ao alcançar a fonte, segue para contorno e regresso
            )
            truncated = self.time >= self.config["max_time"]
            reward = self._calculate_reward(
                self.current_conc, self.prev_conc, visitas_nesta_celula, False, hit_boundary, action
            )
            if truncated:
                reward -= 100.0

        self.prev_action = action
        return self._get_obs(), reward, terminated, truncated, {"at_goal": at_goal}

    def _calculate_reward(
        self, conc, prev_conc, visitas_nesta_celula, at_goal, hit_boundary, action=None
    ):
        threshold = self.config["agent_config"]["agent_concentration_threshold"]
        source_pos = np.array(self.config["source_position"])
        step_size = self.speed * self.config["dt"]
        movement = self.pos - self.prev_pos

        # Verificação de inversão imediata de direção
        is_reversal = False
        if action is not None and getattr(self, "prev_action", None) is not None:
            is_reversal = (
                (action == 0 and self.prev_action == 1)
                or (action == 1 and self.prev_action == 0)
                or (action == 2 and self.prev_action == 3)
                or (action == 3 and self.prev_action == 2)
            )

        # Penalização estrita de anti-estagnação / anti-bloqueio (janela móvel de 16 passos)
        anti_stall_penalty = 0.0
        if hasattr(self, "recent_positions") and len(self.recent_positions) >= 16:
            disp_16 = float(np.linalg.norm(self.pos - self.recent_positions[-16]))
            if disp_16 < 0.40:
                anti_stall_penalty = 8.0 * (1.0 - disp_16 / 0.40)

        # =========================================================================
        # ESTADO 3: RETURNING (Contorno do resto da pluma até à primeira deteção)
        # =========================================================================
        if self.state == "returning":
            if at_goal:
                return 500.0  # Recompensa terminal ao regressar à posição de deteção após contorno completo

            battery_pct = (self.battery / self.battery_capacity) * 100.0
            is_low_battery = (
                getattr(self, "low_battery_return", False)
                or getattr(self, "low_battery_aborted", False)
                or (battery_pct <= 15.0)
            )

            if getattr(self, "return_to_start", False):
                det_pos = getattr(self, "start_pos", np.array([90.0, 25.0]))
            else:
                det_pos = (
                    self.first_detection_pos
                    if self.first_detection_pos is not None
                    else getattr(self, "start_pos", self.pos)
                )

            dist_to_detection = np.linalg.norm(self.pos - det_pos)

            if is_low_battery:
                # Retorno por bateria baixa: NUNCA pontua como sucesso!
                if dist_to_detection < self.config["agent_config"]["success_distance"]:
                    return (
                        -30.0
                    )  # Penalização terminal por missão abortada devido a esgotamento da bateria
                reward = -0.5
                curr_dist = np.linalg.norm(self.pos - det_pos)
                prev_dist = np.linalg.norm(self.prev_pos - det_pos)
                if curr_dist < prev_dist:
                    reward += 1.0
                return reward

            reward = self.reward_config["step_penalty"]
            dy = movement[1]
            dx = movement[0]
            # No regresso a montante (-x), delta_x_upstream > 0 significa avanço para Oeste em direção à deteção
            delta_x_upstream = self.prev_pos[0] - self.pos[0]
            # No regresso a jusante (+x), delta_x_downstream > 0 significa avanço para Este em direção à ponta
            delta_x_downstream = self.pos[0] - self.prev_pos[0]

            side_ida = getattr(self, "skirting_side_ida", 1.0)
            if side_ida is None:
                side_ida = 1.0
            opposite_side = -side_ida
            local_center_y = get_plume_center_y(self.pos[0], self.time, self.plume, self.config)

            is_sinusoid = self.config.get("meander_model", "ou").lower() in [
                "sum_of_sinusoids",
                "sinusoid",
                "sinusoidal",
            ]
            tip_reached = getattr(self, "return_tip_reached", False)
            tip_entered = getattr(self, "distal_tip_entered", False)

            curr_d = np.linalg.norm(self.pos - det_pos)
            prev_d = np.linalg.norm(self.prev_pos - det_pos)

            if is_sinusoid:
                # =====================================================================
                # PLUMA SINUSOIDAL:
                # Contorno ondulante da margem oposta a jusante (+x) até ao mesmo X da deteção.
                # Ao alcançar o mesmo X (self.pos[0] >= det_pos[0] - 0.5), desloca-se diretamente para det_pos!
                # =====================================================================
                is_at_detection_x = tip_reached or (self.pos[0] >= det_pos[0] - 0.5)

                if is_at_detection_x:
                    # Deslocamento direto para o ponto de deteção
                    prog_d = (prev_d - curr_d) / step_size
                    if prog_d > 0.01:
                        reward += 8.0 * prog_d
                    elif prog_d < -0.01:
                        reward -= 6.0 * abs(prog_d)
                    else:
                        reward -= 1.0
                else:
                    # Até alcançar det_pos[0], contorna a margem oposta da pluma sinusoidal a jusante (+x)
                    target_side = opposite_side
                    prog_x = delta_x_downstream / step_size
                    prog_goal = prog_x

                    is_on_correct_side = (
                        (self.pos[1] >= local_center_y)
                        if target_side == 1.0
                        else (self.pos[1] <= local_center_y)
                    )
                    if not is_on_correct_side:
                        reward -= 10.0
                        if dy * target_side > 0:
                            reward += 3.0
                    else:
                        dist_prev = abs(self.prev_pos[1] - local_center_y)
                        dist_curr = abs(self.pos[1] - local_center_y)
                        dy_outward = (dist_curr - dist_prev) / step_size
                        dy_inward = (dist_prev - dist_curr) / step_size

                        if conc > 1.8 * threshold:
                            if dy_outward > 0.05:
                                reward += 6.0 * dy_outward
                            else:
                                reward -= 3.0
                            if dy_inward > 0.05:
                                reward -= 4.0 * dy_inward
                            if prog_goal > 0.01:
                                reward += 1.0 * prog_goal
                            elif prog_goal < -0.01:
                                reward -= 3.0 * abs(prog_goal)
                        elif conc < 0.20 * threshold:
                            # Em água limpa fora da pluma: aproximação OBRIGATÓRIA à margem ondulante
                            if dy_inward > 0.05:
                                reward += 8.0 * dy_inward
                            else:
                                reward -= 4.0
                            if dy_outward > 0.05:
                                reward -= 6.0 * dy_outward
                            # Proibir estritamente cruzeiro downstream em água limpa (impedir atalhos!)
                            if prog_goal > 0.01:
                                if self.pos[0] > source_pos[0] + 3.0:
                                    reward -= 10.0 * prog_goal
                                else:
                                    reward += 2.0 * prog_goal
                        else:
                            # Medição matemática da convergência ao contorno do puff (erro de concentração):
                            prev_err = abs(prev_conc - threshold)
                            curr_err = abs(conc - threshold)
                            delta_contour = (prev_err - curr_err) / max(threshold, 1e-4)

                            log_ratio = abs(np.log10(max(conc, 1e-4) / threshold))
                            adh = float(np.exp(-2.5 * log_ratio))

                            if prog_goal > 0.01:
                                reward += 4.0 * prog_goal + 6.0 * adh
                                if delta_contour > 0.01:
                                    reward += 2.0 * min(delta_contour, 2.0)
                                elif conc > threshold and dy_outward > 0.05:
                                    reward += 2.0 * dy_outward
                                elif conc < threshold and dy_inward > 0.05:
                                    reward += 2.0 * dy_inward
                            elif prog_goal < -0.01:
                                reward -= 5.0 * abs(prog_goal)
                            else:
                                if (
                                    delta_contour > 0.01
                                    or (conc > threshold and dy_outward > 0.05)
                                    or (conc < threshold and dy_inward > 0.05)
                                    or adh > 0.3
                                ):
                                    reward += 6.0 * adh + 3.0 * max(0.0, delta_contour)
                                else:
                                    reward -= 2.0
            else:
                # =====================================================================
                # PLUMA LINEAR:
                # Contorno completo da pluma inteira até à ponta distal (x_turn ~80m),
                # virando depois para a margem de ida até regressar ao ponto de deteção!
                # =====================================================================
                x_turn = min(80.0, max(det_pos[0] + 16.0, 78.0))
                is_at_detection_x = tip_reached and (self.pos[0] <= det_pos[0] + 0.5)
                is_docking = tip_reached and (curr_d <= 5.0 or is_at_detection_x)

                if is_docking:
                    prog_d = (prev_d - curr_d) / step_size
                    if is_at_detection_x:
                        if prog_d > 0.01:
                            reward += 8.0 * prog_d
                        elif prog_d < -0.01:
                            reward -= 6.0 * abs(prog_d)
                        else:
                            reward -= 1.5
                        if self.pos[0] < det_pos[0] - 0.5 and delta_x_upstream > 0.01:
                            reward -= 4.0 * (delta_x_upstream / step_size)
                    else:
                        if prog_d > 0.01:
                            reward += 6.0 * prog_d
                        elif prog_d < -0.01:
                            reward -= 5.0 * abs(prog_d)
                        else:
                            reward -= 1.0
                else:
                    is_rounding_tip = (not tip_reached) and (tip_entered or self.pos[0] >= x_turn)

                    if is_rounding_tip:
                        if dy * side_ida > 0:
                            if conc > 1.5 * threshold:
                                reward -= 4.0
                            else:
                                prev_err = abs(prev_conc - threshold)
                                curr_err = abs(conc - threshold)
                                delta_c = (prev_err - curr_err) / max(threshold, 1e-4)
                                reward += 5.5 + 2.0 * max(0.0, delta_c)
                        elif dy * side_ida < 0:
                            reward -= 4.0
                        else:
                            reward -= 1.0

                        if self.pos[0] > x_turn + 2.0 and movement[0] > 0.01:
                            reward -= 3.5
                    else:
                        if not tip_reached:
                            target_side = opposite_side
                            prog_x = delta_x_downstream / step_size
                            prog_goal = prog_x
                        else:
                            target_side = side_ida
                            prog_x = delta_x_upstream / step_size
                            prog_goal = prog_x

                        is_on_correct_side = (
                            (self.pos[1] >= local_center_y)
                            if target_side == 1.0
                            else (self.pos[1] <= local_center_y)
                        )
                        if not is_on_correct_side:
                            reward -= 6.0
                            if dy * target_side > 0:
                                reward += 2.0
                        else:
                            dist_prev = abs(self.prev_pos[1] - local_center_y)
                            dist_curr = abs(self.pos[1] - local_center_y)
                            dy_outward = (dist_curr - dist_prev) / step_size
                            dy_inward = (dist_prev - dist_curr) / step_size

                            if conc > 1.8 * threshold:
                                if dy_outward > 0.05:
                                    reward += 5.0 * dy_outward
                                else:
                                    reward -= 2.0
                                if dy_inward > 0.05:
                                    reward -= 4.0 * dy_inward
                                if prog_goal > 0.01:
                                    reward += 1.0 * prog_goal
                                elif prog_goal < -0.01:
                                    reward -= 3.0 * abs(prog_goal)
                            elif conc < 0.20 * threshold:
                                if dy_inward > 0.05:
                                    reward += 6.0 * dy_inward
                                else:
                                    reward -= 3.0
                                if dy_outward > 0.05:
                                    reward -= 6.0 * dy_outward
                                if abs(prog_goal) > 0.01:
                                    reward -= 2.5
                            else:
                                prev_err = abs(prev_conc - threshold)
                                curr_err = abs(conc - threshold)
                                delta_contour = (prev_err - curr_err) / max(threshold, 1e-4)

                                log_ratio = abs(np.log10(max(conc, 1e-4) / threshold))
                                adh = float(np.exp(-2.5 * log_ratio))

                                if prog_goal > 0.01:
                                    reward += 4.0 * prog_goal + 6.0 * adh
                                    if delta_contour > 0.01:
                                        reward += 2.0 * min(delta_contour, 2.0)
                                    elif conc > threshold and dy_outward > 0.05:
                                        reward += 2.0 * dy_outward
                                    elif conc < threshold and dy_inward > 0.05:
                                        reward += 2.0 * dy_inward
                                elif prog_goal < -0.01:
                                    reward -= 5.0 * abs(prog_goal)
                                else:
                                    if (
                                        delta_contour > 0.01
                                        or (conc > threshold and dy_outward > 0.05)
                                        or (conc < threshold and dy_inward > 0.05)
                                        or adh > 0.3
                                    ):
                                        reward += 6.0 * adh + 3.0 * max(0.0, delta_contour)
                                    else:
                                        reward -= 2.0

            if hit_boundary or np.linalg.norm(movement) < 1e-6:
                reward -= 4.0

            if is_reversal:
                reward -= 8.0

            if anti_stall_penalty > 0.0:
                reward -= anti_stall_penalty

            if visitas_nesta_celula > 8:
                reward -= 0.6 * min(visitas_nesta_celula - 8, 10)

            if self.steps_in_current_cell > 15:
                reward -= 0.5 * min(self.steps_in_current_cell - 15, 4)

            reward -= self.reward_config.get("battery_drain_penalty_factor", 0.5) * self.last_drain
            return reward

        # =========================================================================
        # ESTADO 2: FOLLOWING_BOUNDARY (Contorno ondulante da margem até à fonte)
        # =========================================================================
        elif self.state == "following_boundary":
            source_pos = np.array(self.config["source_position"])
            dist_to_source = np.linalg.norm(self.pos - source_pos)
            if dist_to_source < self.config["agent_config"]["success_distance"]:
                return 500.0  # Chegou à fonte com sucesso e inicia circling

            reward = self.reward_config["step_penalty"]
            step_size = self.speed * self.config["dt"]
            movement = self.pos - self.prev_pos
            dy = movement[1]
            delta_x = (
                self.prev_pos[0] - self.pos[0]
            )  # delta_x > 0 é Oeste / Esquerda (-x, em direção à fonte)

            side = getattr(self, "skirting_side_ida", 1.0)
            if side is None:
                side = 1.0

            local_center_y = get_plume_center_y(self.pos[0], self.time, self.plume, self.config)
            is_on_correct_side = (
                (self.pos[1] >= local_center_y) if side == 1.0 else (self.pos[1] <= local_center_y)
            )

            # Guia próximo à fonte: incentiva aproximação direta quando a distância é curta
            near_source_radius = self.config["agent_config"]["success_distance"] * 4.0
            if dist_to_source < near_source_radius:
                prev_dist_src = np.linalg.norm(self.prev_pos - source_pos)
                approach_delta = (prev_dist_src - dist_to_source) / step_size
                # Bónus crescente proporcional à proximidade da fonte e à taxa de aproximação
                proximity_factor = 1.0 - (dist_to_source / near_source_radius)
                if approach_delta > 0.01:
                    reward += 6.0 * proximity_factor * approach_delta
                elif approach_delta < -0.01:
                    reward -= 4.0 * proximity_factor * abs(approach_delta)

            if not is_on_correct_side:
                reward -= 10.0
                if dy * side > 0:
                    reward += 3.0
            else:
                prog_x = delta_x / step_size
                dist_prev = abs(self.prev_pos[1] - local_center_y)
                dist_curr = abs(self.pos[1] - local_center_y)
                dy_outward = (dist_curr - dist_prev) / step_size
                dy_inward = (dist_prev - dist_curr) / step_size

                if conc > 1.8 * threshold:
                    # Demasiado no interior/núcleo: incentivar saída para a margem
                    if dy_outward > 0.05:
                        reward += 6.0 * dy_outward
                    else:
                        reward -= 3.0
                    if dy_inward > 0.05:
                        reward -= 4.0 * dy_inward
                    if prog_x > 0.01:
                        reward += 1.0 * prog_x
                    elif prog_x < -0.01:
                        reward -= 3.0 * abs(prog_x)
                elif conc < 0.20 * threshold:
                    # Em água limpa fora da pluma: aproximação OBRIGATÓRIA à margem ondulante
                    if dy_inward > 0.05:
                        reward += 8.0 * dy_inward
                    else:
                        reward -= 4.0
                    if dy_outward > 0.05:
                        reward -= 6.0 * dy_outward
                    # Proibir estritamente avanço em água limpa sem contacto com o contorno da pluma
                    if prog_x > 0.01:
                        reward -= 10.0 * prog_x
                else:
                    # Medição matemática da convergência ao contorno do puff (erro de concentração):
                    prev_err = abs(prev_conc - threshold)
                    curr_err = abs(conc - threshold)
                    delta_contour = (prev_err - curr_err) / max(threshold, 1e-4)

                    log_ratio = abs(np.log10(max(conc, 1e-4) / threshold))
                    adh = float(np.exp(-2.5 * log_ratio))

                    if prog_x > 0.01:
                        reward += 4.0 * prog_x + 6.0 * adh
                        if delta_contour > 0.01:
                            reward += 2.0 * min(delta_contour, 2.0)
                        elif conc > threshold and dy_outward > 0.05:
                            reward += 2.0 * dy_outward
                        elif conc < threshold and dy_inward > 0.05:
                            reward += 2.0 * dy_inward
                    elif prog_x < -0.01:
                        reward -= 5.0 * abs(prog_x)
                    else:
                        # Passo lateral (ação 0 ou 1) para acompanhamento da ondulação sinusoidal
                        if (
                            delta_contour > 0.01
                            or (conc > threshold and dy_outward > 0.05)
                            or (conc < threshold and dy_inward > 0.05)
                            or adh > 0.3
                        ):
                            reward += 6.0 * adh + 3.0 * max(0.0, delta_contour)
                        else:
                            reward -= 2.0

            if hit_boundary or np.linalg.norm(movement) < 1e-6:
                reward -= 4.0

            if is_reversal:
                reward -= 8.0

            if anti_stall_penalty > 0.0:
                reward -= anti_stall_penalty

            if visitas_nesta_celula > 8:
                reward -= 0.6 * min(visitas_nesta_celula - 8, 10)

            if self.steps_in_current_cell > 15:
                reward -= 0.5 * min(self.steps_in_current_cell - 15, 4)

            reward -= self.reward_config.get("battery_drain_penalty_factor", 0.5) * self.last_drain
            return reward

        # =========================================================================
        # ESTADO 1: SEARCHING (Movimentação pura em traça / Cast-and-Surge)
        # =========================================================================
        else:
            reward = self.reward_config["step_penalty"]
            step_size = self.speed * self.config["dt"]
            movement = self.pos - self.prev_pos
            movement_norm = np.linalg.norm(movement)

            # 1. Bónus de Primeira Deteção da Pluma (Interceção direta da pluma)
            if conc > threshold:
                return (
                    50.0  # Forte recompensa imediata por intercetar a pluma através do varrimento
                )

            # 2. Recompensa estrita para Movimento em Traça (Cast):
            y_min_sweep = 0.30 * self.config["world_height"]
            y_max_sweep = 0.70 * self.config["world_height"]
            target_y = y_max_sweep if getattr(self, "cast_dir", 0) == 0 else y_min_sweep
            dist_before = abs(self.prev_pos[1] - target_y)
            dist_after = abs(self.pos[1] - target_y)
            delta_sweep = dist_before - dist_after
            dx = self.pos[0] - self.prev_pos[0]

            # A. Varrimento transversal (Cast): recompensa contínua ao aproximar-se da margem oposta
            if delta_sweep > 0.01:
                reward += 6.0 * (delta_sweep / step_size)
                # Bónus adicional de avanço contra o vento APENAS quando em varrimento transversal efetivo
                if dx < -0.01:
                    reward += 3.0 * (-dx / step_size)
            elif delta_sweep < -0.01:
                # Penalização por afastar-se da margem-alvo do varrimento (inversão vertical errada)
                reward -= 10.0 * (abs(delta_sweep) / step_size)
            else:
                reward -= 4.0

            # B. Penalização de avanço a favor do vento (recuo longitudinal no searching)
            if dx > 0.01:
                reward -= 8.0 * (dx / step_size)

            # C. Penalização estrita por movimentos puramente lineares sem varrimento (Ação 2 e 3)
            # No estado searching, o agente deve estritamente varrer transversalmente (traça)
            if action == 2:
                reward -= 15.0
            elif action == 3:
                reward -= 18.0

            if hit_boundary or movement_norm < 1e-6:
                reward -= 4.0

            if is_reversal and (
                0.32 * self.config["world_height"]
                < self.pos[1]
                < 0.68 * self.config["world_height"]
            ):
                reward -= 10.0

            if anti_stall_penalty > 0.0:
                reward -= anti_stall_penalty

            if visitas_nesta_celula > 8:
                reward -= 0.6 * min(visitas_nesta_celula - 8, 10)

            if self.steps_in_current_cell > 15:
                reward -= 0.5 * min(self.steps_in_current_cell - 15, 4)

            reward -= self.reward_config.get("battery_drain_penalty_factor", 0.5) * self.last_drain
            return reward

    def get_current_plume_center_y(self, x=None):
        if x is None:
            x = self.pos[0]
        return get_plume_center_y(x, self.time, self.plume, self.config)

    def _get_obs(self):
        conc_normalized = np.clip(self.current_conc / self.conc_max, 0, 1)

        vy = self.plume.meander_generator.meandering_vy
        flow = np.array(self.config["mean_wind_velocity"]) + np.array([0.0, vy])

        flow_x_normalized = np.clip(flow[0] / self.flow_max, -1, 1)
        flow_y_normalized = np.clip(flow[1] / self.flow_max, -1, 1)

        obs_x = self.pos[0] / self.config["world_width"]
        obs_y = self.pos[1] / self.config["world_height"]
        conc_val = conc_normalized

        if self.state == "searching":
            conc_val = 0.0
            state_val = 0.0
            border_side_or_cast = float(getattr(self, "cast_dir", 0))
        elif self.state == "following_boundary":
            state_val = 1.0
            border_side_or_cast = 1.0 if getattr(self, "skirting_side_ida", 1.0) == 1.0 else 0.0
        else:  # returning
            is_sinusoid = self.config.get("meander_model", "ou").lower() in [
                "sum_of_sinusoids",
                "sinusoid",
                "sinusoidal",
            ]
            det_pos = (
                self.first_detection_pos
                if self.first_detection_pos is not None
                else getattr(self, "start_pos", np.array([90.0, 25.0]))
            )

            if is_sinusoid:
                # Pluma Sinusoidal: ao alcançar o mesmo X da deteção, transita diretamente para docking
                if getattr(self, "return_tip_reached", False) or self.pos[0] >= det_pos[0] - 0.5:
                    state_val = 3.0
                    border_side_or_cast = 1.0 if det_pos[1] > self.pos[1] else 0.0
                else:
                    state_val = 2.0
                    ret_side = (
                        -getattr(self, "skirting_side_ida", 1.0)
                        if getattr(self, "skirting_side_ida", 1.0) is not None
                        else -1.0
                    )
                    border_side_or_cast = 1.0 if ret_side == 1.0 else 0.0
            else:
                # Pluma Linear: contorno completo até à ponta distal e regresso pela margem de ida
                if getattr(self, "return_tip_reached", False):
                    state_val = 3.0
                    if self.pos[0] <= det_pos[0] + 0.5:
                        border_side_or_cast = 1.0 if det_pos[1] > self.pos[1] else 0.0
                    else:
                        border_side_or_cast = (
                            1.0 if getattr(self, "skirting_side_ida", 1.0) == 1.0 else 0.0
                        )
                else:
                    state_val = 2.0
                    ret_side = (
                        -getattr(self, "skirting_side_ida", 1.0)
                        if getattr(self, "skirting_side_ida", 1.0) is not None
                        else -1.0
                    )
                    border_side_or_cast = 1.0 if ret_side == 1.0 else 0.0

        battery_level = np.clip(self.battery / self.battery_capacity, 0.0, 1.0)

        return np.array(
            [
                obs_x,
                obs_y,
                conc_val,
                flow_x_normalized,
                flow_y_normalized,
                state_val,
                battery_level,
                border_side_or_cast,
            ],
            dtype=np.float32,
        )

    def print_maze_info(self):
        """Print environment information"""
        print(
            f"Continuous Plume Environment: {self.config['world_width']}x{self.config['world_height']}"
        )
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")
