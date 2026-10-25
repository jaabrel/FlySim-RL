from __future__ import annotations

import os
import time

import cv2
import numpy as np

from flysim.agents.rl.qlearning import AGENT_REGISTRY, QLearningSimAgent
from flysim.plume.filament import FilamentPlume
from flysim.plume.kernels import (
    get_plume_center_y,
    plume_exists_at_x,
)
from flysim.simulation.logger import DataLogger


class Simulator:
    """
    Orquestrador principal da simulação física e visualização gráfica em tempo real.
    Gere o ciclo de vida dos ensaios, a integração temporal do fluido, os sensores
    do veículo autónomo e a renderização do painel HUD marinho em Pygame / OpenCV.
    """

    def __init__(self, config):
        """Inicializa o simulador, instancia o agente configurado e prepara a janela visual."""
        self.config = config
        agent_class_name = config["agent_class"]
        if agent_class_name not in AGENT_REGISTRY:
            raise ValueError(f"Classe de agente '{agent_class_name}' não registada.")
        agent_full_config = {**config, **config.get("agent_config", {})}
        self.agent = AGENT_REGISTRY[agent_class_name](agent_full_config)
        if hasattr(self.agent, "config") and "meander_model" in self.agent.config:
            self.config["meander_model"] = self.agent.config["meander_model"]
        self.plume, self.logger = None, None
        self.simulation_time, self.current_experiment = 0.0, 0
        self.loop_time_ema, self.ema_alpha = 0.0, 0.1
        self.is_paused = False
        self.user_quit = False
        self.skip_episode = False

        # Configurar episódios a apresentar (últimos 5 episódios gravados do modelo selecionado)
        q_agent_ref = getattr(self.agent, "q_agent", None)
        self.saved_paths = getattr(q_agent_ref, "saved_paths", {})
        self.saved_detection_points = getattr(q_agent_ref, "saved_detection_points", {})

        all_eps = sorted(self.saved_paths.keys()) if self.saved_paths else []
        if len(all_eps) >= 5:
            self.episodes_to_show = all_eps[-5:]
        else:
            self.episodes_to_show = all_eps

        self.current_ep_num = None
        self.current_ep_path = None
        self.current_path_step = 0

        if config["visualize"]:
            self._init_visualization()

    def reset_experiment(self):
        """Reinicializa as variáveis de ambiente, posicionamento e registo para um novo ensaio."""
        if hasattr(self.agent, "config") and "meander_model" in self.agent.config:
            self.config["meander_model"] = self.agent.config["meander_model"]
        self.logger = DataLogger(self.config, self.current_experiment)
        self.simulation_time = 0.0
        self.is_paused = False
        self.skip_episode = False

        if (
            hasattr(self, "episodes_to_show")
            and self.episodes_to_show
            and self.current_experiment < len(self.episodes_to_show)
        ):
            self.current_ep_num = self.episodes_to_show[self.current_experiment]
            self.current_ep_path = self.saved_paths.get(self.current_ep_num, None)
            self.current_path_step = 0

            det_pt = self.saved_detection_points.get(self.current_ep_num, None)
            self.saved_ep_detection_pt = (
                np.array(det_pt, dtype=np.float64) if det_pt is not None else None
            )
            self.agent.first_detection_pos = None

            total_show = len(self.episodes_to_show)
            ep_idx_str = f"{self.current_experiment + 1}/{total_show}"
            model_disp = getattr(self.agent, "model_name", "Modelo")
            print(
                f"\n[Simulator] A apresentar o Episódio #{self.current_ep_num} ({ep_idx_str}) de '{model_disp}' ({len(self.current_ep_path)} passos)..."
            )
        else:
            self.current_ep_num = self.current_experiment + 1
            self.current_ep_path = None
            self.current_path_step = 0
            self.saved_ep_detection_pt = None
            self.agent.first_detection_pos = None

        # Sincronização e reprodutibilidade estocástica: seed determinístico por episódio
        seed_val = int(self.current_ep_num) * 1009 + 42
        np.random.seed(seed_val)

        self.plume = FilamentPlume(self.config)
        self.agent.plume = self.plume
        self.agent.reset()
        self.config["agent_start_region_center"] = self.agent.config["agent_start_region_center"]

        # Aquecimento inicial idêntico ao ambiente de treino (8.0 segundos)
        # para sincronização temporal e de fase absoluta da pluma com a trajetória gravada
        initial_wait_time = self.config.get("agent_initial_wait_time", 8.0)
        self.plume_time = 0.0
        if getattr(self, "current_ep_path", None) is not None:
            for _ in range(int(initial_wait_time / self.config["dt"])):
                self.plume.update(self.config["dt"], self.plume_time)
                self.plume_time += self.config["dt"]

        if self.current_ep_path and len(self.current_ep_path) > 0:
            p0 = self.current_ep_path[0]
            self.agent.pos = np.array(
                [p0[0] * self.config["world_width"], p0[1] * self.config["world_height"]],
                dtype=np.float64,
            )
            self.agent.start_pos = self.agent.pos.copy()

    def step(self):
        dt = self.config["dt"]

        if getattr(self, "skip_episode", False):
            print(f"\n[Simulator] Episódio #{self.current_ep_num} avançado para o próximo.")
            return True

        # Caso 1: Apresentação dos episódios gravados do modelo selecionado
        if getattr(self, "current_ep_path", None) is not None:
            step_idx = self.current_path_step
            if step_idx >= len(self.current_ep_path):
                print(f"\n[Simulator] Episódio #{self.current_ep_num} concluído com sucesso!")
                return True

            norm_pos = self.current_ep_path[step_idx]
            world_w = self.config["world_width"]
            world_h = self.config["world_height"]
            new_pos = np.array([norm_pos[0] * world_w, norm_pos[1] * world_h], dtype=np.float64)
            old_pos = self.agent.pos.copy()

            if step_idx > 0:
                prev_norm = self.current_ep_path[step_idx - 1]
                delta_p = np.array([norm_pos[0] - prev_norm[0], norm_pos[1] - prev_norm[1]])
                norm_d = np.linalg.norm(delta_p)
                if norm_d > 1e-5:
                    self.agent.direction = delta_p / norm_d
                    desired_theta = np.arctan2(delta_p[1], delta_p[0])
                    if self.agent.theta is None:
                        self.agent.theta = desired_theta
                    else:
                        diff = (desired_theta - self.agent.theta + np.pi) % (2 * np.pi) - np.pi
                        self.agent.theta += np.clip(diff, -6.0 * dt, 6.0 * dt)

            self.agent.pos = new_pos
            self.current_path_step += 1

            self.plume.update(dt, self.plume_time)
            self.plume_time += dt
            self.agent.simulation_time = self.plume_time
            self.agent.plume = self.plume
            self.agent.measure_concentration(self.plume)
            self.agent.measure_flow(self.plume)

            # Consumo real de bateria baseado no deslocamento e vento
            displacement = new_pos - old_pos
            dist_moved = np.linalg.norm(displacement)
            direction = displacement / dist_moved if dist_moved > 1e-6 else np.array([0.0, 0.0])
            flow_vec = self.agent.measured_flow_direction
            opposing_wind = -flow_vec
            wind_resistance = np.dot(direction, opposing_wind)
            base_decay = self.agent.battery_base_decay
            wind_factor = self.agent.battery_wind_factor
            min_decay = self.agent.battery_min_decay
            drain_rate = max(min_decay, base_decay + wind_factor * wind_resistance)
            self.agent.battery = max(0.0, self.agent.battery - drain_rate * dt)

            # Máquina de estados visual do ensaio
            source_pos = np.array(self.config["source_position"])
            dist_to_source = np.linalg.norm(self.agent.pos - source_pos)
            det_pos = getattr(self.agent, "first_detection_pos", None)
            total_steps = max(1, len(self.current_ep_path))

            if det_pos is None:
                conc_thresh = self.config.get("agent_config", {}).get(
                    "agent_concentration_threshold", 1.0
                )
                det_reached = False
                if getattr(self, "saved_ep_detection_pt", None) is not None:
                    if np.linalg.norm(self.agent.pos - self.saved_ep_detection_pt) <= 1.8:
                        det_reached = True

                if self.agent.current_concentration > conc_thresh or det_reached:
                    self.agent.first_detection_pos = (
                        self.saved_ep_detection_pt.copy()
                        if getattr(self, "saved_ep_detection_pt", None) is not None
                        else self.agent.pos.copy()
                    )
                    self.agent.state = "following_boundary"
                    print(
                        f"\n[Simulator] Pluma DETETADA em ({self.agent.first_detection_pos[0]:.1f}m, {self.agent.first_detection_pos[1]:.1f}m)! A iniciar contorno da pluma até à fonte."
                    )
                else:
                    self.agent.state = "searching"
            else:
                dist_to_det = np.linalg.norm(self.agent.pos - det_pos)
                if not getattr(self.agent, "source_reached", False):
                    if dist_to_source <= 2.2:
                        self.agent.source_reached = True
                        self.agent.state = "following_boundary"
                    else:
                        self.agent.state = "following_boundary"
                else:
                    if dist_to_det <= 2.0 and step_idx > int(0.65 * total_steps):
                        self.agent.mission_complete = True
                        self.agent.state = "returning"
                    else:
                        self.agent.state = "returning"

            custom_data = {"state": self.agent.state, "ep_num": self.current_ep_num}
            self.logger.log_step(
                self.simulation_time,
                self.agent.pos.copy(),
                self.agent.current_concentration,
                self.agent.state,
                self.agent.battery,
                custom_data,
            )
            self.simulation_time += dt

            if self.agent.battery <= 0:
                print(f"\n[Simulator] Bateria ESGOTADA no Episódio #{self.current_ep_num}!")
                return True
            if self.agent.mission_complete and step_idx >= len(self.current_ep_path) - 3:
                print(
                    f"\n[Simulator] SUCESSO! Episódio #{self.current_ep_num} concluído com êxito!"
                )
                return True
            return False

        # Caso 2: Simulação dinâmica em tempo real (fallback se o modelo não tiver rotas gravadas)
        if self.agent.battery <= 0:
            print("\n[Simulator] Agent started step with depleted battery! Mission failed.")
            return True

        self.plume.update(dt, self.simulation_time)
        self.agent.simulation_time = self.simulation_time
        self.agent.plume = self.plume
        self.agent.measure_concentration(self.plume)
        self.agent.measure_flow(self.plume)

        # Esperar 8 segundos no início de cada episódio antes de iniciar o movimento
        initial_wait_time = self.config.get("agent_initial_wait_time", 8.0)
        if self.simulation_time < initial_wait_time:
            self.agent.state = "waiting"
            custom_data = {"state": "waiting"}
            self.logger.log_step(
                self.simulation_time,
                self.agent.pos.copy(),
                self.agent.current_concentration,
                self.agent.state,
                self.agent.battery,
                custom_data,
            )
            self.simulation_time += dt
            return False
        elif self.agent.state == "waiting":
            self.agent.state = "searching"

        conc_thresh = self.config["agent_config"].get("agent_concentration_threshold", 1.0)
        source_pos = np.array(self.config["source_position"])
        dist_to_source = np.linalg.norm(self.agent.pos - source_pos)
        success_distance = self.config["agent_config"].get("success_distance", 2.0)

        # --- LOW BATTERY SAFETY CHECK (<= 15%) ---
        battery_pct = (self.agent.battery / self.agent.battery_capacity) * 100.0
        is_low_battery = (
            (battery_pct <= 15.0)
            or getattr(self.agent, "low_battery_direct_return", False)
            or getattr(self.agent, "low_battery_start_return", False)
            or getattr(self.agent, "low_battery_aborted", False)
        )

        if is_low_battery:
            self.agent.mission_complete = False
            self.agent.low_battery_aborted = True

        # 1. State transition from searching -> following_boundary
        if self.agent.state == "searching":
            if self.agent.current_concentration > conc_thresh:
                self.agent.state = "following_boundary"
                self.agent.first_detection_pos = self.agent.pos.copy()
                self.agent.theta = None  # Reset heading for following_boundary
                local_center_y = get_plume_center_y(
                    self.agent.pos[0], self.simulation_time, self.plume, self.config
                )
                self.agent.skirting_side_ida = 1.0 if self.agent.pos[1] >= local_center_y else -1.0
                side_str = "TOP" if self.agent.skirting_side_ida == 1.0 else "BOTTOM"
                print(
                    f"\n[Simulator] Plume DETECTED at {self.agent.first_detection_pos}! Side: {side_str}. Following boundary to source."
                )

        # 2. Check source arrival when following boundary
        elif self.agent.state == "following_boundary":
            if (
                dist_to_source < success_distance
                or (
                    self.agent.pos[0] <= source_pos[0] + 1.8
                    and dist_to_source < success_distance * 1.8
                )
            ) and not self.agent.source_reached:
                self.agent.source_reached = True
                self.agent.circling_step = 0
                print("\n[Simulator] Source REACHED! Starting circling/contouring maneuver.")

        # 3. Check mission completion when returning normally
        elif self.agent.state == "returning":
            if self.agent.first_detection_pos is not None:
                dist_to_detection = np.linalg.norm(self.agent.pos - self.agent.first_detection_pos)
                det_x = self.agent.first_detection_pos[0]
                side_ida = getattr(self.agent, "skirting_side_ida", 1.0)
                if side_ida is None:
                    side_ida = 1.0
                local_center_y = get_plume_center_y(
                    self.agent.pos[0], self.simulation_time, self.plume, self.config
                )
                is_sinusoid = self.config.get("meander_model", "ou").lower() in [
                    "sum_of_sinusoids",
                    "sinusoid",
                    "sinusoidal",
                ]

                if not is_low_battery:
                    if is_sinusoid:
                        # Pluma Sinusoidal: ao alcançar o mesmo X do ponto de deteção, conclui o contorno da pluma:
                        opposite_side = -side_ida
                        is_on_opposite_side = (
                            (self.agent.pos[1] >= local_center_y)
                            if opposite_side == 1.0
                            else (self.agent.pos[1] <= local_center_y)
                        )
                        if self.agent.pos[0] >= det_x - 0.5 and (
                            is_on_opposite_side
                            or self.agent.current_concentration >= 0.15 * conc_thresh
                        ):
                            self.agent.return_tip_reached = True
                            self.agent.return_contour_completed = True
                        if (
                            getattr(self.agent, "return_tip_reached", False)
                            and dist_to_detection < success_distance
                        ):
                            self.agent.return_contour_completed = True
                    else:
                        # Pluma Linear: contorno completo até à ponta distal (x_turn ~80m) e regresso pela margem de ida:
                        x_turn = max(det_x + 8.0, 75.0)
                        x_turn = min(self.config["world_width"] - 6.0, x_turn)

                        # 1. Contorno a jusante da margem oposta (+x) até virar em redor da ponta para a margem de ida:
                        if not getattr(self.agent, "return_tip_reached", False):
                            is_on_ida_side = (
                                (self.agent.pos[1] >= local_center_y)
                                if side_ida == 1.0
                                else (self.agent.pos[1] <= local_center_y)
                            )
                            if self.agent.pos[0] >= x_turn and is_on_ida_side:
                                self.agent.return_tip_reached = True

                        # 2. Contorno a montante da margem de ida (-x) até alcançar det_pos:
                        if getattr(self.agent, "return_tip_reached", False):
                            if dist_to_detection < success_distance:
                                self.agent.return_contour_completed = True

                contour_done = getattr(self.agent, "return_contour_completed", False)

                if dist_to_detection < success_distance and contour_done:
                    # O episódio SÓ é um sucesso se realizou todos os estados da forma correta (contornando toda a pluma e sem aborto por bateria)
                    if (
                        getattr(self.agent, "circling_completed", False)
                        and not is_low_battery
                        and (battery_pct > 15.0)
                    ):
                        self.agent.mission_complete = True
                        print(
                            f"\n[Simulator] SUCESSO! Missão concluída com todos os estados realizados com sucesso em {self.simulation_time:.2f}s (Bateria: {battery_pct:.1f}%)!"
                        )
                    else:
                        self.agent.mission_complete = False
                        self.agent.incomplete_mission = True
                        print(
                            f"\n[Simulator] Chegou ao ponto de deteção, mas o episódio NÃO teve sucesso (missão incompleta ou retorno por bateria baixa: {battery_pct:.1f}%)."
                        )

        old_pos = self.agent.pos.copy()

        # CASO 1: No estado searching (ainda não detetou a pluma de todo), se bateria <= 15% -> Retornar ao ponto de início!
        if (
            (
                self.agent.state in ["searching", "returning_to_start"]
                or getattr(self.agent, "low_battery_start_return", False)
            )
            and self.agent.first_detection_pos is None
            and is_low_battery
        ):
            start_target = getattr(self.agent, "start_pos", np.array([90.0, 25.0]))
            if not getattr(self.agent, "low_battery_start_return", False):
                self.agent.low_battery_start_return = True
                self.agent.state = "returning_to_start"
                print(
                    f"\n[Simulator] Bateria baixa no estado searching ({battery_pct:.1f}% <= 15%) sem detetar pluma! Retornando ao ponto de início {start_target}!"
                )

            dist_to_start = np.linalg.norm(self.agent.pos - start_target)
            if dist_to_start < success_distance:
                # Regresso ao ponto de início por bateria baixa sem detetar a pluma: SEM SUCESSO!
                self.agent.mission_complete = False
                self.agent.low_battery_aborted = True
                print(
                    f"\n[Simulator] Agente regressou ao ponto de início {self.agent.pos} com bateria baixa ({battery_pct:.1f}%). Episódio SEM SUCESSO (pluma não detetada)."
                )
                custom_data = {
                    "state": "returning_start_low_battery_failed",
                    "battery_pct": battery_pct,
                }
            else:
                goal_vec = start_target - self.agent.pos
                goal_norm = np.linalg.norm(goal_vec)
                target_pos = (
                    self.agent.pos + (goal_vec / goal_norm) * self.agent.speed * dt
                    if goal_norm > 1e-6
                    else self.agent.pos
                )
                if goal_norm > 1e-6:
                    self.agent.direction = goal_vec / goal_norm
                custom_data = {"state": "returning_start_low_battery", "battery_pct": battery_pct}
                self.agent.move(target_pos, dt)

        # CASO 2: Se já tinha detetado a pluma (em contorno ou regresso) e bateria <= 15% -> Retornar ao ponto de deteção!
        elif (
            self.agent.state in ["following_boundary", "returning"]
            and self.agent.first_detection_pos is not None
            and is_low_battery
        ):
            if not getattr(self.agent, "low_battery_direct_return", False):
                self.agent.low_battery_direct_return = True
                self.agent.state = "returning"
                print(
                    f"\n[Simulator] Low battery safety trigger! Bateria <= 15% ({battery_pct:.1f}%). Retornando DIRETAMENTE ao ponto de deteção {self.agent.first_detection_pos}!"
                )

            dist_to_detection = np.linalg.norm(self.agent.pos - self.agent.first_detection_pos)
            if dist_to_detection < success_distance:
                # Regresso por bateria baixa: NUNCA tem sucesso!
                self.agent.mission_complete = False
                self.agent.low_battery_aborted = True
                print(
                    f"\n[Simulator] Agente regressou ao ponto de deteção {self.agent.pos} com bateria baixa ({battery_pct:.1f}%). Episódio SEM SUCESSO (não completou todos os estados)."
                )
                custom_data = {
                    "state": "returning_direct_low_battery_failed",
                    "battery_pct": battery_pct,
                }
            else:
                goal_vec = self.agent.first_detection_pos - self.agent.pos
                goal_norm = np.linalg.norm(goal_vec)
                target_pos = (
                    self.agent.pos + (goal_vec / goal_norm) * self.agent.speed * dt
                    if goal_norm > 1e-6
                    else self.agent.pos
                )
                if goal_norm > 1e-6:
                    self.agent.direction = goal_vec / goal_norm
                custom_data = {"state": "returning_direct_low_battery", "battery_pct": battery_pct}
                self.agent.move(target_pos, dt)

        elif self.agent.state == "following_boundary" and self.agent.source_reached:
            target_pos = self.agent.get_circling_target(dt)
            custom_data = {"state": "circling_source", "step": self.agent.circling_step}
            self.agent.move(target_pos, dt)

        elif isinstance(self.agent, QLearningSimAgent):
            # O agente Q-Learning navega ativamente em conformidade com as decisões aprendidas na sua Q-Table
            target_pos, custom_data = self.agent.run_search_algorithm(dt)
            self.agent.move(target_pos, dt)

        elif self.agent.state == "following_boundary":
            local_center_y = get_plume_center_y(
                self.agent.pos[0], self.simulation_time, self.plume, self.config
            )
            if self.agent.skirting_side_ida is None:
                self.agent.skirting_side_ida = 1.0 if self.agent.pos[1] >= local_center_y else -1.0
            side = self.agent.skirting_side_ida
            # Initialize theta if not present or None
            if (
                not hasattr(self.agent, "theta")
                or self.agent.theta is None
                or self.agent.theta == 0.0
            ):
                self.agent.theta = np.pi - (np.pi / 4.0 * side)

            # Dynamic steering logic based on concentration (mesma forma do estado returning)
            omega = 3.2  # turning rate in rad/s, responsive to sinusoidal boundary changes
            if self.agent.current_concentration > conc_thresh:
                # Inside plume: turn away from plume interior
                if side == 1.0:
                    self.agent.theta -= omega * dt
                else:
                    self.agent.theta += omega * dt
            else:
                # Outside plume: turn towards plume interior
                if side == 1.0:
                    self.agent.theta += omega * dt
                else:
                    self.agent.theta -= omega * dt

            # Symmetric clip around upwind direction (pi)
            # Allows steering up to 75 deg (5*pi/12) up or down, hugging sinusoidal wave peaks and troughs
            theta_max = 5.0 * np.pi / 12.0
            self.agent.theta = np.clip(self.agent.theta, np.pi - theta_max, np.pi + theta_max)

            target_pos = (
                self.agent.pos
                + np.array([np.cos(self.agent.theta), np.sin(self.agent.theta)])
                * self.agent.speed
                * dt
            )
            custom_data = {"state": "following_boundary_steering", "theta": self.agent.theta}
            self.agent.move(target_pos, dt)

        elif self.agent.state == "returning":
            opposite_side = (
                -self.agent.skirting_side_ida if self.agent.skirting_side_ida is not None else -1.0
            )
            local_center_y = get_plume_center_y(
                self.agent.pos[0], self.simulation_time, self.plume, self.config
            )
            on_returning_side = (
                (self.agent.pos[1] < local_center_y)
                if opposite_side == -1.0
                else (self.agent.pos[1] >= local_center_y)
            )

            # Check if we reached the X coordinate of detection point after the contour has been completed
            reached_detection_x = False
            if self.agent.return_waypoint_reached and not on_returning_side:
                if self.agent.pos[0] <= self.agent.first_detection_pos[0]:
                    reached_detection_x = True

            if reached_detection_x:
                goal_vec = np.array([0.0, self.agent.first_detection_pos[1] - self.agent.pos[1]])
                goal_norm = np.linalg.norm(goal_vec)
                target_pos = (
                    self.agent.pos + (goal_vec / goal_norm) * self.agent.speed * dt
                    if goal_norm > 1e-6
                    else self.agent.pos
                )
                custom_data = {"state": "returning_vertical_final"}
                self.agent.move(target_pos, dt)
            else:
                # Initialize agent.theta if not present or 0.0
                if (
                    not hasattr(self.agent, "theta")
                    or self.agent.theta is None
                    or self.agent.theta == 0.0
                ):
                    self.agent.theta = -np.pi / 4.0 if opposite_side == -1.0 else np.pi / 4.0

                # Dynamic steering logic based on concentration
                omega = 2.4  # turning rate in rad/s
                if self.agent.current_concentration > conc_thresh:
                    # Inside: turn away from center
                    if opposite_side == -1.0:
                        self.agent.theta -= omega * dt
                    else:
                        self.agent.theta += omega * dt
                else:
                    # Outside: turn towards center
                    if opposite_side == -1.0:
                        self.agent.theta += omega * dt
                    else:
                        self.agent.theta -= omega * dt

                # Update return state
                if not self.agent.return_waypoint_reached:
                    if self.agent.pos[0] >= self.agent.first_detection_pos[0]:
                        if not plume_exists_at_x(self.agent.pos[0], self.plume, self.config):
                            self.agent.return_waypoint_reached = True

                    if self.agent.pos[0] >= 98.0:
                        self.agent.return_waypoint_reached = True

                # Clip theta depending on returning phase to follow silhouette dynamically
                if not self.agent.return_waypoint_reached:
                    # Phase 1: downwind boundary following (limit heading to downwind directions)
                    theta_max_ret = 5.0 * np.pi / 12.0
                    self.agent.theta = np.clip(self.agent.theta, -theta_max_ret, theta_max_ret)
                else:
                    # Phase 2: crossing the end or returning upwind
                    if on_returning_side:
                        # Crossing the end of the plume: allow turning around the end (up to 180 degrees)
                        if opposite_side == -1.0:
                            self.agent.theta = np.clip(self.agent.theta, -np.pi / 3, np.pi)
                        else:
                            self.agent.theta = np.clip(self.agent.theta, -np.pi, np.pi / 3)
                    else:
                        # Returning on the original side (limit heading to upwind directions)
                        if opposite_side == -1.0:
                            self.agent.theta = np.clip(self.agent.theta, np.pi / 2, 1.5 * np.pi)
                        else:
                            self.agent.theta = np.clip(self.agent.theta, -1.5 * np.pi, -np.pi / 2)

                target_pos = (
                    self.agent.pos
                    + np.array([np.cos(self.agent.theta), np.sin(self.agent.theta)])
                    * self.agent.speed
                    * dt
                )
                custom_data = {"state": "returning_steering", "theta": self.agent.theta}
                self.agent.move(target_pos, dt)
        else:
            target_pos, custom_data = self.agent.run_search_algorithm(dt)
            self.agent.move(target_pos, dt)

        # Calculate battery drain based on actual displacement
        displacement = self.agent.pos - old_pos
        dist_moved = np.linalg.norm(displacement)
        direction = displacement / dist_moved if dist_moved > 1e-6 else np.array([0.0, 0.0])

        flow_vec = self.agent.measured_flow_direction
        opposing_wind = -flow_vec
        wind_resistance = np.dot(direction, opposing_wind)

        base_decay = self.agent.battery_base_decay
        wind_factor = self.agent.battery_wind_factor
        min_decay = self.agent.battery_min_decay

        drain_rate = max(min_decay, base_decay + wind_factor * wind_resistance)
        drain = drain_rate * dt
        self.agent.battery = max(0.0, self.agent.battery - drain)

        self.logger.log_step(
            self.simulation_time,
            self.agent.pos.copy(),
            self.agent.current_concentration,
            self.agent.state,
            self.agent.battery,
            custom_data,
        )

        self.simulation_time += dt
        if self.agent.battery <= 0:
            print(
                f"\n[Simulator] Bateria ESGOTADA em {self.agent.pos}! Missão falhada em t={self.simulation_time:.2f}s."
            )
            return True
        if self.agent.mission_complete:
            print(
                f"\nSimulation complete! SUCESSO: completou todos os estados e contornou a pluma em {self.simulation_time:.2f}s!"
            )
            return True
        if getattr(self.agent, "low_battery_aborted", False) or getattr(
            self.agent, "incomplete_mission", False
        ):
            print(
                "\nSimulation complete! Episódio terminado SEM SUCESSO (retorno antecipado por bateria baixa ou contorno incompleto)."
            )
            return True
        return False

    def run(self):
        if hasattr(self, "episodes_to_show") and self.episodes_to_show:
            num_experiments = len(self.episodes_to_show)
        else:
            num_experiments = self.config.get("num_experiments", 5)

        stabilization_time = self.config.get("plume_stabilization_time", 0.0)
        for i in range(num_experiments):
            if getattr(self, "user_quit", False):
                break
            self.current_experiment = i
            self.reset_experiment()
            print(f"Stabilizing plume for {stabilization_time} seconds...")
            self.simulation_time = 0.0
            while self.simulation_time < stabilization_time:
                if getattr(self, "user_quit", False):
                    break
                self.plume.update(self.config["dt"], self.simulation_time)
                self.simulation_time += self.config["dt"]
                if self.config["visualize"]:
                    self._draw()

            if getattr(self, "user_quit", False):
                break

            print("Stabilization complete. Starting agent search...")
            self.simulation_time = 0.0
            step_counter = 0
            while self.simulation_time < self.config["max_time"]:
                if getattr(self, "user_quit", False):
                    break
                loop_start_time = time.time()
                if self.is_paused:
                    if self.config["visualize"]:
                        self._draw()
                    time.sleep(0.01)
                    continue

                if self.step():
                    break
                if self.config["visualize"]:
                    self._draw()
                elapsed_time = time.time() - loop_start_time
                if self.loop_time_ema == 0.0:
                    self.loop_time_ema = elapsed_time
                else:
                    self.loop_time_ema = (self.ema_alpha * elapsed_time) + (
                        (1 - self.ema_alpha) * self.loop_time_ema
                    )
                step_counter += 1
                if step_counter % 20 == 0:
                    hz = 1.0 / self.loop_time_ema if self.loop_time_ema > 0 else 0.0
                    print(
                        f"  > Sim Time: {self.simulation_time:.1f}s, Running Freq: {hz:.1f} Hz",
                        end="\r",
                    )
                if self.config["visualize"]:
                    if self.config["real_time_pacing"]:
                        wait_time = self.config["dt"] - elapsed_time
                        if wait_time > 0:
                            time.sleep(wait_time)
                    else:
                        time.sleep(0.003)  # Atrasa ligeiramente a simulação (~300 FPS max)
            print()
            if self.logger:
                self.logger.save()
        print("-" * 40 + "\nAll experiments finished.")
        if self.config["visualize"]:
            cv2.destroyAllWindows()

    def _init_visualization(self):
        self.viz_w, self.viz_h = self.config["viz_width"], self.config["viz_height"]
        self.world_w, self.world_h = (
            self.config["world_width"],
            self.config["world_height"],
        )
        cv2.namedWindow("Odour Simulation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Odour Simulation", self.viz_w, self.viz_h)

        self.agent_sprite = None
        if os.path.exists("boat_sprite.png"):
            img = cv2.imread("boat_sprite.png", cv2.IMREAD_UNCHANGED)
            if img is not None:
                agent_radius = self.config.get("agent_config", {}).get("agent_radius", 0.5)
                sprite_size_px = int((agent_radius * 8) / self.world_w * self.viz_w)
                h, w = img.shape[:2]
                if w > h:
                    new_w = sprite_size_px
                    new_h = int(h * (new_w / w))
                else:
                    new_h = sprite_size_px
                    new_w = int(w * (new_h / h))
                self.agent_sprite = cv2.resize(img, (new_w, new_h))

    def _world_to_pixel(self, pos):
        px = int(pos[0] / self.world_w * self.viz_w)
        py = int(self.viz_h - (pos[1] / self.world_h * self.viz_h))
        return px, py

    def _draw_axes(self, canvas):
        axis_color, font, font_scale, font_thickness = (
            (200, 200, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            1,
        )
        cv2.line(
            canvas,
            self._world_to_pixel((0, 0)),
            self._world_to_pixel((self.world_w, 0)),
            axis_color,
            2,
        )
        cv2.line(
            canvas,
            self._world_to_pixel((0, 0)),
            self._world_to_pixel((0, self.world_h)),
            axis_color,
            2,
        )
        for i in range(6):
            x_w = (i / 5) * self.world_w
            x_p, y_p = self._world_to_pixel((x_w, 0))
            cv2.line(canvas, (x_p, y_p - 5), (x_p, y_p + 5), axis_color, 1)
            t_s = cv2.getTextSize(str(int(x_w)), font, font_scale, font_thickness)[0]
            cv2.putText(
                canvas,
                str(int(x_w)),
                (x_p - t_s[0] // 2, y_p + 25),
                font,
                font_scale,
                axis_color,
                font_thickness,
            )
        for i in range(6):
            y_w = (i / 5) * self.world_h
            x_p, y_p = self._world_to_pixel((0, y_w))
            cv2.line(canvas, (x_p - 5, y_p), (x_p + 5, y_p), axis_color, 1)
            t_s = cv2.getTextSize(str(int(y_w)), font, font_scale, font_thickness)[0]
            cv2.putText(
                canvas,
                str(int(y_w)),
                (x_p - t_s[0] - 15, y_p + t_s[1] // 2),
                font,
                font_scale,
                axis_color,
                font_thickness,
            )
        cv2.putText(
            canvas,
            "(0,0)",
            (
                self._world_to_pixel((0, 0))[0] + 10,
                self._world_to_pixel((0, 0))[1] - 10,
            ),
            font,
            font_scale,
            axis_color,
            font_thickness,
        )
        max_x_t = f"({int(self.world_w)}, 0)"
        t_s = cv2.getTextSize(max_x_t, font, font_scale, font_thickness)[0]
        cv2.putText(
            canvas,
            max_x_t,
            (
                self._world_to_pixel((self.world_w, 0))[0] - t_s[0] - 10,
                self._world_to_pixel((self.world_w, 0))[1] - 10,
            ),
            font,
            font_scale,
            axis_color,
            font_thickness,
        )
        max_y_t = f"(0, {int(self.world_h)})"
        t_s = cv2.getTextSize(max_y_t, font, font_scale, font_thickness)[0]
        cv2.putText(
            canvas,
            max_y_t,
            (
                self._world_to_pixel((0, self.world_h))[0] + 10,
                self._world_to_pixel((0, self.world_h))[1] + 50,
            ),
            font,
            font_scale,
            axis_color,
            font_thickness,
        )
        start_c = self._world_to_pixel(self.config["agent_start_region_center"])
        start_r = int(self.config["agent_start_region_radius"] / self.world_w * self.viz_w)
        cv2.circle(canvas, start_c, start_r, (0, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def _draw(self):
        # 1. Create a canvas with marine blue background
        marine_bg_color = (110, 55, 20)  # BGR: tom de azul marinho
        canvas = np.full((self.viz_h, self.viz_w, 3), marine_bg_color, dtype=np.uint8)

        # Draw axes onto the canvas first
        canvas = self._draw_axes(canvas)

        # 2. Draw puffs
        if self.plume.puffs_array.shape[0] > 0:
            min_mass = self.config["initial_puff_mass"] * 0.01
            for puff in self.plume.puffs_array:
                if puff[3] <= min_mass:
                    continue
                pos_px, radius_px = (
                    self._world_to_pixel(puff[0:2]),
                    max(1, int(puff[2] / self.world_w * self.viz_w)),
                )
                color_val = min(255, int(puff[3] / self.config["initial_puff_mass"] * 255))
                cv2.circle(
                    canvas,
                    pos_px,
                    radius_px,
                    (color_val, color_val, color_val),
                    -1,
                    cv2.LINE_AA,
                )

        source_px = self._world_to_pixel(self.config["source_position"])
        cv2.drawMarker(canvas, source_px, (0, 0, 255), cv2.MARKER_STAR, 20, 2)

        if self.agent.first_detection_pos is not None and self.agent.state not in [
            "searching",
            "waiting",
        ]:
            detection_px = self._world_to_pixel(self.agent.first_detection_pos)
            cv2.circle(canvas, detection_px, 6, (0, 255, 255), -1, cv2.LINE_AA)  # Yellow solid dot
            cv2.putText(
                canvas,
                "detection",
                (detection_px[0] + 10, detection_px[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )
        elif getattr(self.agent, "low_battery_start_return", False) and hasattr(
            self.agent, "start_pos"
        ):
            start_px = self._world_to_pixel(self.agent.start_pos)
            cv2.circle(canvas, start_px, 6, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "start point",
                (start_px[0] + 10, start_px[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 200, 255),
                1,
            )

        # 4. Draw agent
        conc_thresh = self.config.get("agent_config", {}).get("agent_concentration_threshold", 1.0)
        agent_color = (0, 255, 0) if self.agent.current_concentration > conc_thresh else (255, 0, 0)
        agent_px = self._world_to_pixel(self.agent.pos)
        agent_radius_px = max(2, int(self.agent.radius / self.world_w * self.viz_w))

        if hasattr(self, "agent_sprite") and self.agent_sprite is not None:
            sprite = self.agent_sprite

            # Não rodar a imagem, manter a original
            rotated_sprite = sprite
            bound_w = sprite.shape[1]
            bound_h = sprite.shape[0]

            if rotated_sprite.shape[2] == 4:
                bgr = rotated_sprite[:, :, :3]
                alpha = rotated_sprite[:, :, 3] / 255.0
            else:
                bgr = rotated_sprite
                alpha = np.ones(bgr.shape[:2])

            top_left_x = agent_px[0] - bound_w // 2
            top_left_y = agent_px[1] - bound_h // 2

            y1, y2 = max(0, top_left_y), min(canvas.shape[0], top_left_y + bound_h)
            x1, x2 = max(0, top_left_x), min(canvas.shape[1], top_left_x + bound_w)

            y1o, y2o = max(0, -top_left_y), min(bound_h, canvas.shape[0] - top_left_y)
            x1o, x2o = max(0, -top_left_x), min(bound_w, canvas.shape[1] - top_left_x)

            if y1 < y2 and x1 < x2:
                alpha_crop = alpha[y1o:y2o, x1o:x2o, np.newaxis]
                canvas[y1:y2, x1:x2] = (
                    alpha_crop * bgr[y1o:y2o, x1o:x2o] + (1 - alpha_crop) * canvas[y1:y2, x1:x2]
                ).astype(np.uint8)
        else:
            cv2.circle(canvas, agent_px, agent_radius_px, agent_color, 2, cv2.LINE_AA)
            arrow_end_px = self._world_to_pixel(
                self.agent.pos + self.agent.direction * self.agent.radius * 2.5
            )
            cv2.arrowedLine(canvas, agent_px, arrow_end_px, (255, 255, 0), 2, cv2.LINE_AA)

        # 5. Draw updated text info
        font_scale = 0.55
        hz = 1.0 / self.loop_time_ema if self.loop_time_ema > 0 else 0.0
        flow_vec = self.agent.measured_flow_direction
        state_str = self.agent.state.upper()
        if getattr(self.agent, "low_battery_direct_return", False):
            state_str = "RETURNING (LOW BAT <15%)"
        elif getattr(self.agent, "low_battery_start_return", False):
            state_str = "RETURNING TO START (LOW BAT <=15%)"

        # Barra de fundo escura para o HUD no topo
        cv2.rectangle(canvas, (0, 0), (self.viz_w, 36), (15, 15, 20), -1)

        wind_arrow = "->" if flow_vec[0] >= 0 else "<-"
        total_eps = len(getattr(self, "episodes_to_show", [])) or 5
        curr_ep_idx = self.current_experiment + 1
        ep_prefix = (
            f"Ep #{self.current_ep_num} ({curr_ep_idx}/{total_eps}) | "
            if getattr(self, "current_ep_num", None) is not None
            else f"Episódio {curr_ep_idx}/{total_eps} | "
        )
        info_left = f"{ep_prefix}Time: {self.simulation_time:.1f}s | State: {state_str} | Conc: {self.agent.current_concentration:.2f} | Bat: {self.agent.battery:.1f}% | Vento: {wind_arrow} ({flow_vec[0]:.2f}, {flow_vec[1]:.2f}) m/s"
        info_right = f"Agent Pos: ({self.agent.pos[0]:.1f}, {self.agent.pos[1]:.1f})"
        hud_color = (0, 255, 255)  # Amarelo de alto contraste sobre o fundo marinho

        cv2.putText(
            canvas,
            info_left,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            hud_color,
            1,
            cv2.LINE_AA,
        )

        t_s_r = cv2.getTextSize(info_right, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

        cv2.putText(
            canvas,
            info_right,
            (self.viz_w - t_s_r[0] - 10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            hud_color,
            1,
            cv2.LINE_AA,
        )

        # 6. Display
        cv2.imshow("Odour Simulation", canvas)
        wait_ms = 30 if self.is_paused else 1
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == 32:  # Barra de espaço: comuta entre pausa e retomada
            self.is_paused = not self.is_paused
        elif key == ord("n") or key == ord(
            "N"
        ):  # Tecla 'n': avançar imediatamente para o próximo episódio
            self.skip_episode = True
            self.is_paused = False
        elif key == 27 or key == ord("q"):  # ESC (27) ou 'q': terminar e fechar a simulação
            self.user_quit = True
            self.is_paused = False
            self.simulation_time = self.config["max_time"]
            cv2.destroyAllWindows()

        try:
            if cv2.getWindowProperty("Odour Simulation", cv2.WND_PROP_VISIBLE) < 1:
                self.user_quit = True
                self.is_paused = False
                self.simulation_time = self.config["max_time"]
        except Exception:
            pass
