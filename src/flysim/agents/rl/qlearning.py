"""
Módulo de Aprendizagem por Reforço (Q-Learning) para Navegação em Plumas de Odor.

Este módulo implementa o agente QLearningAgent, responsável por aprender
a política ótima de navegação do veículo autónomo (AUV) para localizar a fonte
de emissão da pluma através de amostragem sensorial e discretização de estados.
"""

from __future__ import annotations

import datetime
import logging
import os
import pickle
import random

import numpy as np

from flysim.agents.base import Agent
from flysim.plume.kernels import get_plume_center_y


class QLearningAgent:
    """
    Agente de Aprendizagem por Reforço baseado no algoritmo Q-Learning tabular.

    O espaço de estados contínuo (posição, concentração de odor, direção da corrente,
    estado da missão e nível de bateria) é discretizado em intervalos finitos (bins).
    """

    def __init__(
        self,
        env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.998,
        min_exploration_rate: float = 0.01,
        state_bins: dict = None,
    ):
        """
        Inicializa o agente Q-Learning com os respetivos hiperparâmetros.

        Argumentos:
            env: Ambiente de simulação compatível com Gymnasium (PlumeRLWrapper).
            learning_rate: Taxa de aprendizagem (alpha), controla o peso das novas informações.
            discount_factor: Fator de desconto temporal (gamma), pondera recompensas futuras.
            exploration_rate: Taxa inicial de exploração estocástica (epsilon).
            exploration_decay: Fator multiplicativo de decaimento do epsilon por episódio.
            min_exploration_rate: Valor mínimo admissível para a taxa de exploração.
            state_bins: Dicionário com o número de partições discretas para cada dimensão.
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate

        # Configuração padrão de partições para discretização do estado contínuo
        self.state_bins = state_bins or {
            "x": 20,  # Partições ao longo do comprimento (Eixo X)
            "y": 10,  # Partições ao longo da largura (Eixo Y)
            "conc": 3,  # Níveis de concentração de odor (baixa, média, alta)
            "wind": 4,  # Quadrantes da direção do escoamento/vento (4 pontos cardeais)
            "battery": 3,  # Níveis de carga da bateria (baixa, média, alta)
        }

        # Tabela Q: mapeia tuplos de estado discreto para arrays de valores Q por ação
        self.q_table = {}

        # Histórico de métricas de treino para análise estatística
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_history = []
        self.saved_paths = {}
        self.saved_detection_points = {}
        self._cached_legacy = False

    def _is_legacy_model(self) -> bool:
        """
        Deteta se a Q-Table em memória foi carregada de um ficheiro legado
        (onde as chaves tinham menos de 7 elementos: 4, 5 ou 6).
        Modelos novos em treino utilizam sempre o formato moderno de 7 dimensões.
        """
        if getattr(self, "_cached_legacy", None) is not None:
            return self._cached_legacy
        if not self.q_table:
            return False
        first_key = next(iter(self.q_table.keys()))
        self._cached_legacy = bool(len(first_key) < 7)
        return self._cached_legacy

    def _state_key(self, state) -> tuple:
        """
        Converte o vetor contínuo de observação num tuplo discreto para indexação na Q-Table.

        Estrutura do estado:
            [norm_x, norm_y, conc, flow_x, flow_y, mission_state, battery_level]
        Todos os valores encontram-se normalizados no intervalo [0, 1].
        """
        state = np.asarray(state, dtype=np.float32)

        # 1. Discretização espacial (Posição X e Y)
        x_b = int(np.clip(state[0] * self.state_bins["x"], 0, self.state_bins["x"] - 1))
        y_b = int(np.clip(state[1] * self.state_bins["y"], 0, self.state_bins["y"] - 1))

        # 2. Discretização da concentração de odor recebida no sensor
        conc = state[2]
        legacy = self._is_legacy_model()

        if legacy:
            if conc <= 0.33:
                c_b = 0  # Concentração nula ou muito baixa
            elif conc <= 0.67:
                c_b = 1  # Concentração moderada (limiar da pluma)
            else:
                c_b = 2  # Concentração elevada (núcleo da pluma)
        else:
            # Calibração para deteção de fronteira (threshold físico = 1.0 / 10.0 = 0.10):
            # 0: Ar limpo / Fora da pluma (< 0.5 unidades)
            # 1: Fronteira / Limiar da pluma (0.5 a 2.0 unidades)
            # 2: Núcleo denso da pluma (> 2.0 unidades)
            if conc < 0.05:
                c_b = 0
            elif conc <= 0.20:
                c_b = 1
            else:
                c_b = 2

        # 3. Discretização da orientação do escoamento do fluido (4 quadrantes)
        fx, fy = state[3], state[4]
        angle = np.arctan2(fy, fx)
        wind_dir = int(np.round(angle / (np.pi / 2))) % 4

        # Compatibilidade retroativa com modelos anteriores (tuplos de 4 ou 5 elementos)
        if self.q_table:
            first_key = next(iter(self.q_table.keys()))
            if len(first_key) == 4:
                return (x_b, y_b, c_b, wind_dir)
            elif len(first_key) == 5:
                mission_state_bin = 0
                if len(state) > 5:
                    mission_state_bin = int(np.round(state[5]))
                return (x_b, y_b, c_b, wind_dir, mission_state_bin)

        # 4. Estado da missão (0: Procura, 1: Contorno/Seguimento, 2: Regresso Contorno, 3: Regresso Fecho)
        mission_state_bin = 0
        if len(state) > 5:
            raw_m = float(state[5])
            if legacy:
                mission_state_bin = 1 if raw_m >= 1.5 else int(np.round(raw_m))
            else:
                mission_state_bin = int(np.clip(np.round(raw_m), 0, 3))

        # 5. Nível de carga da bateria (crítico, operacional, elevado)
        battery_bin = 0
        if len(state) > 6:
            battery_val = state[6]
            if battery_val <= 0.3:
                battery_bin = 0  # Bateria baixa (<30%)
            elif battery_val <= 0.7:
                battery_bin = 1  # Bateria média
            else:
                battery_bin = 2  # Bateria alta

        # 6. Sentido de varrimento transversal (Cast: 0=Subindo, 1=Descendo)
        if legacy:
            return (x_b, y_b, c_b, wind_dir, mission_state_bin, battery_bin)

        cast_dir_bin = 0
        if len(state) > 7:
            cast_dir_bin = int(np.clip(np.round(state[7]), 0, 1))

        return (x_b, y_b, c_b, wind_dir, mission_state_bin, battery_bin, cast_dir_bin)

    def choose_action(self, state, greedy: bool = False) -> int:
        """
        Seleciona a ação a executar segundo a política epsilon-greedy.

        Argumentos:
            state: Vetor de estado atual retornado pelo ambiente.
            greedy: Se True, força a seleção determinística da melhor ação conhecida.

        Retorna:
            Índice da ação selecionada (0: Cima, 1: Baixo, 2: Esquerda, 3: Direita).
        """
        key = self._state_key(state)

        # Fase de Exploração: escolha estocástica aleatória
        if not greedy and random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.q_table.get(key, None)
            if q_values is None or np.all(q_values == 0):
                # Estado não aprendido na Q-Table: decisão aleatória uniforme (sem regras hardcoded)
                return random.randint(0, self.env.action_space.n - 1)
            return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done, truncated=False):
        """
        Atualiza o valor Q na tabela através da Equação de Bellman para Q-Learning:
        Q(s, a) = Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
        """
        key = self._state_key(state)
        next_key = self._state_key(next_state)

        # Inicializa o estado com zeros caso seja a primeira visita
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.env.action_space.n)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.env.action_space.n)

        current_q = self.q_table[key][action]

        # Tratamento do estado terminal (sem estimativa de recompensas futuras)
        if done or truncated:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_key])
            target_q = reward + self.gamma * max_next_q

        # Aplicação da taxa de aprendizagem à discrepância temporal
        self.q_table[key][action] += self.lr * (target_q - current_q)

    def train(self, env, num_episodes=500, print_every=50):
        """
        Executa o ciclo completo de treino do agente com registo de telemetria e rotas.

        Argumentos:
            env: Instância do ambiente PlumeRLWrapper.
            num_episodes: Quantidade de episódios a treinar.
            print_every: Intervalo de episódios para apresentação do progresso no terminal.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pasta_logs = os.path.join(base_dir, "logs")
        os.makedirs(pasta_logs, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(pasta_logs, f"treino_{timestamp}.log")
        self.current_log_file = log_filename

        logger = logging.getLogger(f"QLearningTrain_{timestamp}")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(fh)

        logger.info("=== INÍCIO DO TREINO Q-LEARNING ===")
        logger.info(f"Episódios a treinar: {num_episodes}")
        logger.info(
            f"Taxa de Aprendizagem (Alpha): {self.lr}, Fator de Desconto (Gamma): {self.gamma}"
        )
        logger.info(
            f"Epsilon Inicial: {self.epsilon}, Decaimento: {self.epsilon_decay}, Mínimo: {self.min_epsilon}"
        )
        logger.info(f"Partições de Estado (Bins): {self.state_bins}")

        print(f"\n[INFO] Ficheiro de registo detalhado do treino criado em: {log_filename}")
        print(f"\n{'=' * 70}")
        print(f"  TREINO DO AGENTE Q-LEARNING — {num_episodes} episódios")
        print(
            f"  Partições de estado: {self.state_bins['x']}×{self.state_bins['y']}×{self.state_bins['conc']}×{self.state_bins['wind']}"
        )
        print(f"{'=' * 70}\n")

        episodios_iniciais = len(self.episode_rewards)

        for ep_relativo in range(1, num_episodes + 1):
            ep_absoluto = episodios_iniciais + ep_relativo

            state, info = env.reset()
            current_path = [
                (
                    float(env.pos[0] / env.config["world_width"]),
                    float(env.pos[1] / env.config["world_height"]),
                )
            ]
            done = truncated = False
            total_reward = 0
            step = 0

            # Execução dos passos do episódio
            while not (done or truncated):
                action = self.choose_action(state)

                next_state, reward, done, truncated, info = env.step(action)
                current_path.append(
                    (
                        float(env.pos[0] / env.config["world_width"]),
                        float(env.pos[1] / env.config["world_height"]),
                    )
                )

                self.update(state, action, reward, next_state, done, truncated)

                state = next_state
                total_reward += reward
                step += 1

            # Registo de estatísticas
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step)
            is_success = info.get("at_goal", False)
            self.success_history.append(1 if is_success else 0)

            # Registo no ficheiro de log
            logger.info(
                f"Episódio {ep_absoluto} | Recompensa: {total_reward:.2f} | "
                f"Passos: {step} | Sucesso: {is_success} | "
                f"Epsilon: {self.epsilon:.4f} | Estados Q: {len(self.q_table)}"
            )

            # Decaimento adaptativo para convergir epsilon a min_epsilon nos últimos 25-30% dos episódios
            decay_target_ep = max(50, int(0.70 * num_episodes))
            computed_decay = (self.min_epsilon / 1.0) ** (1.0 / decay_target_ep)
            effective_decay = min(self.epsilon_decay, computed_decay)

            # Decaimento progressivo da taxa de exploração estocástica
            if self.epsilon > self.min_epsilon:
                self.epsilon = max(self.min_epsilon, self.epsilon * effective_decay)

            # Paragem antecipada (Early Stopping) se a taxa de sucesso for 80% ou mais em 300 episódios seguidos
            if len(self.success_history) >= 300:
                avg_success_300 = np.mean(self.success_history[-300:])
                if avg_success_300 >= 0.80:
                    logger.info(
                        f"Convergência atingida com {avg_success_300 * 100:.1f}% de sucesso nos últimos 300 episódios no ep {ep_absoluto}. Paragem antecipada."
                    )
                    print(
                        f"\n[Early Stopping] Convergência verificada ({avg_success_300 * 100:.1f}% de sucesso nos últimos 300 episódios) no episódio {ep_absoluto}!"
                    )
                    break

            # Armazenamento seletivo de trajetórias para inspeção no mapa do Dashboard
            save_interval = 1 if num_episodes <= 100 else (5 if num_episodes <= 500 else 20)
            if (
                ep_relativo <= 10
                or ep_relativo > num_episodes - 25
                or ep_absoluto % save_interval == 0
            ):
                self.saved_paths[ep_absoluto] = current_path
                det_pt = getattr(env, "first_detection_pos", None)
                if det_pt is not None:
                    self.saved_detection_points[ep_absoluto] = (float(det_pt[0]), float(det_pt[1]))
                else:
                    self.saved_detection_points[ep_absoluto] = None

            # Apresentação do progresso periódico na consola
            if ep_relativo % print_every == 0:
                w = min(print_every, ep_relativo)
                avg_reward = np.mean(self.episode_rewards[-w:])
                avg_length = np.mean(self.episode_lengths[-w:])
                avg_success = np.mean(self.success_history[-w:]) * 100

                print(
                    f"  Ep {ep_absoluto:>5} (+{ep_relativo}) | "
                    f"Recompensa Média: {avg_reward:>7.2f} | "
                    f"Passos Médios: {avg_length:>6.1f} | "
                    f"Sucesso: {avg_success:>5.1f}% | "
                    f"Eps: {self.epsilon:.4f} | "
                    f"Estados Q: {len(self.q_table):>5}"
                )

        logger.info("=== CONCLUSÃO DO TREINO ===")
        logger.removeHandler(fh)
        fh.close()

    def save(self, filename=None):
        """
        Serializa e guarda o modelo treinado, a Q-Table e as métricas num ficheiro .pkl.

        Argumentos:
            filename: Nome opcional para o ficheiro (se None, gera carimbo de data/hora).

        Retorna:
            Caminho absoluto ou relativo do ficheiro gravado.
        """
        pasta = "modelos"
        os.makedirs(pasta, exist_ok=True)

        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qlearning_plume_{timestamp}.pkl"
        elif not filename.endswith(".pkl"):
            filename = f"{filename}.pkl"

        caminho_completo = os.path.join(pasta, filename)

        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "state_bins": self.state_bins,
            "hyperparameters": {
                "learning_rate": self.lr,
                "discount_factor": self.gamma,
                "exploration_decay": self.epsilon_decay,
                "min_epsilon": self.min_epsilon,
            },
            "stats": {
                "rewards": self.episode_rewards,
                "lengths": self.episode_lengths,
                "successes": self.success_history,
                "paths": getattr(self, "saved_paths", {}),
                "detection_points": getattr(self, "saved_detection_points", {}),
            },
            "source_pos": getattr(self.env, "config", {}).get("source_position", (20.0, 25.0)),
            "world_size": (
                getattr(self.env, "config", {}).get("world_width", 100.0),
                getattr(self.env, "config", {}).get("world_height", 50.0),
            ),
        }

        with open(caminho_completo, "wb") as f:
            pickle.dump(data, f)

        # Guarda e sincroniza o ficheiro de log do treino na pasta logs
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pasta_logs = os.path.join(base_dir, "logs")
        os.makedirs(pasta_logs, exist_ok=True)
        if getattr(self, "current_log_file", None) and os.path.exists(self.current_log_file):
            import shutil

            base_model_name = os.path.splitext(os.path.basename(caminho_completo))[0]
            log_model_dest = os.path.join(pasta_logs, f"{base_model_name}.log")
            if os.path.abspath(log_model_dest) != os.path.abspath(self.current_log_file):
                try:
                    shutil.copyfile(self.current_log_file, log_model_dest)
                    print(f"    Registo de treino sincronizado em: {log_model_dest}")
                except Exception:
                    pass

        print(f"\n[+] Modelo e estatísticas gravados com sucesso em: {caminho_completo}")
        print(f"    Episódios acumulados: {len(self.episode_rewards)}")
        print(f"    Estados descobertos na Q-Table: {len(self.q_table)}")
        print(f"    Taxa final de sucesso: {np.mean(self.success_history[-100:]) * 100:.1f}%")

        return caminho_completo

    def load(self, filepath):
        """
        Carrega um modelo pré-treinado a partir de um ficheiro .pkl.

        Argumentos:
            filepath: Caminho para o ficheiro do modelo a carregar.

        Retorna:
            True em caso de sucesso, False se ocorrer algum erro ou o ficheiro não existir.
        """
        if not os.path.exists(filepath):
            print(f"[-] AVISO: Ficheiro {filepath} não encontrado.")
            return False

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            self.q_table = dict(data["q_table"])
            self._cached_legacy = None
            self.epsilon = data.get("epsilon", self.epsilon)
            self.state_bins = data.get("state_bins", self.state_bins)
            self.loaded_model_name = os.path.basename(filepath)

            if "stats" in data:
                self.episode_rewards = data["stats"].get("rewards", [])
                self.episode_lengths = data["stats"].get("lengths", [])
                self.success_history = data["stats"].get("successes", [])
                self.saved_paths = data["stats"].get("paths", {})
                self.saved_detection_points = data["stats"].get("detection_points", {})

            print(f"[+] Modelo carregado com sucesso de {filepath}")
            print(f"    Estados na Q-Table: {len(self.q_table)}")
            print(f"    Histórico de episódios: {len(self.episode_rewards)}")

            return True

        except Exception as e:
            print(f"[-] Erro ao carregar o modelo: {e}")
            return False

    def get_statistics(self):
        """Retorna um resumo com as principais métricas de desempenho do treino."""
        if not self.episode_rewards:
            return None

        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards),
            "avg_length": np.mean(self.episode_lengths),
            "success_rate": np.mean(self.success_history),
            "q_table_size": len(self.q_table),
            "current_epsilon": self.epsilon,
        }


class QLearningSimAgent(Agent):
    """O Agente que utiliza o modelo Q-Learning para tomar decisões no Simulador principal."""

    def __init__(self, config):
        if "agent_radius" not in config and "agent_config" in config:
            config = {**config, **config.get("agent_config", {})}
        super().__init__(config)
        from flysim.environment.plume_env import PlumeRLWrapper

        dummy_env = PlumeRLWrapper(config)

        self.q_agent = QLearningAgent(dummy_env)
        self.model_name = "Desconhecido"

        # 1. Se foi passado o agente já treinado diretamente em memória:
        if config.get("trained_q_agent") is not None:
            self.q_agent = config["trained_q_agent"]
            self.model_name = getattr(
                self.q_agent, "loaded_model_name", "Treino Atual (Em Memória)"
            )
            print(
                f"\n[+] Simulador: Q-Table em memória conectada ({len(self.q_agent.q_table)} estados aprendidos)."
            )
        else:
            # 2. Carregar ficheiro .pkl especificado com resolução robusta de caminho
            chosen_model = self.config.get("model_to_load", "qlearning_plume.pkl")
            caminhos_tentar = [
                chosen_model,
                os.path.join("modelos", chosen_model),
                f"{chosen_model}.pkl" if not chosen_model.endswith(".pkl") else chosen_model,
                os.path.join("modelos", f"{chosen_model}.pkl")
                if not chosen_model.endswith(".pkl")
                else os.path.join("modelos", chosen_model),
            ]
            carregou = False
            for p in caminhos_tentar:
                if p and os.path.exists(p):
                    if self.q_agent.load(p):
                        self.model_name = os.path.basename(p)
                        self.q_agent.loaded_model_name = self.model_name
                        carregou = True
                        saved_meander = getattr(self.q_agent, "meander_model", None)
                        if saved_meander:
                            self.config["meander_model"] = saved_meander
                        elif "sinusoid" in self.model_name.lower():
                            self.config["meander_model"] = "sum_of_sinusoids"
                        else:
                            self.config["meander_model"] = "linear"
                        print(
                            f"\n[+] Simulador: Modelo '{p}' carregado com sucesso ({len(self.q_agent.q_table)} estados na Q-Table). Tipo de pluma: {self.config.get('meander_model')}."
                        )
                        break
            if not carregou:
                self.model_name = os.path.basename(chosen_model) if chosen_model else "Vazio"
                print(
                    f"\n[-] AVISO: Não foi possível encontrar o ficheiro do modelo '{chosen_model}'. Q-Table iniciada a zeros ({len(self.q_agent.q_table)} estados)!"
                )

    def reset(self):
        super().reset()
        self.current_action = None
        self.cast_dir = int(np.random.choice([0, 1]))
        self.return_at_detection_x = False
        self.return_contour_completed = False
        self.distal_tip_entered = False
        self.theta = None
        self.circling_step = 0
        self.circling_angle = 0.0
        self.recent_sim_positions = []
        self.recent_sim_actions = []

    def run_search_algorithm(self, dt):
        """
        Executa as decisões do agente no simulador COM BASE NA Q-TABLE TREINADA,
        com proteção de momento e anti-chattering para evitar aprisionamento determinístico
        em ciclos de 2 estados / reversões de 180° entre células vizinhas.
        """
        conc_thresh = float(
            self.config.get("agent_config", {}).get("agent_concentration_threshold", 1.0)
        )
        source_pos = np.array(self.config["source_position"])
        dist_to_source = np.linalg.norm(self.pos - source_pos)
        success_distance = float(self.config.get("agent_config", {}).get("success_distance", 2.0))

        # --- Máquina de Estados da Missão ---
        if self.state == "searching":
            y_min_sweep = 0.30 * self.config["world_height"]
            y_max_sweep = 0.70 * self.config["world_height"]
            if self.pos[1] >= y_max_sweep:
                self.cast_dir = 1
            elif self.pos[1] <= y_min_sweep:
                self.cast_dir = 0

            if self.current_concentration > conc_thresh:
                self.state = "following_boundary"
                self.first_detection_pos = self.pos.copy()
                local_center_y = get_plume_center_y(
                    self.pos[0], self.simulation_time, self.plume, self.config
                )
                self.skirting_side_ida = 1.0 if self.pos[1] >= local_center_y else -1.0

        elif self.state == "following_boundary":
            if (
                dist_to_source < success_distance
                or (self.pos[0] <= source_pos[0] + 1.8 and dist_to_source < success_distance * 1.8)
            ) and not self.source_reached:
                self.source_reached = True
                self.circling_step = 0

        # Contorno circular da fonte
        if self.state == "following_boundary" and self.source_reached:
            target_pos = self.get_circling_target(dt)
            return target_pos, {"action": "circling", "state": "circling_source"}

        # Estado Returning: verificação do alcance da ponta e conclusão de contorno
        if self.state == "returning":
            det_pos = (
                self.first_detection_pos
                if self.first_detection_pos is not None
                else np.array([90.0, 25.0])
            )
            side_ida = getattr(self, "skirting_side_ida", 1.0)
            if side_ida is None:
                side_ida = 1.0
            local_center_y = get_plume_center_y(
                self.pos[0], self.simulation_time, self.plume, self.config
            )
            is_sinusoid = self.config.get("meander_model", "ou").lower() in [
                "sum_of_sinusoids",
                "sinusoid",
                "sinusoidal",
            ]

            dist_to_det = np.linalg.norm(self.pos - det_pos)

            if is_sinusoid:
                # Pluma Sinusoidal: ao alcançar o mesmo X do ponto de deteção, desloca-se diretamente para det_pos!
                opposite_side = -side_ida
                is_on_opposite_side = (
                    (self.pos[1] >= local_center_y)
                    if opposite_side == 1.0
                    else (self.pos[1] <= local_center_y)
                )
                if self.pos[0] >= det_pos[0] - 0.5 and (
                    is_on_opposite_side or self.current_concentration >= 0.15 * conc_thresh
                ):
                    self.return_tip_reached = True
                    self.return_contour_completed = True

                if dist_to_det < success_distance:
                    self.mission_complete = True
                    self.return_contour_completed = True

                if getattr(self, "return_tip_reached", False):
                    goal_vec = det_pos - self.pos
                    goal_norm = np.linalg.norm(goal_vec)
                    move_vec = goal_vec / goal_norm if goal_norm > 1e-6 else np.array([0.0, 0.0])

                    target_pos = self.pos + move_vec * self.speed * dt
                    target_pos[0] = np.clip(target_pos[0], 0.0, self.config["world_width"])
                    target_pos[1] = np.clip(target_pos[1], 0.0, self.config["world_height"])

                    desired_theta = np.arctan2(move_vec[1], move_vec[0])
                    if self.theta is None:
                        self.theta = desired_theta
                    else:
                        diff = (desired_theta - self.theta + np.pi) % (2 * np.pi) - np.pi
                        self.theta += np.clip(diff, -6.0 * dt, 6.0 * dt)

                    action = 1 if det_pos[1] > self.pos[1] else 0
                    return target_pos, {"action": action, "theta": self.theta}
            else:
                # Pluma Linear: contorno completo de toda a pluma até à ponta distal (x_turn ~80m)
                x_turn = min(80.0, max(det_pos[0] + 16.0, 78.0))

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

                if getattr(self, "return_tip_reached", False) and dist_to_det < success_distance:
                    self.mission_complete = True
                    self.return_contour_completed = True

                # Quando estiver no mesmo X que o ponto de deteção e após o contorno total da pluma linear:
                if getattr(self, "return_tip_reached", False) and self.pos[0] <= det_pos[0] + 0.5:
                    goal_vec = det_pos - self.pos
                    goal_norm = np.linalg.norm(goal_vec)
                    move_vec = goal_vec / goal_norm if goal_norm > 1e-6 else np.array([0.0, 0.0])

                    target_pos = self.pos + move_vec * self.speed * dt
                    target_pos[0] = np.clip(target_pos[0], 0.0, self.config["world_width"])
                    target_pos[1] = np.clip(target_pos[1], 0.0, self.config["world_height"])

                    desired_theta = np.arctan2(move_vec[1], move_vec[0])
                    if self.theta is None:
                        self.theta = desired_theta
                    else:
                        diff = (desired_theta - self.theta + np.pi) % (2 * np.pi) - np.pi
                        self.theta += np.clip(diff, -6.0 * dt, 6.0 * dt)

                    action = 1 if det_pos[1] > self.pos[1] else 0
                    return target_pos, {"action": action, "theta": self.theta}

                # Contorno distal fluido da margem direita: viragem suave em redor dos puffs para mudar de lado:
                if getattr(self, "distal_tip_entered", False) and not getattr(
                    self, "return_tip_reached", False
                ):
                    turn_act = 1 if side_ida == 1.0 else 0
                    if self.current_concentration > 1.5 * conc_thresh:
                        action = 2  # passo em -x para contornar o bordo do puff pelo exterior
                    else:
                        action = turn_act

                    dirs = {
                        0: np.array([0.0, -1.0]),
                        1: np.array([0.0, 1.0]),
                        2: np.array([-1.0, 0.0]),
                        3: np.array([1.0, 0.0]),
                    }
                    move_vec = dirs[action]
                    target_pos = self.pos + move_vec * self.speed * dt
                    target_pos[0] = np.clip(target_pos[0], 0.0, self.config["world_width"])
                    target_pos[1] = np.clip(target_pos[1], 0.0, self.config["world_height"])

                    desired_theta = np.arctan2(move_vec[1], move_vec[0])
                    if self.theta is None:
                        self.theta = desired_theta
                    else:
                        diff = (desired_theta - self.theta + np.pi) % (2 * np.pi) - np.pi
                        self.theta += np.clip(diff, -6.0 * dt, 6.0 * dt)

                    return target_pos, {"action": action, "theta": self.theta}

        # --- Vetor de Observação para a Q-Table ---
        obs_x = self.pos[0] / self.config["world_width"]
        obs_y = self.pos[1] / self.config["world_height"]
        conc_val = np.clip(self.current_concentration / 10.0, 0, 1)

        vy = self.measured_flow_direction[1]
        flow_x_normalized = np.clip(self.measured_flow_direction[0] / 5.0, -1, 1)
        flow_y_normalized = np.clip(vy / 5.0, -1, 1)

        if self.state == "searching":
            conc_val = 0.0
            state_val = 0.0
            border_side_or_cast = float(getattr(self, "cast_dir", 0))
        elif self.state == "following_boundary":
            state_val = 1.0
            border_side_or_cast = 1.0 if getattr(self, "skirting_side_ida", 1.0) == 1.0 else 0.0
        else:
            is_sinusoid = self.config.get("meander_model", "ou").lower() in [
                "sum_of_sinusoids",
                "sinusoid",
                "sinusoidal",
            ]
            det_pos = (
                self.first_detection_pos
                if self.first_detection_pos is not None
                else np.array([90.0, 25.0])
            )
            if is_sinusoid:
                # Pluma Sinusoidal: ao alcançar o mesmo X de det_pos, transita para docking
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
                # Pluma Linear: contorno completo
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

        obs = np.array(
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

        # Escolha da ação com base na Q-Table aprendida, com filtro de anti-bloqueio não-reversivo
        key = self.q_agent._state_key(obs)
        q_vals = self.q_agent.q_table.get(key, None)
        opposite_dirs = {0: 1, 1: 0, 2: 3, 3: 2}

        if q_vals is None or np.all(q_vals == 0):
            action = int(np.random.choice(4))
        else:
            sorted_acts = [int(a) for a in np.argsort(q_vals)[::-1]]
            best = sorted_acts[0]

            if self.state == "searching":
                action = best
            else:
                is_ping_pong = False
                if hasattr(self, "recent_sim_actions") and len(self.recent_sim_actions) >= 4:
                    acts = self.recent_sim_actions[-4:]
                    if (
                        acts[0] == acts[2]
                        and acts[1] == acts[3]
                        and acts[0] == opposite_dirs.get(acts[1])
                    ):
                        if (
                            hasattr(self, "recent_sim_positions")
                            and len(self.recent_sim_positions) >= 4
                        ):
                            disp = np.linalg.norm(self.pos - self.recent_sim_positions[-4])
                            if disp < 0.15:
                                is_ping_pong = True

                is_stalled = False
                if hasattr(self, "recent_sim_positions") and len(self.recent_sim_positions) >= 20:
                    disp = np.linalg.norm(self.pos - self.recent_sim_positions[-20])
                    if disp < 0.20:
                        is_stalled = True

                if is_ping_pong or is_stalled:
                    reversing_act = (
                        opposite_dirs.get(self.recent_sim_actions[-1])
                        if hasattr(self, "recent_sim_actions") and self.recent_sim_actions
                        else None
                    )
                    action = best
                    for alt in sorted_acts:
                        if alt != reversing_act:
                            action = alt
                            break
                else:
                    action = best

                # Quando próximo da fonte em contorno de ida, proibir recuo a jusante (ação 3)
                if self.state == "following_boundary" and dist_to_source < success_distance * 3.0:
                    if action == 3:
                        for alt in sorted_acts:
                            if alt != 3:
                                action = alt
                                break

                # No estado returning, proibir recuo a montante em direção à fonte (ação 2)
                if self.state == "returning" and action == 2:
                    for alt in sorted_acts:
                        if alt != 2:
                            action = alt
                            break

        self.current_action = action
        if not hasattr(self, "recent_sim_actions"):
            self.recent_sim_actions = []
        self.recent_sim_actions.append(action)
        if len(self.recent_sim_actions) > 25:
            self.recent_sim_actions.pop(0)

        if not hasattr(self, "recent_sim_positions"):
            self.recent_sim_positions = []
        self.recent_sim_positions.append(self.pos.copy())
        if len(self.recent_sim_positions) > 25:
            self.recent_sim_positions.pop(0)

        if self.state == "searching":
            search_angle_deg = float(self.config.get("agent_config", {}).get("search_angle", 75.0))
            crab_upwind = float(np.cos(np.deg2rad(search_angle_deg)))
            crosswind = float(np.sin(np.deg2rad(search_angle_deg)))
            dirs = {
                0: np.array([-crab_upwind, -crosswind]),
                1: np.array([-crab_upwind, crosswind]),
                2: np.array([-1.0, 0.0]),
                3: np.array([1.0, 0.0]),
            }
        else:
            dirs = {
                0: np.array([0.0, -1.0]),
                1: np.array([0.0, 1.0]),
                2: np.array([-1.0, 0.0]),
                3: np.array([1.0, 0.0]),
            }

        move_vec = dirs[action]
        target_pos = self.pos + move_vec * self.speed * dt
        target_pos[0] = np.clip(target_pos[0], 0.0, self.config["world_width"])
        target_pos[1] = np.clip(target_pos[1], 0.0, self.config["world_height"])

        # Orientação visual contínua para renderização realista do sprite
        desired_theta = np.arctan2(move_vec[1], move_vec[0])
        if self.theta is None:
            self.theta = desired_theta
        else:
            diff = (desired_theta - self.theta + np.pi) % (2 * np.pi) - np.pi
            self.theta += np.clip(diff, -6.0 * dt, 6.0 * dt)

        return target_pos, {"action": action, "theta": self.theta}


AGENT_REGISTRY = {
    "QLearningAgent": QLearningSimAgent,
}
