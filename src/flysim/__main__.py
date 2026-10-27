from __future__ import annotations

import glob
import json
import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from flysim.agents.rl.qlearning import QLearningAgent
from flysim.dashboard.app import DashboardApp
from flysim.environment.plume_env import PlumeRLWrapper
from flysim.simulation.runner import Simulator


def main():
    default_config = {
        "dt": 0.1,
        "max_time": 300.0,
        "save_path": "experiments",
        "experiment_name": "bio_inspired_test",
        "visualize": True,
        "real_time_pacing": False,
        "viz_width": 1200,
        "viz_height": 600,
        "world_width": 100.0,
        "world_height": 50.0,
        "source_position": (20.0, 25.0),
        "emission_rate": 5,
        "initial_puff_mass": 450.0,
        "puff_decay_rate": 0.02,
        "puff_expansion_rate": 0.20,
        "battery_capacity": 100.0,
        "battery_base_decay": 0.08,
        "battery_wind_factor": 0.80,
        "battery_min_decay": 0.02,
        "concentration_model": "gaussian",
        "num_experiments": 5,
        "plume_stabilization_time": 0.0,
        "agent_initial_wait_time": 8.0,
        "agent_start_region_center": (90.0, 25.0),
        "agent_start_region_radius": 2.0,
        # --- MEANDER MODEL CONFIGURATION ---
        "meander_model": "OU",
        "ou_meander_config": {
            "intensity": 0.4,
            "timescale": 0.15,
        },
        "sinusoid_meander_config": {
            "num_harmonics": 10,
            "v_var": 0.8,
            "integral_length_scale_L": 20.0,
            "drift_correction_timescale": 10,
            "amplitude": 1.5,  # Elegant, stable spatial wave sweep
            "frequency": 0.08,
        },
        # 'sinusoid_meander_config': {
        # 'num_harmonics': 10,
        # 'v_var': 0.4,  # <-- Strong, but not excessive
        # 'integral_length_scale_L': 25.0,
        # 'drift_correction_timescale': 45.0
        # },
        # --- TURBULENCE INTENSITY ---
        "turbulence_intensity": 0.15,
        # 'turbulence_intensity': 0.7,
        # --- MEAN WIND VELOCITY ---
        "mean_wind_velocity": (1.5, 0),
        "agent_class": "QLearningAgent",
        "agent_config": {
            "agent_radius": 0.5,
            "agent_speed": 2.4,
            "success_distance": 2.0,
            "agent_concentration_threshold": 1.0,
            "surge_length": 4.0,
            "cast_length": 25.0,
            "cast_angle": 75.0,
            "search_length": 25.0,
            "search_angle": 75.0,
            "cast_length_increase_factor": 0.1,
            "confirmation_steps": 3,
        },
    }

    config_file_path = "config.json"

    if not os.path.exists(config_file_path):
        with open(config_file_path, "w") as f:
            json.dump(default_config, f, indent=4)

    with open(config_file_path) as f:
        config = json.load(f)

    pasta_modelos = "modelos"
    os.makedirs(pasta_modelos, exist_ok=True)
    modelos = glob.glob(os.path.join(pasta_modelos, "*.pkl"))
    modelos.sort(key=os.path.getmtime, reverse=True)
    nomes_modelos = [os.path.basename(m) for m in modelos]

    escolhas = {
        "acao": None,
        "tipo_treino": "novo",
        "modelo_alvo": None,
        "nome_guardar": None,
        "episodios": 3000,
    }
    # --- INÍCIO DA INTERFACE GRÁFICA (TEMA CLARO E PROFISSIONAL) ---
    root = tk.Tk()
    root.title("Lançador - Simulador de Pluma & RL")
    root.geometry("520x560")
    root.minsize(480, 520)

    # Paleta de Cores Clara
    bg_main = "#f1f5f9"
    bg_card = "#ffffff"
    border_color = "#cbd5e1"
    text_dark = "#0f172a"
    text_muted = "#64748b"
    primary_color = "#0284c7"

    root.configure(bg=bg_main)
    root.option_add("*TCombobox*Listbox.background", "#ffffff")
    root.option_add("*TCombobox*Listbox.foreground", "#0f172a")
    root.option_add("*TCombobox*Listbox.selectBackground", primary_color)
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    style = ttk.Style(root)
    style.theme_use("clam")

    # Configurações do TTK para Tema Claro
    style.configure(".", background=bg_main, foreground=text_dark)
    style.configure("TFrame", background=bg_main)
    style.configure("Card.TLabelframe", background=bg_card, bordercolor=border_color)
    style.configure(
        "Card.TLabelframe.Label",
        background=bg_card,
        foreground=text_dark,
        font=("Segoe UI", 10, "bold"),
    )
    style.configure("TLabel", background=bg_card, foreground=text_dark, font=("Segoe UI", 9))
    style.configure("Muted.TLabel", background=bg_card, foreground=text_muted, font=("Segoe UI", 8))
    style.configure("TRadiobutton", background=bg_card, foreground=text_dark, font=("Segoe UI", 9))
    style.configure(
        "TEntry", fieldbackground="#ffffff", foreground=text_dark, insertcolor=text_dark
    )
    style.map(
        "TEntry",
        fieldbackground=[("!disabled", "#ffffff"), ("disabled", "#f1f5f9")],
        foreground=[("!disabled", text_dark), ("disabled", text_muted)],
    )
    style.configure(
        "TCombobox", fieldbackground="#ffffff", foreground=text_dark, background="#ffffff"
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", "#ffffff"), ("!disabled", "#ffffff")],
        foreground=[("readonly", text_dark), ("!disabled", text_dark)],
    )

    style.configure(
        "Action.TButton",
        background=primary_color,
        foreground="#ffffff",
        font=("Segoe UI", 10, "bold"),
        padding=8,
        borderwidth=0,
    )
    style.map("Action.TButton", background=[("active", "#0369a1")])

    # Variáveis do Tkinter
    var_acao = tk.StringVar(value="treinar")
    var_tipo_treino = tk.StringVar(value="novo")
    var_episodios = tk.StringVar(value="3000")
    var_nome_novo = tk.StringVar(value="")
    var_modelo = tk.StringVar()
    var_modelo_simular = tk.StringVar()

    if nomes_modelos:
        var_modelo.set(nomes_modelos[0])
        var_modelo_simular.set(nomes_modelos[0])

    def atualizar_interface(*args):
        """Ativa/Desativa/Mostra os frames dependendo da opção principal escolhida"""
        frame_treino.pack_forget()
        frame_simular.pack_forget()
        frame_dashboard.pack_forget()

        if var_acao.get() == "treinar":
            frame_treino.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)
            if var_tipo_treino.get() == "continuar":
                combo_modelo_treino.config(state="readonly")
                entry_nome_novo.config(state="disabled")
            else:
                combo_modelo_treino.config(state="disabled")
                entry_nome_novo.config(state="normal")
            btn_iniciar.config(text="Iniciar Treino")
        elif var_acao.get() == "simular":
            frame_simular.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)
            btn_iniciar.config(text="Iniciar Simulação")
        else:  # dashboard
            frame_dashboard.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)
            btn_iniciar.config(text="Abrir Dashboard")

    def iniciar():
        """Guarda as opções e fecha a janela para o script continuar"""
        escolhas["acao"] = var_acao.get()

        if escolhas["acao"] == "dashboard":
            import matplotlib.pyplot as plt

            janela_dashboard = tk.Toplevel(root)
            DashboardApp(janela_dashboard)

            def fechar_dashboard():
                plt.close("all")
                janela_dashboard.destroy()

            janela_dashboard.protocol("WM_DELETE_WINDOW", fechar_dashboard)
            return

        if escolhas["acao"] == "treinar":
            escolhas["episodios"] = int(var_episodios.get() or 3000)
            escolhas["tipo_treino"] = var_tipo_treino.get()
            if var_tipo_treino.get() == "continuar":
                escolhas["modelo_alvo"] = var_modelo.get()
                escolhas["nome_guardar"] = var_modelo.get()
            else:
                escolhas["modelo_alvo"] = None
                nome_digitado = var_nome_novo.get().strip()
                if nome_digitado:
                    import re

                    nome_limpo = re.sub(r'[\\/*?:"<>|]', "", nome_digitado).strip()
                    if not nome_limpo.endswith(".pkl"):
                        nome_limpo += ".pkl"
                    escolhas["nome_guardar"] = nome_limpo
                else:
                    escolhas["nome_guardar"] = None
        elif escolhas["acao"] == "simular":
            escolhas["modelo_alvo"] = var_modelo_simular.get()
            if not escolhas["modelo_alvo"]:
                messagebox.showerror("Erro", "Nenhum modelo selecionado para simular!")
                return

        root.destroy()

    # --- CABEÇALHO ---
    header_frame = tk.Frame(root, bg=bg_card, bd=1, relief="solid", padx=16, pady=10)
    header_frame.pack(fill=tk.X, padx=14, pady=(12, 6))

    lbl_logo = tk.Label(
        header_frame,
        text="Simulador de Pluma & Reinforcement Learning",
        font=("Segoe UI", 12, "bold"),
        bg=bg_card,
        fg=text_dark,
    )
    lbl_logo.pack(anchor="w")

    lbl_sublogo = tk.Label(
        header_frame,
        text="Navegação Robótica Subaquática",
        font=("Segoe UI", 9),
        bg=bg_card,
        fg=text_muted,
    )
    lbl_sublogo.pack(anchor="w", pady=(1, 0))

    # --- MODO DE OPERAÇÃO ---
    frame_modo = ttk.LabelFrame(
        root, text="  Modo de Operação  ", style="Card.TLabelframe", padding=10
    )
    frame_modo.pack(fill=tk.X, padx=14, pady=6)

    ttk.Radiobutton(
        frame_modo,
        text="Treinar Modelo Q-Learning",
        variable=var_acao,
        value="treinar",
        command=atualizar_interface,
    ).pack(anchor=tk.W, pady=3)

    ttk.Radiobutton(
        frame_modo,
        text="Executar Simulação Visual 2D",
        variable=var_acao,
        value="simular",
        command=atualizar_interface,
    ).pack(anchor=tk.W, pady=3)

    ttk.Radiobutton(
        frame_modo,
        text="Abrir Dashboard",
        variable=var_acao,
        value="dashboard",
        command=atualizar_interface,
    ).pack(anchor=tk.W, pady=3)

    # --- OPÇÕES DE TREINO (Dinâmico) ---
    frame_treino = ttk.LabelFrame(
        root, text="  Opções de Treino  ", style="Card.TLabelframe", padding=10
    )

    ttk.Radiobutton(
        frame_treino,
        text="Iniciar Novo Treino",
        variable=var_tipo_treino,
        value="novo",
        command=atualizar_interface,
    ).pack(anchor=tk.W, pady=(2, 2))

    frame_novo_nome = tk.Frame(frame_treino, bg=bg_card)
    frame_novo_nome.pack(fill=tk.X, padx=18, pady=(1, 6))
    ttk.Label(frame_novo_nome, text="Nome do modelo:").pack(anchor=tk.W)
    entry_nome_novo = ttk.Entry(frame_novo_nome, textvariable=var_nome_novo, width=42)
    entry_nome_novo.pack(anchor=tk.W, pady=(2, 0))

    ttk.Radiobutton(
        frame_treino,
        text="Continuar Treino de um Modelo",
        variable=var_tipo_treino,
        value="continuar",
        command=atualizar_interface,
    ).pack(anchor=tk.W, pady=(4, 2))

    frame_continuar = tk.Frame(frame_treino, bg=bg_card)
    frame_continuar.pack(fill=tk.X, padx=18, pady=(1, 6))
    combo_modelo_treino = ttk.Combobox(
        frame_continuar,
        textvariable=var_modelo,
        values=nomes_modelos,
        state="disabled",
        width=39,
    )
    combo_modelo_treino.pack(anchor=tk.W)

    frame_ep = tk.Frame(frame_treino, bg=bg_card)
    frame_ep.pack(fill=tk.X, padx=4, pady=(6, 2))
    ttk.Label(frame_ep, text="Número de episódios:").pack(anchor=tk.W)
    entry_ep = ttk.Entry(frame_ep, textvariable=var_episodios, width=16)
    entry_ep.pack(anchor=tk.W, pady=(2, 0))

    # --- OPÇÕES DE SIMULAÇÃO (Dinâmico) ---
    frame_simular = ttk.LabelFrame(
        root, text="  Opções de Simulação  ", style="Card.TLabelframe", padding=10
    )

    ttk.Label(frame_simular, text="Modelo a carregar:").pack(anchor=tk.W, pady=(2, 4))
    combo_modelo_sim = ttk.Combobox(
        frame_simular,
        textvariable=var_modelo_simular,
        values=nomes_modelos,
        state="readonly",
        width=44,
    )
    combo_modelo_sim.pack(anchor=tk.W, pady=(0, 4))

    ttk.Label(
        frame_simular,
        text="A simulação apresentará os últimos 5 episódios do modelo selecionado.",
        font=("Segoe UI", 8),
        foreground="#475569",
    ).pack(anchor=tk.W, pady=(0, 6))

    if not nomes_modelos:
        tk.Label(
            frame_simular,
            text="Nenhum modelo treinado encontrado na pasta 'modelos'.",
            font=("Segoe UI", 8),
            bg=bg_card,
            fg="#dc2626",
        ).pack(anchor=tk.W, pady=(0, 6))

    # --- OPÇÕES DE DASHBOARD (Dinâmico, sem lista de recursos) ---
    frame_dashboard = ttk.LabelFrame(
        root, text="  Dashboard  ", style="Card.TLabelframe", padding=14
    )

    ttk.Label(
        frame_dashboard,
        text="Abre a janela do Dashboard para analisar o desempenho,\nrotas de cada episódio e configurações do modelo.",
        font=("Segoe UI", 9),
        justify=tk.LEFT,
    ).pack(anchor=tk.W, pady=8)

    # --- BOTÃO DE AÇÃO ---
    btn_iniciar = ttk.Button(root, text="Iniciar", command=iniciar, style="Action.TButton")
    btn_iniciar.pack(side=tk.BOTTOM, pady=16, fill=tk.X, padx=14)

    # Inicializa a interface corretamente
    atualizar_interface()

    root.mainloop()

    # --- LÓGICA DE EXECUÇÃO APÓS A INTERFACE FECHAR ---

    # Se a janela for fechada sem clicar em Iniciar, o script termina

    if escolhas["acao"] is None:
        print("Execução cancelada.")

        sys.exit()

    if escolhas["acao"] == "treinar":
        nome_treino = (escolhas.get("nome_guardar") or escolhas.get("modelo_alvo") or "").lower()
        if "linear" in nome_treino:
            config["meander_model"] = "linear"
            print(
                "\n[+] Treino Linear detetado. Ambiente configurado com meander_model = 'linear'."
            )
        elif "sinusoid" in nome_treino:
            config["meander_model"] = "sum_of_sinusoids"
            print(
                "\n[+] Treino Sinusoidal detetado. Ambiente configurado com meander_model = 'sum_of_sinusoids'."
            )

        env = PlumeRLWrapper(config)
        agent = QLearningAgent(env)

        if escolhas["tipo_treino"] == "continuar" and escolhas["modelo_alvo"]:
            caminho_modelo = os.path.join(pasta_modelos, escolhas["modelo_alvo"])

            if agent.load(caminho_modelo):
                print(
                    f"\n[!] A continuar o treino a partir do episódio {len(agent.episode_rewards)}"
                )

                print(f"[!] Epsilon atual (taxa de exploração): {agent.epsilon:.4f}")

            else:
                print("[-] Erro ao carregar. A iniciar um novo treino do zero...")

        episodios = escolhas["episodios"]

        agent.train(env, num_episodes=episodios, print_every=50)

        caminho_salvo = agent.save(filename=escolhas.get("nome_guardar"))

        if config.get("visualize", True):
            nome_ficheiro = os.path.basename(caminho_salvo)

            config["agent_class"] = "QLearningAgent"
            config["model_to_load"] = caminho_salvo
            config["trained_q_agent"] = agent
            config["num_experiments"] = 5

            print("\n[!] Treino concluído com sucesso!")
            print(
                f"\n[!] A iniciar o simulador para mostrar os últimos 5 episódios do treino atual ({nome_ficheiro})...\n"
            )

            sim = Simulator(config)
            sim.run()

    elif escolhas["acao"] == "simular":
        ficheiro_escolhido = os.path.join(pasta_modelos, escolhas["modelo_alvo"])

        config["agent_class"] = "QLearningAgent"
        config["model_to_load"] = ficheiro_escolhido
        config["trained_q_agent"] = None
        config["num_experiments"] = 5

        nome_alvo_low = escolhas["modelo_alvo"].lower()
        if "sinusoid" in nome_alvo_low:
            config["meander_model"] = "sum_of_sinusoids"
            print(
                f"\n[+] Modelo Sinusoidal detetado ({escolhas['modelo_alvo']}). Configurando simulador com meander_model = 'sum_of_sinusoids'."
            )
        else:
            config["meander_model"] = "linear"
            print(
                f"\n[+] Modelo Linear detetado ({escolhas['modelo_alvo']}). Configurando simulador com meander_model = 'linear'."
            )

        print(f"\nIniciando Simulador com: QLearningAgent usando {ficheiro_escolhido}")

        sim = Simulator(config)
        sim.run()


if __name__ == "__main__":
    main()
