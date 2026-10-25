from __future__ import annotations

import json
import os
from datetime import datetime

import h5py
import numpy as np


class DataLogger:
    """Gere o registo e exportação de dados temporais da simulação para ficheiros HDF5 (.h5)."""

    def __init__(self, config, experiment_num):
        self.save_path = config["save_path"]
        self.experiment_name = config["experiment_name"]
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(
            self.save_path, f"{self.experiment_name}_exp{experiment_num}_{timestamp}.h5"
        )
        self.data_buffer = {
            "time": [],
            "agent_position": [],
            "concentration": [],
            "agent_state": [],
            "battery": [],
            "custom_osl_data": [],
        }
        self.config = config

    def log_step(
        self, sim_time, agent_pos, concentration, agent_state, battery, custom_osl_data=None
    ):
        """Regista um passo temporal na memória interna para posterior gravação."""
        self.data_buffer["time"].append(sim_time)
        self.data_buffer["agent_position"].append(agent_pos)
        self.data_buffer["concentration"].append(concentration)
        self.data_buffer["agent_state"].append(agent_state)
        self.data_buffer["battery"].append(battery)
        if custom_osl_data:
            self.data_buffer["custom_osl_data"].append(json.dumps(custom_osl_data))

    def save(self):
        """Escreve todos os dados acumulados no ficheiro HDF5 estruturado."""
        with h5py.File(self.filename, "w") as f:
            for key, data in self.data_buffer.items():
                if not data:
                    continue
                if key == "custom_osl_data":
                    dt = h5py.special_dtype(vlen=str)
                    f.create_dataset("osl/custom_data", data=data, dtype=dt)
                elif key == "agent_state":
                    dt = h5py.special_dtype(vlen=str)
                    f.create_dataset("agent/states", data=data, dtype=dt)
                elif key == "agent_position":
                    f.create_dataset("agent/positions", data=np.array(data))
                elif key == "concentration":
                    f.create_dataset("agent/concentrations", data=np.array(data))
                elif key == "battery":
                    f.create_dataset("agent/battery", data=np.array(data))
                else:
                    f.create_dataset(key, data=np.array(data))

            for key, value in self.config.items():
                try:
                    f.attrs[key] = value
                except TypeError:
                    f.attrs[key] = str(value)
