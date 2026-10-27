import os
import tempfile

import numpy as np

from flysim.simulation.logger import DataLogger
from flysim.simulation.runner import Simulator


def test_data_logger():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"save_path": tmpdir, "experiment_name": "test_run"}
        logger = DataLogger(cfg, experiment_num=0)
        logger.log_step(0.1, np.array([10.0, 20.0]), 1.5, "SEARCHING", 99.0)
        logger.save()
        assert os.path.exists(logger.filename)
        assert logger.filename.endswith(".h5")


def test_simulator_headless(base_config):
    cfg = dict(base_config)
    cfg["visualize"] = False
    cfg["num_experiments"] = 1
    cfg["max_time"] = 0.5
    sim = Simulator(cfg)
    assert sim.agent is not None
    sim.run()
