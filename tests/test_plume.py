import numpy as np

from flysim.plume import kernels
from flysim.plume.filament import (
    FilamentPlume,
    LinearMeanderModel,
    OUMeanderModel,
    SinusoidMeanderModel,
)


def test_meander_models(base_config):
    dt = 0.1
    t = 1.0

    linear = LinearMeanderModel(base_config, {})
    assert linear.update(dt, t) == 0.0

    ou = OUMeanderModel(base_config, base_config["ou_meander_config"])
    vy_ou = ou.update(dt, t)
    assert isinstance(vy_ou, (float, np.floating))

    sin = SinusoidMeanderModel(base_config, base_config["sinusoid_meander_config"])
    vy_sin = sin.update(dt, t)
    assert isinstance(vy_sin, (float, np.floating))


def test_filament_plume_update(base_config):
    plume = FilamentPlume(base_config)
    for step in range(15):
        plume.update(base_config["dt"], step * base_config["dt"])
    active_puffs = (plume.puffs_array[:, 3] > 0).sum()
    assert active_puffs > 0
    assert plume.puffs_array.shape[1] == 5


def test_numba_concentration_kernels():
    puffs = np.zeros((10, 5), dtype=np.float32)
    puffs[0] = [20.0, 25.0, 1.0, 100.0, 1.0]
    pos = np.array([20.0, 25.0], dtype=np.float64)
    conc_tophat = kernels.calculate_concentration_numba(pos, 1.0, puffs)
    conc_gauss = kernels.calculate_concentration_gaussian_numba(pos, puffs)
    assert conc_tophat > 0.0
    assert conc_gauss > 0.0
