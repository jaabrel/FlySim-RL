from flysim.environment.plume_env import PlumeRLWrapper


def test_environment_reset_and_step(base_config):
    env = PlumeRLWrapper(base_config)

    obs, info = env.reset()
    assert obs.shape == (8,)
    assert isinstance(info, dict)

    # Executa uma ação (0: Cima, 1: Baixo, 2: Esquerda, 3: Direita)
    next_obs, reward, terminated, truncated, step_info = env.step(0)

    assert next_obs.shape == (8,)
    assert isinstance(reward, (float, int))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "battery" in step_info
