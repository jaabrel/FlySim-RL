import tempfile

from flysim.agents.base import Agent
from flysim.agents.bio_inspired import BioInspiredAgent
from flysim.agents.rl.qlearning import QLearningAgent
from flysim.agents.simple_upwind import SimpleUpwindAgent
from flysim.environment.plume_env import PlumeRLWrapper


def test_base_and_heuristic_agents(base_config):
    base_agent = Agent(base_config)
    assert base_agent.battery_capacity == 100.0
    bio_agent = BioInspiredAgent(base_config)
    bio_agent.reset()
    assert bio_agent.state == "SEARCHING"
    upwind_agent = SimpleUpwindAgent(base_config)
    upwind_agent.reset()
    assert hasattr(upwind_agent, "step")


def test_qlearning_agent_flow(base_config):
    env = PlumeRLWrapper(base_config)
    agent = QLearningAgent(env)
    obs, _ = env.reset()
    action = agent.choose_action(obs, greedy=True)
    assert action in [0, 1, 2, 3]
    next_obs, reward, term, trunc, _ = env.step(action)
    agent.update(obs, action, reward, next_obs, term, trunc)
    key = agent._state_key(obs)
    assert key in agent.q_table
    # Validação de persistência (save / load)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        path = agent.save(tmp.name)
        assert agent.load(path) is True
