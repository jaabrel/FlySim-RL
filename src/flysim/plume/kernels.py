from __future__ import annotations

import numpy as np
from numba import jit


@jit(nopython=True)
def update_puffs_numba(
    puffs_array,
    dt,
    wind_velocity,
    meandering_vy,
    turbulence_intensity,
    puff_decay_rate,
    puff_expansion_rate,
    min_mass,
):
    """
    Atualiza o estado físico de todos os pacotes de odor (puffs) para o modelo padrão/OU.
    Esta função é compilada com Numba (nopython) para máxima eficiência temporal.

    Operações por pacote:
    - Advecção pelo vento/corrente média e velocidade transversal de meandro.
    - Perturbação turbulenta estocástica isotrópica (Browniana).
    - Expansão volumétrica do raio proporcional à raiz quadrada da idade do puff.
    - Decaimento exponencial da massa de odor por evaporação/dissipação.
    """
    num_puffs = puffs_array.shape[0]
    sqrt_dt = np.sqrt(dt)

    for i in range(num_puffs):
        if puffs_array[i, 3] <= min_mass:
            continue

        # Advecção pelo escoamento médio e meandro transversal
        puffs_array[i, 0] += wind_velocity[0] * dt
        puffs_array[i, 1] += wind_velocity[1] * dt
        puffs_array[i, 1] += meandering_vy * dt

        # Perturbação turbulenta isotrópica para dispersão natural
        turbulence_x = np.random.randn() * turbulence_intensity * sqrt_dt
        turbulence_y = np.random.randn() * turbulence_intensity * sqrt_dt
        puffs_array[i, 0] += turbulence_x
        puffs_array[i, 1] += turbulence_y

        # Expansão radial e decaimento exponencial de massa
        puffs_array[i, 2] = puff_expansion_rate * np.sqrt(puffs_array[i, 4])
        puffs_array[i, 3] *= 1.0 - puff_decay_rate * dt
        puffs_array[i, 4] += dt

        # Desativar pacotes que saíram dos limites físicos do domínio para libertar slots
        if (
            puffs_array[i, 0] > 115.0
            or puffs_array[i, 0] < -15.0
            or puffs_array[i, 1] > 65.0
            or puffs_array[i, 1] < -15.0
        ):
            puffs_array[i, 3] = 0.0

    return puffs_array


@jit(nopython=True)
def update_puffs_sinusoid_numba(
    puffs_array,
    dt,
    current_time,
    src_x,
    src_y,
    wind_velocity,
    amplitude,
    frequency,
    turbulence_intensity,
    puff_decay_rate,
    puff_expansion_rate,
    min_mass,
):
    """
    Atualiza os pacotes de odor para o modelo de pluma sinusoidal meandrada.
    Garante que a pluma emerge de forma contínua e centrada na fonte (src_x, src_y),
    abrindo suavemente a amplitude ondulatória à medida que os pacotes avançam a jusante.
    """
    num_puffs = puffs_array.shape[0]
    sqrt_dt = np.sqrt(dt)
    vx = wind_velocity[0] if abs(wind_velocity[0]) > 1e-4 else 0.4
    ramp_l = 5.0
    two_pi_f = 2.0 * np.pi * frequency

    for i in range(num_puffs):
        if puffs_array[i, 3] <= min_mass:
            continue

        # Advecção longitudinal com turbulência
        puffs_array[i, 0] += wind_velocity[0] * dt
        puffs_array[i, 0] += np.random.randn() * turbulence_intensity * sqrt_dt

        # Onda sinusoidal com rampa de transição suave a partir da fonte
        dx = max(0.0, puffs_array[i, 0] - src_x)
        ramp = np.tanh(dx / ramp_l)
        target_center_y = src_y + amplitude * ramp * np.sin(two_pi_f * (current_time - dx / vx))

        # Posição transversal com componente turbulenta
        puffs_array[i, 1] = target_center_y + np.random.randn() * turbulence_intensity * sqrt_dt
        puffs_array[i, 2] = puff_expansion_rate * np.sqrt(puffs_array[i, 4])
        puffs_array[i, 3] *= 1.0 - puff_decay_rate * dt
        puffs_array[i, 4] += dt

        # Desativar pacotes que saíram dos limites físicos do domínio para libertar slots
        if (
            puffs_array[i, 0] > 115.0
            or puffs_array[i, 0] < -15.0
            or puffs_array[i, 1] > 65.0
            or puffs_array[i, 1] < -15.0
        ):
            puffs_array[i, 3] = 0.0

    return puffs_array


@jit(nopython=True)
def calculate_concentration_numba(agent_pos, agent_radius, puffs_array):
    """
    Calcula a concentração sensorial no ponto do veículo segundo o modelo Top-Hat
    (concentração uniforme no interior do raio de cada pacote de odor).
    """
    total_concentration = 0.0
    agent_x, agent_y = agent_pos[0], agent_pos[1]

    for i in range(puffs_array.shape[0]):
        puff_mass = puffs_array[i, 3]
        if puff_mass < 1e-6:
            continue

        puff_x, puff_y, puff_r = (
            puffs_array[i, 0],
            puffs_array[i, 1],
            puffs_array[i, 2],
        )
        dist_sq = (agent_x - puff_x) ** 2 + (agent_y - puff_y) ** 2
        radii_sum_sq = (agent_radius + puff_r) ** 2
        if dist_sq < radii_sum_sq:
            puff_concentration = puff_mass / (np.pi * puff_r**2) if puff_r > 1e-6 else 0.0
            total_concentration += puff_concentration
    return total_concentration


@jit(nopython=True)
def calculate_concentration_gaussian_numba(agent_pos, puffs_array):
    """
    Calcula a concentração local segundo um perfil Gaussiano contínuo,
    em que cada pacote de odor atua como uma distribuição normal com desvio padrão sigma = r / 2.
    """
    total_concentration = 0.0
    agent_x, agent_y = agent_pos[0], agent_pos[1]
    for i in range(puffs_array.shape[0]):
        puff_x, puff_y, puff_r, puff_mass = (
            puffs_array[i, 0],
            puffs_array[i, 1],
            puffs_array[i, 2],
            puffs_array[i, 3],
        )
        if puff_mass < 1e-6:
            continue
        sigma = puff_r / 2.0
        if sigma < 1e-6:
            continue
        dist_sq = (agent_x - puff_x) ** 2 + (agent_y - puff_y) ** 2
        if dist_sq > (puff_r * 3) ** 2:
            continue
        sigma_sq = sigma**2
        concentration_k = (puff_mass / (2 * np.pi * sigma_sq)) * np.exp(-dist_sq / (2 * sigma_sq))
        total_concentration += concentration_k
    return total_concentration


def plume_exists_at_x(x, plume, config):
    if plume is None:
        return False
    world_h = config["world_height"]
    agent_config = config.get("agent_config", {})
    conc_thresh = agent_config.get("agent_concentration_threshold", 1.0)
    agent_radius = agent_config.get("agent_radius", 1.0)
    model = config.get("concentration_model", "top-hat")

    y_coords = np.linspace(0.0, world_h, 25)
    for y in y_coords:
        pos = np.array([x, y])
        if model == "gaussian":
            conc = calculate_concentration_gaussian_numba(pos, plume.puffs_array)
        else:
            conc = calculate_concentration_numba(pos, agent_radius, plume.puffs_array)
        if conc > conc_thresh:
            return True
    return False


def get_plume_center_y(x, t, plume, config):
    """
    Retorna a linha central transversal (coordenada Y) da pluma na posição longitudinal x
    e instante de simulação t.
    - Para plumas sinusoidais: calcula analiticamente o eixo ondulatório instantâneo.
    - Para plumas lineares ou OU: retorna a coordenada Y da fonte emissora.
    """
    src_x, src_y = config["source_position"]
    is_sinusoid = config.get("meander_model", "ou").lower() in [
        "sum_of_sinusoids",
        "sinusoid",
        "sinusoidal",
    ]
    if is_sinusoid:
        sin_cfg = config.get("sinusoid_meander_config", {})
        amplitude = float(sin_cfg.get("amplitude", sin_cfg.get("v_var", 8.0)))
        frequency = float(sin_cfg.get("frequency", 0.018))
        ramp_l = 5.0
        two_pi_f = 2.0 * np.pi * frequency
        vx = config["mean_wind_velocity"][0] if abs(config["mean_wind_velocity"][0]) > 1e-4 else 0.4
        dx = max(0.0, x - src_x)
        ramp = np.tanh(dx / ramp_l)
        return float(src_y + amplitude * ramp * np.sin(two_pi_f * (t - dx / vx)))
    return float(src_y)
