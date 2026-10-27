# FlySim-RL

**FlySim-RL** é um simulador computacional desenvolvido em Python para o treino, avaliação e validação de algoritmos de Aprendizagem por Reforço (*Reinforcement Learning* - RL) aplicados à exploração e mapeamento de plumas de poluição em ambientes marinhos.

Este projeto foi desenvolvido no âmbito académico na **Universidade Politécnica de Tomar (UPT)** por **João Alexandre de Abreu** e **Rodrigo Mendes Pinheiro**.

## 🏗️ Estrutura do Repositório
```text
FlySim-RL/
├── src/
│   └── flysim/
│       ├── __init__.py          # Versão e metadados do pacote
│       ├── __main__.py          # Lançador interativo CLI / Tkinter
│       ├── config/              # Validação e esquemas de configuração
│       │   ├── schema.py
│       │   └── defaults.json
│       ├── plume/               # Física de dispersão e modelos de meandro
│       │   ├── kernels.py       # Funções aceleradas com @jit Numba
│       │   ├── models.py        # Modelos matemáticos de esteira
│       │   └── filament.py      # Gestor lagrangeano de filamentos
│       ├── agents/              # Estrutura e comportamentos dos AUVs
│       │   ├── base.py          # Classe base Agent
│       │   ├── bio_inspired.py  # Agente bioinspirado
│       │   ├── simple_upwind.py # Agente de teste Upwind
│       │   └── rl/
│       │       └── qlearning.py # Agente Q-Learning e SimAgent de execução
│       ├── environment/         # Ambiente de treino Gymnasium
│       │   └── plume_env.py
│       ├── simulation/          # Motor de simulação e HUD
│       │   ├── logger.py        # Gravação de dados HDF5
│       │   └── runner.py        # Execução de ensaios e renderização OpenCV
│       └── dashboard/           # Aplicação gráfica de análise
│           └── app.py
├── modelos/                     # Modelos treinados (.pkl)
├── logs/                        # Registos de treino (.log)
├── experiments/                 # Dados de simulação gravados (.h5)
├── boat_sprite.png              
├── config.json                  # Ficheiro de parâmetros da simulação
├── pyproject.toml               
├── uv.lock                      
└── README.md

```
## Guia de Setup e Execução

### 1\. Pré-requisitos

* **Python 3.12**
* **UV**

### 2\. Dependências do Projeto

Na **raiz do projeto**, executar:

```bash
uv sync
```
### 3\. Execução do Projeto

Todos os comandos devem ser executados na **raiz do projeto**:

```bash
uv run flysim
```


## Autores

- **João Alexandre de Abreu**
- **Rodrigo Mendes Pinheiro**

*Licenciatura em Engenharia Informática - Universidade Politécnica de Tomar (UPT)*

