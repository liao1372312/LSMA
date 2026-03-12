"""
QoS-LSMA Configuration
=======================
Configuration dataclass for the QoS-LSMA framework.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class QoSLSMAConfig:
    """Global configuration for the QoS-LSMA framework.

    All hyper-parameters and API credentials are centralised here so that
    experimenting with different settings requires only one change.
    """

    # ------------------------------------------------------------------
    # LLM settings
    # ------------------------------------------------------------------
    llm_model: str = "deepseek-chat"       # e.g. "deepseek-chat", "gpt-4o"
    llm_api_key: str = ""                  # Your LLM API key
    llm_base_url: Optional[str] = "https://api.deepseek.com"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048

    # ------------------------------------------------------------------
    # Embedding settings  (paper: OpenAI text-embedding-3-small)
    # ------------------------------------------------------------------
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536              # dim of text-embedding-3-small
    embedding_api_key: str = ""            # OpenAI key for embeddings
    embedding_base_url: Optional[str] = None

    # ------------------------------------------------------------------
    # Memory retrieval settings  (Section III-E)
    # ------------------------------------------------------------------
    top_k_candidates: int = 10             # K seeds from embedding index
    hop_size: int = 2                      # neighbourhood expansion hops (h)
    max_edges_per_seed: int = 20           # cap edges per seed in subgraph
    priority_relations: List[str] = field(
        default_factory=lambda: [
            "uses", "calls", "failed_because", "fixed_by", "mitigates"
        ]
    )

    # ------------------------------------------------------------------
    # DQN / RL settings  (Section III-G)
    # ------------------------------------------------------------------
    dqn_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dqn_lr: float = 1e-3
    dqn_gamma: float = 0.99                # discount factor γ
    dqn_epsilon_start: float = 1.0         # ε-greedy initial exploration
    dqn_epsilon_end: float = 0.05
    dqn_epsilon_decay: float = 0.995
    dqn_batch_size: int = 32
    dqn_target_update_freq: int = 10       # update target network every N steps
    replay_buffer_size: int = 10_000
    dqn_update_every: int = 1              # DQN update every N interactions

    # ------------------------------------------------------------------
    # Reward settings  (Equation 2)
    # ------------------------------------------------------------------
    reward_baseline_alpha: float = 0.1    # EMA coefficient for R̄_user
    reval_max_score: float = 5.0           # max evaluator score

    # ------------------------------------------------------------------
    # Agent / composition settings  (Section III-C)
    # ------------------------------------------------------------------
    max_retries: int = 3                   # Supervisor retry threshold
    supervisor_enabled: bool = True

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------
    max_stm_size: int = 100               # STM buffer cap
    max_ltm_nodes: int = 10_000           # LTM node cap
