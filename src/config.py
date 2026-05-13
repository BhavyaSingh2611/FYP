from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SETTINGS_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


class HardwareConfig(BaseModel):
    device: str = "auto"
    num_workers: int = 4


class EngineConfig(BaseModel):
    path: str = "/opt/homebrew/bin/stockfish"
    default_depth: int = 15
    default_multipv: int = 5


class PathsConfig(BaseModel):
    database: str = "data"
    checkpoints: str = "training_results/"


class CNNConfig(BaseModel):
    num_blocks: int = 10
    channels: int = 256


class TransformerConfig(BaseModel):
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1


class GNNConfig(BaseModel):
    hidden_dim: int = 256
    num_layers: int = 6
    edge_type: str = "hybrid"
    heads: int = 4


class ModelConfig(BaseModel):
    head: str = "dual"
    cnn: CNNConfig = CNNConfig()
    transformer: TransformerConfig = TransformerConfig()
    gnn: GNNConfig = GNNConfig()


class LRSchedulerConfig(BaseModel):
    type: str = "cosine"
    step_size: int = 10
    gamma: float = 0.1


class TrainingConfig(BaseModel):
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 50
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    gradient_accumulation_steps: int = 1
    grad_clip_max_norm: float = 1.0
    save_every: int = 5
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()


class EnginesConfig(BaseModel):
    stockfish: EngineConfig = EngineConfig()


class Settings(BaseSettings):
    hardware: HardwareConfig = HardwareConfig()
    engines: EnginesConfig = EnginesConfig()
    paths: PathsConfig = PathsConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

    model_config = {"env_prefix": "CHESS_", "env_nested_delimiter": "__"}


@lru_cache
def get_settings() -> Settings:
    """Load settings from config/settings.yaml, with env var overrides."""
    if _SETTINGS_PATH.exists():
        with open(_SETTINGS_PATH) as f:
            file_data = yaml.safe_load(f) or {}
        return Settings(**file_data)
    return Settings()


settings = get_settings()
