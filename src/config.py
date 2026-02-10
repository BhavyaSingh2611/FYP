"""
Configuration management module.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class HardwareConfig:
    device: str = "auto"
    num_workers: int = 4


@dataclass
class EngineConfig:
    path: str = "/opt/homebrew/bin/stockfish"
    default_depth: int = 15
    default_multipv: int = 5


@dataclass
class PathsConfig:
    database: str = "data"
    checkpoints: str = "training_results/"


@dataclass
class CNNConfig:
    num_blocks: int = 10
    channels: int = 256
    input_channels: int = 18


@dataclass
class TransformerConfig:
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    vocab_size: int = 13  # 13 for Square, 12 for Piece
    max_len: int = 64     # 64 for Square, 32 for Piece


@dataclass
class GNNConfig:
    hidden_dim: int = 256
    num_layers: int = 6
    edge_type: str = "hybrid"  # static, dynamic, hybrid
    heads: int = 4  # For GAT only
    input_dim: int = 18  # Node feature dim


@dataclass
class ModelConfig:
    backbone: str = "resnet"
    head: str = "dual"
    cnn: CNNConfig = field(default_factory=CNNConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)


@dataclass
class LRSchedulerConfig:
    type: str = "cosine"
    step_size: int = 10
    gamma: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 50
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)


@dataclass
class Config:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    engines: dict = field(default_factory=lambda: {"stockfish": EngineConfig()})
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _dict_to_dataclass(cls, data: dict) -> Any:
    """Recursively convert a dictionary to a dataclass."""
    if data is None:
        return cls()
    
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    
    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            # Check if the field type is a dataclass
            if hasattr(field_type, '__dataclass_fields__'):
                kwargs[key] = _dict_to_dataclass(field_type, value)
            else:
                kwargs[key] = value
    
    return cls(**kwargs)


def load_config(config_path: str | Path) -> Config:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        Config: Parsed configuration object.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        return Config()
    
    # Parse each section
    config = Config()
    
    if "hardware" in raw_config:
        config.hardware = _dict_to_dataclass(HardwareConfig, raw_config["hardware"])
    
    if "engines" in raw_config:
        config.engines = {}
        for engine_name, engine_data in raw_config["engines"].items():
            config.engines[engine_name] = _dict_to_dataclass(EngineConfig, engine_data)
    
    if "paths" in raw_config:
        config.paths = _dict_to_dataclass(PathsConfig, raw_config["paths"])
    
    if "model" in raw_config:
        model_data = raw_config["model"]
        config.model = ModelConfig(
            backbone=model_data.get("backbone", "resnet"),
            head=model_data.get("head", "dual"),
            cnn=_dict_to_dataclass(CNNConfig, model_data.get("cnn")),
            transformer=_dict_to_dataclass(TransformerConfig, model_data.get("transformer")),
            gnn=_dict_to_dataclass(GNNConfig, model_data.get("gnn")),
        )
    
    if "training" in raw_config:
        training_data = raw_config["training"]
        lr_scheduler = _dict_to_dataclass(
            LRSchedulerConfig, 
            training_data.get("lr_scheduler")
        )
        config.training = TrainingConfig(
            batch_size=training_data.get("batch_size", 256),
            learning_rate=training_data.get("learning_rate", 0.001),
            weight_decay=training_data.get("weight_decay", 0.0001),
            epochs=training_data.get("epochs", 50),
            policy_loss_weight=training_data.get("policy_loss_weight", 1.0),
            value_loss_weight=training_data.get("value_loss_weight", 1.0),
            lr_scheduler=lr_scheduler,
        )
    
    return config


def get_config_value(config: Config, key_path: str) -> Any:
    """
    Get a configuration value using a dot-separated path.
    
    Args:
        config: Configuration object.
        key_path: Dot-separated path (e.g., 'model.backbone').
    
    Returns:
        The configuration value.
    """
    obj = config
    for key in key_path.split('.'):
        if hasattr(obj, key):
            obj = getattr(obj, key)
        elif isinstance(obj, dict):
            obj = obj[key]
        else:
            raise KeyError(f"Configuration key not found: {key_path}")
    return obj
