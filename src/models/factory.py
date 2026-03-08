"""
Model factory for hot-swapping neural network backbones.
"""

from ..config import ModelConfig, settings
from .base import ChessModel
from .cnn import ConvNet, ResNet
from .gnn import GAT, GCN
from .heads import create_head
from .transformer import PieceTransformer, SquareTransformer


def create_model(config: ModelConfig | None = None) -> ChessModel:
    """
    Create a complete chess model from configuration.

    Args:
        config: Optional ModelConfig. Defaults to settings.model.

    Returns:
        A ChessModel with backbone and head attached.
    """
    if config is None:
        config = settings.model
    backbone = config.backbone.lower()

    # Create backbone
    if backbone == "convnet":
        model = ConvNet(
            channels=config.cnn.channels,
            num_layers=6,  # Fixed for ConvNet
        )

    elif backbone == "resnet":
        model = ResNet(
            channels=config.cnn.channels,
            num_blocks=config.cnn.num_blocks,
        )

    elif backbone == "square_transformer":
        model = SquareTransformer(
            embed_dim=config.transformer.embed_dim,
            num_heads=config.transformer.num_heads,
            num_layers=config.transformer.num_layers,
            dropout=config.transformer.dropout,
        )

    elif backbone == "piece_transformer":
        model = PieceTransformer(
            embed_dim=config.transformer.embed_dim,
            num_heads=config.transformer.num_heads,
            num_layers=config.transformer.num_layers,
            dropout=config.transformer.dropout,
        )

    elif backbone == "gcn":
        model = GCN(
            hidden_dim=config.gnn.hidden_dim,
            num_layers=config.gnn.num_layers,
            edge_type=config.gnn.edge_type,
        )

    elif backbone == "gat":
        model = GAT(
            hidden_dim=config.gnn.hidden_dim,
            num_layers=config.gnn.num_layers,
            edge_type=config.gnn.edge_type,
            heads=config.gnn.heads,
        )

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    # Create and attach head
    head = create_head(
        head_type=config.head,
        input_dim=model.get_backbone_output_dim(),
        hidden_dim=256,
    )
    model.set_head(head)

    return model


def get_encoder_for_model(backbone_type: str):
    """
    Get the appropriate encoder class for a model backbone.

    Args:
        backbone_type: Model backbone name.

    Returns:
        Encoder class (not instantiated).
    """
    from ..chess_env.encoders import CNNEncoder, GNNEncoder, TransformerEncoder

    backbone_type = backbone_type.lower()

    if backbone_type in ["convnet", "resnet"]:
        return CNNEncoder
    elif backbone_type in ["square_transformer", "piece_transformer"]:
        # Return the encoder with correct tokenizer type
        tokenizer_type = "square" if backbone_type == "square_transformer" else "piece"
        return lambda: TransformerEncoder(tokenizer_type=tokenizer_type)
    elif backbone_type in ["gcn", "gat"]:
        return GNNEncoder
    else:
        raise ValueError(f"Unknown backbone: {backbone_type}")


def list_available_models() -> dict:
    """
    List all available model configurations.

    Returns:
        Dictionary with model information.
    """
    return {
        "backbones": {
            "cnn": ["convnet", "resnet"],
            "transformer": ["square_transformer", "piece_transformer"],
            "gnn": ["gcn", "gat"],
        },
        "heads": ["policy", "value", "dual"],
        "gnn_edge_types": ["static", "dynamic", "hybrid"],
        "total_configurations": 30,  # 10 backbones × 3 heads
    }
