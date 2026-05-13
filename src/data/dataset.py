"""
PyTorch IterableDataset for loading chess training data from a Parquet file.

Streams row-groups one at a time so the full file is never resident in memory.
"""

from pathlib import Path
from typing import Any

import chess
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset

from ..chess_env.move_index import UCI_MOVE_TO_INDEX


class ChessDataset(IterableDataset):
    """
    Streams training examples from a single Parquet file.

    Parquet schema:
        f – FEN string
        b – best move in UCI notation
        v – pre-computed value (-1.0 to +1.0, from side-to-move perspective)

    Only one row-group is in memory at a time, so RAM usage stays bounded
    regardless of file size.  Data is assumed to be pre-shuffled.
    """

    def __init__(
        self,
        path: Path,
        encoder,
        num_samples: int | None = None,
        include_value: bool = True,
    ):
        super().__init__()

        self.path = Path(path)
        self.encoder = encoder
        self.num_samples = num_samples
        self.include_value = include_value

        self._total_rows = pq.read_metadata(self.path).num_rows
        self._effective_len = min(self.num_samples, self._total_rows) if self.num_samples else self._total_rows

    def __len__(self) -> int:
        return int(self._effective_len)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        pf = pq.ParquetFile(self.path)
        num_row_groups = pf.metadata.num_row_groups
        row_groups = list(range(num_row_groups))

        if worker_info is not None:
            row_groups = row_groups[worker_info.id :: worker_info.num_workers]

        per_worker_limit = None
        if self.num_samples:
            n_workers = worker_info.num_workers if worker_info else 1
            per_worker_limit = self.num_samples // n_workers

        emitted = 0
        use_batch = hasattr(self.encoder, "encode_batch")

        for rg_idx in row_groups:
            table = pf.read_row_group(rg_idx)
            fens = table.column("f").to_pylist()
            best_moves = table.column("b").to_pylist()
            values = table.column("v").to_pylist()

            # collect valid rows up to the worker limit
            rows: list[tuple] = []
            for i in range(table.num_rows):
                if per_worker_limit and emitted + len(rows) >= per_worker_limit:
                    break
                if fens[i] is None:
                    continue
                try:
                    board = chess.Board(fens[i])
                except ValueError:
                    continue
                rows.append((board, best_moves[i], values[i]))

            if not rows:
                continue

            # encode all boards at once or one-by-one
            boards = [r[0] for r in rows]
            encoded = self.encoder.encode_batch(boards) if use_batch else [self.encoder.encode(b) for b in boards]

            for enc, (_, move, val) in zip(encoded, rows, strict=False):
                if per_worker_limit and emitted >= per_worker_limit:
                    return

                result = {
                    "input": enc,
                    "policy_target": _build_policy(move),
                }

                if self.include_value:
                    result["value_target"] = torch.tensor([val], dtype=torch.float32)

                emitted += 1
                yield result


def _build_policy(best_move: str | None) -> torch.Tensor:
    if best_move is not None and str(best_move) != "nan":
        idx = UCI_MOVE_TO_INDEX.get(str(best_move), -1)
        if idx >= 0:
            return torch.tensor(idx, dtype=torch.long)

    return torch.tensor(0, dtype=torch.long)


def _is_graph_input(first_input: dict) -> bool:
    return isinstance(first_input, dict) and "edge_index" in first_input


def _collate_graph_inputs(batch: list[dict]) -> dict:
    """Batch graph inputs by concatenating edges with node-index offsets."""
    num_nodes = 64

    node_features = torch.stack([b["input"]["x"] for b in batch])  # (B, 64, F)
    batch_size = node_features.size(0)

    edge_indices, edge_attrs = [], []
    for i, b in enumerate(batch):
        edge_indices.append(b["input"]["edge_index"] + i * num_nodes)

        if "edge_attr" in b["input"]:
            edge_attrs.append(b["input"]["edge_attr"])

    graph_batch = torch.arange(batch_size).unsqueeze(1).expand(-1, num_nodes).reshape(-1)

    result = {
        "x": node_features.view(-1, node_features.size(-1)),  # (B*64, F)
        "edge_index": torch.cat(edge_indices, dim=1),  # (2, total_E)
        "batch": graph_batch,
        "side_to_move": torch.stack([b["input"]["side_to_move"] for b in batch]),
        "castling": torch.stack([b["input"]["castling"] for b in batch]),
    }

    if edge_attrs:
        result["edge_attr"] = torch.cat(edge_attrs, dim=0)

    return result


def collate_fn(batch: list[dict]) -> dict:
    first_input = batch[0]["input"]
    result: dict[str, Any] = {}

    if isinstance(first_input, torch.Tensor):
        result["input"] = torch.stack([b["input"] for b in batch])
    elif _is_graph_input(first_input):
        result["input"] = _collate_graph_inputs(batch)
    else:
        result["input"] = {
            key: (torch.stack(vals) if torch.is_tensor(vals[0]) else vals)
            for key in first_input
            for vals in [[b["input"][key] for b in batch]]
        }

    result["policy_target"] = torch.stack([b["policy_target"] for b in batch])

    if "value_target" in batch[0]:
        result["value_target"] = torch.stack([b["value_target"] for b in batch])

    return result


def create_dataloader(
    db_path: str | Path,
    encoder,
    batch_size: int = 256,
    num_workers: int = 0,
    include_value: bool = True,
    num_samples: int | None = None,
) -> DataLoader:
    """
    Create a DataLoader from a single Parquet file.

    Data is streamed row-group by row-group so the full file is never
    loaded into memory.  Data is assumed to be pre-shuffled.
    """
    dataset = ChessDataset(
        path=Path(db_path),
        encoder=encoder,
        num_samples=num_samples,
        include_value=include_value,
    )

    loader_kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": True,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **loader_kwargs)
