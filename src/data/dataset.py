"""
PyTorch IterableDataset for loading chess training data from Parquet files.

Streams row-groups one at a time so the full file is never resident in memory.
"""

from pathlib import Path

import chess
import torch
from torch.utils.data import DataLoader, IterableDataset

from ..chess_env.board_wrapper import NUM_MOVES, UCI_MOVE_TO_INDEX


class ChessDataset(IterableDataset):
    """
    Streams training examples from one or more Parquet files.

    Parquet schema:
        f – FEN string
        b – best move in UCI notation
        v – pre-computed value (-1.0 to +1.0, from side-to-move perspective)

    Only one row-group is in memory at a time, so RAM usage stays bounded
    regardless of file size.
    """

    def __init__(
        self,
        paths: list[Path],
        encoder,
        num_samples: int | None = None,
        include_value: bool = True,
        shuffle: bool = True,
    ):
        super().__init__()
        self.paths = [Path(p) for p in paths]
        self.encoder = encoder
        self.num_samples = num_samples
        self.include_value = include_value
        self.shuffle = shuffle

        import pyarrow.parquet as pq

        self._total_rows = 0
        for p in self.paths:
            meta = pq.read_metadata(p)
            self._total_rows += meta.num_rows
        self._effective_len = (
            min(self.num_samples, self._total_rows)
            if self.num_samples
            else self._total_rows
        )

    def __len__(self) -> int:
        return self._effective_len

    def __iter__(self):
        import numpy as np
        import pyarrow.parquet as pq

        worker_info = torch.utils.data.get_worker_info()

        all_row_groups = []
        for fi, p in enumerate(self.paths):
            meta = pq.read_metadata(p)
            for rg in range(meta.num_row_groups):
                all_row_groups.append((fi, rg))

        if worker_info is not None:
            all_row_groups = all_row_groups[worker_info.id :: worker_info.num_workers]

        if self.shuffle:
            np.random.shuffle(all_row_groups)

        per_worker_limit = None
        if self.num_samples:
            n_workers = worker_info.num_workers if worker_info else 1
            per_worker_limit = self.num_samples // n_workers

        emitted = 0

        has_batch_encode = hasattr(self.encoder, "encode_batch")

        for fi, rg_idx in all_row_groups:
            pf = pq.ParquetFile(self.paths[fi])
            table = pf.read_row_group(rg_idx)
            fens = table.column("f").to_pylist()
            best_moves = table.column("b").to_pylist()
            values = table.column("v").to_pylist()
            n_rows = table.num_rows

            indices = np.arange(n_rows)
            if self.shuffle:
                np.random.shuffle(indices)

            if has_batch_encode:
                batch_fens = []
                batch_best = []
                batch_vals = []
                for i in indices:
                    if per_worker_limit and emitted >= per_worker_limit:
                        break
                    fen = fens[i]
                    if fen is None:
                        continue
                    batch_fens.append(fen)
                    batch_best.append(best_moves[i])
                    batch_vals.append(values[i])

                if not batch_fens:
                    continue

                boards = []
                valid_mask = []
                for fen in batch_fens:
                    try:
                        boards.append(chess.Board(fen))
                        valid_mask.append(True)
                    except ValueError:
                        boards.append(None)
                        valid_mask.append(False)

                valid_boards = [b for b in boards if b is not None]
                if not valid_boards:
                    continue

                encoded_batch = self.encoder.encode_batch(valid_boards)
                enc_idx = 0
                for j, valid in enumerate(valid_mask):
                    if per_worker_limit and emitted >= per_worker_limit:
                        return
                    if not valid:
                        continue
                    policy_target = self._build_policy(batch_best[j])
                    result = {
                        "input": encoded_batch[enc_idx],
                        "policy_target": policy_target,
                    }
                    if self.include_value:
                        result["value_target"] = torch.tensor(
                            [batch_vals[j]],
                            dtype=torch.float32,
                        )
                    enc_idx += 1
                    emitted += 1
                    yield result
            else:
                for i in indices:
                    if per_worker_limit and emitted >= per_worker_limit:
                        return

                    example = self._row_to_example(fens[i], best_moves[i], values[i])
                    if example is not None:
                        emitted += 1
                        yield example

    def _row_to_example(
        self, fen: str | None, best_move: str | None, value: float
    ) -> dict | None:
        if fen is None:
            return None

        try:
            board = chess.Board(fen)
        except ValueError:
            return None

        encoded = self.encoder.encode(board)
        policy_target = self._build_policy(best_move)

        result = {"input": encoded, "policy_target": policy_target}

        if self.include_value:
            result["value_target"] = torch.tensor([value], dtype=torch.float32)

        return result

    def _build_policy(self, best_move: str | None) -> torch.Tensor:
        if best_move is not None and str(best_move) != "nan":
            idx = UCI_MOVE_TO_INDEX.get(str(best_move), -1)
            if idx >= 0:
                return torch.tensor(idx, dtype=torch.long)
        return torch.tensor(0, dtype=torch.long)


def _is_graph_input(first_input: dict) -> bool:
    return isinstance(first_input, dict) and "edge_index" in first_input


def _collate_graph_inputs(batch: list[dict]) -> dict:
    """Batch graph inputs by concatenating edges with node-index offsets."""
    num_nodes_per_graph = 64

    node_features = torch.stack([b["input"]["x"] for b in batch])  # (B, 64, F)
    batch_size = node_features.size(0)

    edge_indices = []
    edge_attrs = []
    for i, b in enumerate(batch):
        offset = i * num_nodes_per_graph
        edge_indices.append(b["input"]["edge_index"] + offset)
        if "edge_attr" in b["input"]:
            edge_attrs.append(b["input"]["edge_attr"])

    batched_edge_index = torch.cat(edge_indices, dim=1)  # (2, total_E)

    graph_batch = (
        torch.arange(batch_size)
        .unsqueeze(1)
        .expand(-1, num_nodes_per_graph)
        .reshape(-1)
    )

    result = {
        "x": node_features.view(-1, node_features.size(-1)),  # (B*64, F)
        "edge_index": batched_edge_index,
        "batch": graph_batch,
        "side_to_move": torch.stack([b["input"]["side_to_move"] for b in batch]),
        "castling": torch.stack([b["input"]["castling"] for b in batch]),
    }

    if edge_attrs:
        result["edge_attr"] = torch.cat(edge_attrs, dim=0)  # (total_E, 3)

    return result


def collate_fn(batch: list[dict]) -> dict:
    result = {}

    first_input = batch[0]["input"]
    if isinstance(first_input, torch.Tensor):
        result["input"] = torch.stack([b["input"] for b in batch])
    elif _is_graph_input(first_input):
        result["input"] = _collate_graph_inputs(batch)
    else:
        result["input"] = {}
        for key in first_input:
            values = [b["input"][key] for b in batch]
            if torch.is_tensor(values[0]):
                result["input"][key] = torch.stack(values)
            else:
                result["input"][key] = values

    result["policy_target"] = torch.stack([b["policy_target"] for b in batch])

    if "value_target" in batch[0]:
        result["value_target"] = torch.stack([b["value_target"] for b in batch])

    return result


def create_dataloader(
    db_path: str | Path,
    encoder,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    include_value: bool = True,
    num_samples: int | None = None,
) -> DataLoader:
    """
    Create a DataLoader from Parquet file(s).

    Accepts a single .parquet file or a directory of .parquet files.
    Data is streamed row-group by row-group so the full file is never
    loaded into memory.
    """
    db_path = Path(db_path)

    parquet_files = sorted(db_path.glob("*.parquet")) if db_path.is_dir() else [db_path]

    dataset = ChessDataset(
        paths=parquet_files,
        encoder=encoder,
        num_samples=num_samples,
        include_value=include_value,
        shuffle=shuffle,
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
