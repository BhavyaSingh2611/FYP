"""
PyTorch IterableDataset for loading chess training data from Parquet files.

Streams row-groups one at a time so the full file is never resident in memory.
"""
from pathlib import Path
from typing import Optional

import chess
import torch
from torch.utils.data import DataLoader, IterableDataset

from ..chess_env.board_wrapper import UCI_MOVE_TO_INDEX, NUM_MOVES


class ChessDataset(IterableDataset):
    """
    Streams training examples from one or more Parquet files.

    Supports two Parquet schemas:

    Engine-evaluated positions (primary):
        fen    – FEN string
        line   – PV line in UCI (first move = best move)
        cp     – centipawn evaluation (None if mate)
        mate   – mate-in-N evaluation (None if no mate)

    Legacy multi-move format:
        fen            – FEN string
        best_move      – UCI move string
        result         – game result ('1-0', '0-1', '1/2-1/2')
        move_uci_0 … move_uci_N   – UCI strings for the top-N moves
        move_score_0 … move_score_N – centipawn scores for those moves

    Only one row-group is in memory at a time, so RAM usage stays bounded
    regardless of file size.
    """

    def __init__(
        self,
        paths: list[Path],
        encoder,
        num_samples: int | None = None,
        use_soft_labels: bool = True,
        include_value: bool = True,
        shuffle: bool = True,
    ):
        super().__init__()
        self.paths = [Path(p) for p in paths]
        self.encoder = encoder
        self.num_samples = num_samples
        self.use_soft_labels = use_soft_labels
        self.include_value = include_value
        self.shuffle = shuffle

        import pyarrow.parquet as pq

        self._total_rows = 0
        for p in self.paths:
            meta = pq.read_metadata(p)
            self._total_rows += meta.num_rows
        self._effective_len = (
            min(self.num_samples, self._total_rows) if self.num_samples else self._total_rows
        )

    def __len__(self) -> int:
        return self._effective_len

    def __iter__(self):
        import pyarrow.parquet as pq
        import numpy as np

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

        for fi, rg_idx in all_row_groups:
            pf = pq.ParquetFile(self.paths[fi])
            table = pf.read_row_group(rg_idx)
            cols = {name: table.column(name).to_pylist() for name in table.schema.names}
            n_rows = table.num_rows

            indices = np.arange(n_rows)
            if self.shuffle:
                np.random.shuffle(indices)

            for i in indices:
                if per_worker_limit and emitted >= per_worker_limit:
                    return

                row = {k: v[i] for k, v in cols.items()}
                example = self._row_to_example(row)
                if example is not None:
                    emitted += 1
                    yield example

    def _row_to_example(self, row) -> dict | None:
        fen = row.get("fen") or row.get("FEN")
        if fen is None:
            return None

        try:
            board = chess.Board(fen)
        except ValueError:
            return None

        encoded = self.encoder.encode(board)
        policy_target = self._build_policy(row)

        result = {"input": encoded, "policy_target": policy_target}

        if self.include_value:
            result["value_target"] = torch.tensor(
                [self._compute_value(row, board)],
                dtype=torch.float32,
            )

        return result

    def _extract_best_move(self, row) -> str | None:
        best = row.get("best_move") or row.get("best_move_uci")
        if best is not None and str(best) != "nan":
            return str(best)

        line = row.get("line")
        if line is not None and str(line) != "nan":
            first_move = str(line).strip().split()[0]
            if first_move:
                return first_move

        return None

    def _build_policy(self, row) -> torch.Tensor:
        policy = torch.zeros(NUM_MOVES)

        if self.use_soft_labels:
            indices, scores = [], []
            for i in range(20):
                uci_col = f"move_uci_{i}"
                score_col = f"move_score_{i}"
                uci_val = row.get(uci_col)
                if (
                    uci_val is not None
                    and str(uci_val) != "nan"
                ):
                    idx = UCI_MOVE_TO_INDEX.get(str(uci_val), -1)
                    if idx >= 0:
                        indices.append(idx)
                        scores.append(float(row.get(score_col, 0)))

            if indices:
                s = torch.tensor(scores, dtype=torch.float32)
                s = s / s.abs().max().clamp(min=1.0)
                probs = torch.softmax(s, dim=0)
                for idx, p in zip(indices, probs):
                    policy[idx] = p
                return policy

        best = self._extract_best_move(row)
        if best is not None:
            idx = UCI_MOVE_TO_INDEX.get(best, -1)
            if idx >= 0:
                policy[idx] = 1.0

        return policy

    def _compute_value(self, row, board: chess.Board) -> float:
        result_str = row.get("result") or row.get("game_result")
        side = 0 if board.turn == chess.WHITE else 1

        if result_str is not None and str(result_str) != "nan":
            if result_str == "1-0":
                return 1.0 if side == 0 else -1.0
            elif result_str == "0-1":
                return -1.0 if side == 0 else 1.0
            elif result_str == "1/2-1/2":
                return 0.0

        mate = row.get("mate")
        if mate is not None and str(mate) != "nan":
            mate = int(mate)
            val = 1.0 if mate > 0 else -1.0
            return val if side == 0 else -val

        cp = row.get("cp")
        if cp is not None and str(cp) != "nan":
            val = max(-1.0, min(1.0, float(cp) / 1000.0))
            return val if side == 0 else -val

        return 0.0


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

    graph_batch = torch.arange(batch_size).unsqueeze(1).expand(-1, num_nodes_per_graph).reshape(-1)

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
        for key in first_input.keys():
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
    use_soft_labels: bool = True,
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

    if db_path.is_dir():
        parquet_files = sorted(db_path.glob("*.parquet"))
    else:
        parquet_files = [db_path]

    dataset = ChessDataset(
        paths=parquet_files,
        encoder=encoder,
        num_samples=num_samples,
        use_soft_labels=use_soft_labels,
        include_value=include_value,
        shuffle=shuffle,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
