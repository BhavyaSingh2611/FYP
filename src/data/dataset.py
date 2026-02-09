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

    Each row is expected to contain at least:
        fen            – FEN string
        best_move      – UCI move string
        result         – game result ('1-0', '0-1', '1/2-1/2')

    Optional soft-label columns:
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

        emitted = 0

        file_order = list(range(len(self.paths)))
        if self.shuffle:
            np.random.shuffle(file_order)

        for fi in file_order:
            pf = pq.ParquetFile(self.paths[fi])
            rg_order = list(range(pf.metadata.num_row_groups))
            if self.shuffle:
                np.random.shuffle(rg_order)

            for rg_idx in rg_order:
                table = pf.read_row_group(rg_idx)
                df = table.to_pandas()

                if self.shuffle:
                    df = df.sample(frac=1).reset_index(drop=True)

                for _, row in df.iterrows():
                    if self.num_samples and emitted >= self.num_samples:
                        return

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

    def _build_policy(self, row) -> torch.Tensor:
        policy = torch.zeros(NUM_MOVES)

        if self.use_soft_labels:
            indices, scores = [], []
            for i in range(20):
                uci_col = f"move_uci_{i}"
                score_col = f"move_score_{i}"
                if (
                    uci_col in row.index
                    and row[uci_col] is not None
                    and str(row[uci_col]) != "nan"
                ):
                    idx = UCI_MOVE_TO_INDEX.get(str(row[uci_col]), -1)
                    if idx >= 0:
                        indices.append(idx)
                        scores.append(float(row.get(score_col, 0)))

            if indices:
                s = torch.tensor(scores, dtype=torch.float32) / 100.0
                probs = torch.softmax(s, dim=0)
                for idx, p in zip(indices, probs):
                    policy[idx] = p
                return policy

        best = row.get("best_move") or row.get("best_move_uci")
        if best is not None:
            idx = UCI_MOVE_TO_INDEX.get(str(best), 0)
            policy[idx] = 1.0

        return policy

    def _compute_value(self, row, board: chess.Board) -> float:
        result_str = row.get("result") or row.get("game_result")
        side = 0 if board.turn == chess.WHITE else 1

        if result_str == "1-0":
            return 1.0 if side == 0 else -1.0
        elif result_str == "0-1":
            return -1.0 if side == 0 else 1.0
        return 0.0


def collate_fn(batch: list[dict]) -> dict:
    result = {}

    first_input = batch[0]["input"]
    if isinstance(first_input, torch.Tensor):
        result["input"] = torch.stack([b["input"] for b in batch])
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
