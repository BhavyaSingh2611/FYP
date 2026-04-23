"""
UCI Engine agent for wrapping external engines like Stockfish.
"""

import contextlib
from pathlib import Path
from typing import Any

import chess
import chess.engine


class UCIAgent:
    """
    Agent that wraps UCI-compatible chess engines (Stockfish, etc.).

    Uses python-chess's engine interface for communication.
    Supports MultiPV for getting move distributions with centipawn scores.
    """

    def __init__(
        self,
        engine_path: str | Path,
        depth: int = 15,
        time_limit: float | None = None,
        skill_level: int | None = None,
        uci_elo: int | None = None,
        threads: int = 1,
        hash_mb: int = 128,
        multipv: int = 5,
    ):
        """
        Initialize UCI agent.

        Args:
            engine_path: Path to UCI engine binary.
            depth: Default search depth.
            time_limit: Default time limit per move (seconds).
            skill_level: Stockfish skill level (0-20). None for full strength.
            uci_elo: Standard UCI Elo level to limit strength to.
            threads: Number of threads for engine.
            hash_mb: Hash table size in MB.
            multipv: Number of principal variations for move distribution.
        """
        self.engine_path = Path(engine_path)
        self.depth = depth
        self.time_limit = time_limit
        self.skill_level = skill_level
        self.uci_elo = uci_elo
        self.threads = threads
        self.hash_mb = hash_mb
        self.multipv = multipv

        self._engine: chess.engine.SimpleEngine | None = None
        self._name = f"UCI_{self.engine_path.stem}"

    @property
    def name(self) -> str:
        return self._name

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        """Ensure engine is running, start if needed."""
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(str(self.engine_path))

            # Configure engine options
            try:
                self._engine.configure(
                    {
                        "Threads": self.threads,
                        "Hash": self.hash_mb,
                    }
                )

                if self.uci_elo is not None:
                    self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.uci_elo})
                elif self.skill_level is not None:
                    self._engine.configure({"Skill Level": self.skill_level})
            except chess.engine.EngineError:
                # Some options may not be supported
                pass

        return self._engine

    def get_move(
        self,
        board: chess.Board,
        time_limit: float | None = None,
    ) -> chess.Move:
        """
        Get the best move from the engine.

        Args:
            board: Current chess board.
            time_limit: Time limit in seconds (overrides default).

        Returns:
            Best move.
        """
        engine = self._ensure_engine()

        # Set time limit
        limit = self._get_limit(time_limit)

        result = engine.play(board, limit)

        if result.move is None:
            # Fallback to first legal move
            return list(board.legal_moves)[0]

        return result.move

    def get_move_with_info(
        self,
        board: chess.Board,
        time_limit: float | None = None,
    ) -> dict:
        """
        Get move with evaluation score and principal variation.
        """
        engine = self._ensure_engine()
        limit = self._get_limit(time_limit)

        result = engine.play(board, limit, info=chess.engine.INFO_ALL)

        info: dict[str, Any] = {
            "move": result.move or list(board.legal_moves)[0],
        }

        if result.info:
            if "score" in result.info:
                score = result.info["score"].relative
                if score.is_mate():
                    mate = score.mate()
                    info["mate_in"] = mate
                    info["score"] = 10000 if (mate is not None and mate > 0) else -10000
                else:
                    info["score"] = score.score() or 0

            if "pv" in result.info:
                info["pv"] = [m.uci() for m in result.info["pv"][:5]]

            if "depth" in result.info:
                info["depth"] = result.info["depth"]

        return info

    def get_move_distribution(
        self,
        board: chess.Board,
        num_moves: int = 5,
        depth: int | None = None,
    ) -> list[dict]:
        """
        Get top moves with centipawn scores using MultiPV.

        Args:
            board: Current chess board.
            num_moves: Number of moves to return.
            depth: Search depth (overrides default).

        Returns:
            List of dicts with 'move' and 'score' (centipawns).
        """
        engine = self._ensure_engine()

        # Use MultiPV mode
        with contextlib.suppress(chess.engine.EngineError):
            engine.configure({"MultiPV": num_moves})

        # Set depth limit
        search_depth = depth or self.depth
        limit = chess.engine.Limit(depth=search_depth)

        # Analyze position
        with engine.analysis(board, limit, multipv=num_moves) as analysis:
            results = []
            seen_moves = set()

            for info in analysis:
                if "pv" not in info or len(info["pv"]) == 0:
                    continue

                move = info["pv"][0]
                if move.uci() in seen_moves:
                    continue
                seen_moves.add(move.uci())

                entry: dict[str, Any] = {"move": move}

                if "score" in info:
                    score = info["score"].relative
                    if score.is_mate():
                        mate_score = score.mate()
                        entry["score"] = (
                            10000 - abs(mate_score)
                            if (mate_score is not None and mate_score > 0)
                            else -10000 + abs(mate_score or 0)
                        )
                        entry["mate_in"] = mate_score
                    else:
                        entry["score"] = score.score() or 0
                else:
                    entry["score"] = 0

                results.append(entry)

                if len(results) >= num_moves:
                    break

        # Reset MultiPV
        with contextlib.suppress(chess.engine.EngineError):
            engine.configure({"MultiPV": 1})

        # Sort by score descending
        results.sort(key=lambda x: int(x.get("score", 0)), reverse=True)

        return results or [{"move": list(board.legal_moves)[0], "score": 0}]

    def _get_limit(self, time_limit: float | None = None) -> chess.engine.Limit:
        """Get engine limit based on time or depth."""
        tl = time_limit or self.time_limit

        if tl is not None:
            return chess.engine.Limit(time=tl)
        else:
            return chess.engine.Limit(depth=self.depth)

    def reset(self) -> None:
        """Reset the engine (clear hash, etc.)."""
        if self._engine is not None:
            with contextlib.suppress(chess.engine.EngineError):
                self._engine.configure({"Clear Hash": None})

    def close(self) -> None:
        """Close the engine process."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        return f"UCIAgent('{self.engine_path.name}', depth={self.depth})"
