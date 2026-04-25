#!/usr/bin/env python3
"""
Interactive step-based terminal chess benchmark client.

Features:
- Unicode chess board rendered in the terminal
- Model debug info: full policy distribution, value head output
- Stockfish position evaluation with best line
- Step through moves one at a time or let the model auto-play
- Side-by-side comparison of model vs Stockfish move rankings
"""

import argparse
import contextlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine
import torch
import torch.nn.functional as F

from src.agents.learning_agent import LearningAgent
from src.chess_env.move_index import INDEX_TO_UCI_MOVE, UCI_MOVE_TO_INDEX
from src.config import settings
from src.device import get_device
from src.models.factory import create_model, get_encoder_for_model

LOGGER = logging.getLogger(__name__)

PIECE_SYMBOLS = {
    "R": "♜",
    "N": "♞",
    "B": "♝",
    "Q": "♛",
    "K": "♚",
    "P": "♟",
    "r": "♖",
    "n": "♘",
    "b": "♗",
    "q": "♕",
    "k": "♔",
    "p": "♙",
}

TRAINED_MODELS = [
    "convnet",
    "resnet",
    "square_transformer",
    "piece_transformer",
    "gcn",
    "gat",
]

EVAL_DEPTH = 18


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def colorize(text: str, fg: int) -> str:
    return f"\033[{fg}m{text}\033[0m"


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


def render_board(board: chess.Board, last_move: chess.Move | None = None) -> str:
    highlight = {last_move.from_square, last_move.to_square} if last_move else set()
    header = bold("     a   b   c   d   e   f   g   h")
    lines = ["", header, "   ┌───┬───┬───┬───┬───┬───┬───┬───┐"]

    for rank in range(7, -1, -1):
        row = f" {rank + 1} │"
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            symbol = PIECE_SYMBOLS.get(piece.symbol(), piece.symbol()) if piece else " "

            if sq in highlight:
                cell = f"\033[43m {symbol} \033[0m"
            elif (rank + file) % 2 == 1:
                cell = f"\033[47m\033[30m {symbol} \033[0m"
            else:
                cell = f"\033[100m\033[97m {symbol} \033[0m"

            row += cell + "│"
        lines.append(f"{row} {rank + 1}")
        if rank > 0:
            lines.append("   ├───┼───┼───┼───┼───┼───┼───┼───┤")

    lines += ["   └───┴───┴───┴───┴───┴───┴───┴───┘", header, ""]
    return "\n".join(lines)


def format_eval(cp: int) -> str:
    if abs(cp) >= 9000:
        mate_in = (10000 - abs(cp)) // 10
        return f"#{'+' if cp > 0 else '-'}{mate_in}"
    return f"{'+' if cp >= 0 else ''}{cp / 100:.2f}"


def eval_bar(cp: int, width: int = 30) -> str:
    filled = int(((max(-500, min(500, cp)) + 500) / 1000) * width)
    bar = f"[{'█' * filled}{'░' * (width - filled)}] {format_eval(cp)}"
    if cp >= 100:
        return colorize(bar, 32)
    return colorize(bar, 31) if cp <= -100 else bar


def _score_to_cp(score) -> int:
    """Convert a chess.engine score to centipawns."""
    if score.is_mate():
        mate_in = score.mate()
        return int((10000 - abs(mate_in) * 10) * (1 if mate_in > 0 else -1))
    return int(score.score() or 0)


def evaluate_position(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int = EVAL_DEPTH) -> dict:
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=3)
        pv_list = info if isinstance(info, list) else [info]
        results = [
            {
                "cp": _score_to_cp(pv["score"].white()),  # type: ignore
                "pv": [m.uci() for m in pv.get("pv", [])[:6]],
            }
            for pv in pv_list
        ]
        return {"cp": results[0]["cp"], "lines": results}
    except Exception:
        return {"cp": 0, "lines": []}


def get_model_debug_info(
    agent: LearningAgent,
    board: chess.Board,
    device: torch.device,
    show_illegal: bool = False,
) -> dict:
    encoded = agent.encoder.encode(board)

    if isinstance(encoded, torch.Tensor):
        x = encoded.unsqueeze(0).to(device)
    elif isinstance(encoded, dict):
        x = {
            k: (v.to(device) if k in ("edge_index", "edge_attr") else v.unsqueeze(0).to(device))
            if torch.is_tensor(v)
            else v
            for k, v in encoded.items()
        }
    else:
        x = encoded

    with torch.no_grad():
        output = agent.model(x)

    result: dict[str, Any] = {}

    if "value" in output:
        result["value"] = output["value"][0].item()

    if "policy" in output:
        policy_logits = output["policy"][0]
        legal_pairs = [(m, idx) for m in board.legal_moves if (idx := UCI_MOVE_TO_INDEX.get(m.uci(), -1)) >= 0]

        if legal_pairs:
            legal_indices_set = {idx for _, idx in legal_pairs}

            mask = torch.full_like(policy_logits, float("-inf"))
            for _, i in legal_pairs:
                mask[i] = 0
            probs = F.softmax(policy_logits + mask, dim=-1)

            move_probs = sorted(
                [
                    {
                        "move": m,
                        "san": board.san(m),
                        "prob": probs[idx].item(),
                        "logit": policy_logits[idx].item(),
                        "legal": True,
                    }
                    for m, idx in legal_pairs
                ],
                key=lambda x: x["prob"],  # type: ignore
                reverse=True,
            )  # type: ignore
            result["move_probs"] = move_probs
            result["entropy"] = -(probs[probs > 0] * probs[probs > 0].log()).sum().item()
            result["top1_prob"] = move_probs[0]["prob"] if move_probs else 0
            result["num_legal"] = len(list(board.legal_moves))

            # --- Illegal move analysis (raw logits, no legal-move mask) ---
            if show_illegal:
                raw_probs = F.softmax(policy_logits, dim=-1)
                # Gather top illegal moves by raw logit
                illegal_entries: list[dict] = []
                _, top_indices = policy_logits.topk(min(200, policy_logits.size(0)))
                for idx_t in top_indices:
                    idx = idx_t.item()
                    if idx in legal_indices_set:
                        continue
                    uci = INDEX_TO_UCI_MOVE.get(idx)
                    if uci is None:
                        continue
                    illegal_entries.append(
                        {
                            "uci": uci,
                            "prob_raw": raw_probs[idx].item(),
                            "logit": policy_logits[idx].item(),
                            "legal": False,
                        }
                    )
                    if len(illegal_entries) >= 10:
                        break
                result["illegal_move_probs"] = illegal_entries

                # Raw (unmasked) entropy for comparison
                result["entropy_raw"] = -(raw_probs[raw_probs > 0] * raw_probs[raw_probs > 0].log()).sum().item()

    return result


def render_policy_bar(prob: float, width: int = 20) -> str:
    filled = int(prob * width)
    return "█" * filled + "░" * (width - filled)


def _pv_to_san(board: chess.Board, pv_uci: list[str]) -> list[str]:
    """Convert a UCI PV line to SAN moves, stopping on illegal moves."""
    pv_board = board.copy()
    san_moves = []
    for uci in pv_uci[:8]:
        try:
            m = chess.Move.from_uci(uci)
            if m not in pv_board.legal_moves:
                break
            san_moves.append(pv_board.san(m))
            pv_board.push(m)
        except ValueError:
            break
    return san_moves


def _format_pv_with_numbers(board: chess.Board, san_moves: list[str]) -> str:
    """Format SAN moves with proper move numbers."""
    ply = board.ply()
    parts = []
    for j, san in enumerate(san_moves):
        cur_ply = ply + j
        if cur_ply % 2 == 0:
            parts.append(f"{cur_ply // 2 + 1}. {san}")
        elif j == 0:
            parts.append(f"{cur_ply // 2 + 1}... {san}")
        else:
            parts.append(san)
    return " ".join(parts)


def display_state(
    board: chess.Board,
    move_num: int,
    last_move: chess.Move | None,
    last_san: str | None,
    model_debug: dict,
    sf_eval: dict,
    model_name: str,
    elapsed: float,
    move_history: list[str],
    opponent_info: str = "",
    show_illegal: bool = False,
):
    clear_screen()

    turn = "White" if board.turn == chess.WHITE else "Black"
    header = f"  ♔ Deep Dive — {bold(model_name)} │ Move {move_num} │ {turn} to play"
    if opponent_info:
        header += f" │ {opponent_info}"

    print(f"""\
{bold("═" * 72)}
{header}
{bold("═" * 72)}
{render_board(board, last_move)}""")

    if last_san:
        print(f"  Last move: {bold(last_san)}  ({elapsed:.3f}s)")
    print()

    # Stockfish evaluation
    cp = sf_eval.get("cp", 0)
    print(f"""\
{bold("  ── Stockfish Evaluation ──────────────────────────────────────")}
  Eval: {eval_bar(cp)}""")
    for i, line in enumerate(sf_eval.get("lines", [])[:3]):
        prefix = "  ►" if i == 0 else "   "
        san_moves = _pv_to_san(board, line["pv"])
        pv_str = _format_pv_with_numbers(board, san_moves)
        print(f"{prefix} PV{i + 1}: {format_eval(line['cp']):>8}  {pv_str}")
    print()

    # Model debug info
    print(bold("  ── Model Debug ───────────────────────────────────────────────"))
    if "value" in model_debug:
        v = model_debug["value"]
        v_pct = (v + 1) / 2 * 100
        v_color = 32 if v > 0.1 else (31 if v < -0.1 else 33)
        print(f"  Value head: {colorize(f'{v:+.4f}', v_color)}  (White win prob ≈ {v_pct:.1f}%)")

    if "entropy" in model_debug:
        entropy_str = f"  Entropy:    {model_debug['entropy']:.3f}"
        if "entropy_raw" in model_debug:
            entropy_str += f"  (raw: {model_debug['entropy_raw']:.3f})"
        entropy_str += f"  │  Legal moves: {model_debug['num_legal']}"
        print(entropy_str)
    print()

    if "move_probs" in model_debug:
        top_n = model_debug["move_probs"][:10]
        print(f"""\
{bold("  ── Policy Distribution (top 10) ──────────────────────────────")}
  {"#":<4} {"Move":<8} {"Prob":>8}  {"Logit":>8}  Bar
  {"─" * 4} {"─" * 8} {"─" * 8}  {"─" * 8}  {"─" * 22}""")
        for i, mp in enumerate(top_n):
            bar = render_policy_bar(mp["prob"])
            marker = " ◄" if i == 0 else ""
            print(f"  {i + 1:<4} {mp['san']:<8} {mp['prob'] * 100:>7.2f}%  {mp['logit']:>8.2f}  {bar}{marker}")

        remaining = model_debug["move_probs"][10:]
        if remaining:
            rest_prob = sum(m["prob"] for m in remaining) * 100
            print(f"  ...  ({len(remaining)} more moves, {rest_prob:.2f}% combined)")

    # --- Illegal moves the model wants to play (greyed out) ---
    if show_illegal and "illegal_move_probs" in model_debug:
        illegal = model_debug["illegal_move_probs"]
        if illegal:
            print()
            print(
                colorize(
                    bold("  ── Illegal Moves (raw logits, no mask) ──────────────────────"),
                    90,
                )
            )
            _h = "#"
            _u = "UCI"
            _r = "Raw %"
            _l = "Logit"
            _s = "─"
            print(colorize(f"  {_h:<4} {_u:<8} {_r:>8}  {_l:>8}  Bar", 90))
            print(colorize(f"  {_s * 4} {_s * 8} {_s * 8}  {_s * 8}  {_s * 22}", 90))
            for j, ip in enumerate(illegal):
                uci_str = ip["uci"]
                raw_pct = ip["prob_raw"] * 100
                logit_val = ip["logit"]
                raw_bar = render_policy_bar(ip["prob_raw"])
                print(
                    colorize(
                        f"  {j + 1:<4} {uci_str:<8} {raw_pct:>7.2f}%  {logit_val:>8.2f}  {raw_bar}",
                        90,
                    )
                )
    print()

    # Move history
    if move_history:
        print(bold("  ── Move History ──────────────────────────────────────────────"))
        moves_line = "  "
        for i, san in enumerate(move_history):
            if i % 2 == 0:
                moves_line += f"{i // 2 + 1}. "
            moves_line += f"{san} "
            if len(moves_line) > 68:
                print(moves_line)
                moves_line = "  "
        if moves_line.strip():
            print(moves_line)
    print()

    if board.is_check():
        print(colorize("  ⚠  CHECK!", 31))
    if board.is_game_over():
        result = board.result()
        print(bold(colorize(f"  ★  Game Over: {result}", 33)))
        outcome = board.outcome()
        if outcome:
            print(f"  Reason: {outcome.termination.name}")


def load_agent(model_name: str, checkpoint_path: Path, device: torch.device) -> LearningAgent:
    model_cfg = settings.model.model_copy(update={"head": "dual"})
    model = create_model(model_name, model_cfg)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    encoder_factory = get_encoder_for_model(model_name)
    encoder = encoder_factory()

    return LearningAgent(
        model=model,
        encoder=encoder,
        device=device,
        temperature=0.0,
    )  # type: ignore


def _refresh_display(
    agent: LearningAgent,
    evaluator: chess.engine.SimpleEngine,
    board: chess.Board,
    move_num: int,
    last_move: chess.Move | None,
    last_san: str | None,
    model_name: str,
    device: torch.device,
    elapsed: float,
    move_history: list[str],
    opp_info: str,
    eval_depth: int,
    show_illegal: bool = False,
):
    """Evaluate position and refresh the TUI display."""
    sf_eval = evaluate_position(evaluator, board, eval_depth)
    model_debug = (
        get_model_debug_info(agent, board, device, show_illegal=show_illegal) if not board.is_game_over() else {}
    )
    display_state(
        board,
        move_num,
        last_move,
        last_san,
        model_debug,
        sf_eval,
        model_name,
        elapsed,
        move_history,
        opp_info,
        show_illegal=show_illegal,
    )


def interactive_loop(
    agent: LearningAgent,
    evaluator: chess.engine.SimpleEngine,
    model_name: str,
    device: torch.device,
    opponent_depth: int = 5,
    model_color: str = "white",
    skill_level: int | None = None,
    eval_depth: int = EVAL_DEPTH,
    show_illegal: bool = False,
):
    board = chess.Board()
    move_num = 0
    last_move = None
    last_san = None
    elapsed = 0.0
    move_history: list[str] = []

    opp_info = f"SF d{opponent_depth}" + (f" skill {skill_level}" if skill_level is not None else "")

    def refresh():
        return _refresh_display(
            agent,
            evaluator,
            board,
            move_num,
            last_move,
            last_san,
            model_name,
            device,
            elapsed,
            move_history,
            opp_info,
            eval_depth,
            show_illegal=show_illegal,
        )

    # Show initial state
    sf_eval = evaluate_position(evaluator, board, eval_depth)
    model_debug = get_model_debug_info(agent, board, device, show_illegal=show_illegal)
    display_state(
        board,
        move_num,
        last_move,
        last_san,
        model_debug,
        sf_eval,
        model_name,
        elapsed,
        move_history,
        opp_info,
        show_illegal=show_illegal,
    )

    model_is_white = model_color == "white"

    while not board.is_game_over():
        is_model_turn = (board.turn == chess.WHITE) == model_is_white
        actor = "model" if is_model_turn else "stockfish"
        print(
            bold("  Controls: ")
            + f"[enter] {actor} plays │ [m <uci>] {'force' if is_model_turn else 'manual'} move │ [u] undo │ [q] quit"
        )

        try:
            cmd = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            return

        if cmd == "q":
            print("  Goodbye!")
            return

        if cmd == "u":
            if move_history:
                board.pop()
                move_history.pop()
                move_num = len(move_history)
                last_move = board.peek() if board.move_stack else None
                last_san = move_history[-1] if move_history else None
                refresh()
            continue

        if cmd.startswith("m "):
            uci_str = cmd[2:].strip()
            try:
                move = chess.Move.from_uci(uci_str)
                if move not in board.legal_moves:
                    print(colorize("  Illegal move!", 31))
                    continue
                san = board.san(move)
                board.push(move)
                move_num += 1
                last_move, last_san, elapsed = move, san, 0.0
                move_history.append(san)
                refresh()
                continue
            except ValueError:
                print(colorize(f"  Invalid UCI format: {uci_str}", 31))
                continue

        # Default: auto-play
        start = time.time()
        move = (
            agent.get_move(board)
            if is_model_turn
            else evaluator.play(board, chess.engine.Limit(depth=opponent_depth)).move
        )
        if move is None:
            break

        elapsed = time.time() - start
        san = board.san(move)  # type: ignore
        board.push(move)  # type: ignore
        move_num += 1
        last_move, last_san = move, san
        move_history.append(san)
        refresh()

    print("\n  Press enter to exit...")
    with contextlib.suppress(EOFError, KeyboardInterrupt):
        input()


def main():
    parser = argparse.ArgumentParser(description="Interactive deep-dive chess benchmark")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=TRAINED_MODELS,
        help="Model architecture",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument(
        "--stockfish",
        type=str,
        default="/opt/homebrew/bin/stockfish",
        help="Stockfish binary path",
    )
    parser.add_argument("--opponent-depth", type=int, default=5, help="Stockfish opponent search depth")
    parser.add_argument("--eval-depth", type=int, default=EVAL_DEPTH, help="Stockfish evaluation depth")
    parser.add_argument(
        "--color",
        type=str,
        default="white",
        choices=["white", "black"],
        help="Model plays as",
    )
    parser.add_argument(
        "--skill-level",
        type=int,
        default=None,
        help="Stockfish skill level (0-20, None for full strength)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name (loads from runs/<name>/training/<model>/final.pt)",
    )
    parser.add_argument(
        "--show-illegal",
        action="store_true",
        default=False,
        help="Show greyed-out illegal moves the model wants to play (raw logit distribution without legal-move mask)",
    )

    args = parser.parse_args()

    if not args.checkpoint and not args.name:
        parser.error("Either --checkpoint or --name is required")

    checkpoint_path = (
        Path(f"runs/{args.name}/training/{args.model}/final.pt")
        if args.name and not args.checkpoint
        else Path(args.checkpoint)
    )

    if not checkpoint_path.exists():
        LOGGER.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    device = get_device()
    LOGGER.info("Loading %s from %s on %s...", args.model, checkpoint_path, device)

    agent = load_agent(args.model, checkpoint_path, device)

    evaluator = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    if args.skill_level is not None:
        with contextlib.suppress(chess.engine.EngineError):
            evaluator.configure({"Skill Level": args.skill_level})
    try:
        interactive_loop(
            agent=agent,
            evaluator=evaluator,
            model_name=args.model,
            device=device,
            opponent_depth=args.opponent_depth,
            model_color=args.color,
            skill_level=args.skill_level,
            eval_depth=args.eval_depth,
            show_illegal=args.show_illegal,
        )
    finally:
        evaluator.quit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
