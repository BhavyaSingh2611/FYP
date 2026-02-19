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
import os
import sys
import time
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn.functional as F

from src.config import load_config
from src.models.factory import create_model, get_encoder_for_model
from src.agents.learning_agent import LearningAgent
from src.agents.mcts_agent import MCTSAgent
from src.chess_env.board_wrapper import UCI_MOVE_TO_INDEX, INDEX_TO_UCI_MOVE
from src.device import get_device


PIECE_SYMBOLS = {
    'R': '♜', 'N': '♞', 'B': '♝', 'Q': '♛', 'K': '♚', 'P': '♟',
    'r': '♖', 'n': '♘', 'b': '♗', 'q': '♕', 'k': '♔', 'p': '♙',
}

TRAINED_MODELS = ["convnet", "resnet", "square_transformer", "piece_transformer", "gcn", "gat"]

EVAL_DEPTH = 18


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def colorize(text: str, fg: int) -> str:
    return f"\033[{fg}m{text}\033[0m"


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


def render_board(board: chess.Board, last_move: chess.Move | None = None) -> str:
    lines = []
    lines.append("")
    lines.append(bold("     a   b   c   d   e   f   g   h"))
    lines.append("   ┌───┬───┬───┬───┬───┬───┬───┬───┐")

    highlight_squares = set()
    if last_move:
        highlight_squares.add(last_move.from_square)
        highlight_squares.add(last_move.to_square)

    for rank in range(7, -1, -1):
        row = f" {rank + 1} │"
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            is_light = (rank + file) % 2 == 1
            is_highlight = sq in highlight_squares

            if piece:
                symbol = PIECE_SYMBOLS.get(piece.symbol(), piece.symbol())
            else:
                symbol = " "

            if is_highlight:
                cell = colorize(f" {symbol} ", 43)  # yellow bg via escape
                cell = f"\033[43m {symbol} \033[0m"
            elif is_light:
                cell = f"\033[47m\033[30m {symbol} \033[0m"
            else:
                cell = f"\033[100m\033[97m {symbol} \033[0m"

            row += cell + "│"
        row += f" {rank + 1}"
        lines.append(row)
        if rank > 0:
            lines.append("   ├───┼───┼───┼───┼───┼───┼───┼───┤")

    lines.append("   └───┴───┴───┴───┴───┴───┴───┴───┘")
    lines.append(bold("     a   b   c   d   e   f   g   h"))
    lines.append("")
    return "\n".join(lines)


def format_eval(cp: int) -> str:
    if abs(cp) >= 9000:
        mate_in = (10000 - abs(cp)) // 10
        sign = "+" if cp > 0 else "-"
        return f"#{sign}{mate_in}"
    sign = "+" if cp >= 0 else ""
    return f"{sign}{cp / 100:.2f}"


def eval_bar(cp: int, width: int = 30) -> str:
    clamped = max(-500, min(500, cp))
    ratio = (clamped + 500) / 1000
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    label = format_eval(cp)
    if cp >= 100:
        return colorize(f"[{bar}] {label}", 32)  # green
    elif cp <= -100:
        return colorize(f"[{bar}] {label}", 31)  # red
    return f"[{bar}] {label}"


def evaluate_position(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int = EVAL_DEPTH) -> dict:
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=3)
        if isinstance(info, list):
            results = []
            for pv_info in info:
                score = pv_info['score'].white()
                if score.is_mate():
                    mate_in = score.mate()
                    cp = (10000 - abs(mate_in) * 10) * (1 if mate_in > 0 else -1)
                else:
                    cp = score.score()
                pv_moves = [m.uci() for m in pv_info.get('pv', [])[:6]]
                results.append({'cp': cp, 'pv': pv_moves})
            return {'cp': results[0]['cp'], 'lines': results}
        else:
            score = info['score'].white()
            if score.is_mate():
                mate_in = score.mate()
                cp = (10000 - abs(mate_in) * 10) * (1 if mate_in > 0 else -1)
            else:
                cp = score.score()
            pv_moves = [m.uci() for m in info.get('pv', [])[:6]]
            return {'cp': cp, 'lines': [{'cp': cp, 'pv': pv_moves}]}
    except Exception:
        return {'cp': 0, 'lines': []}


def get_model_debug_info(agent: LearningAgent | MCTSAgent, board: chess.Board, device: torch.device) -> dict:
    encoder = agent.encoder
    encoded = encoder.encode(board)

    if isinstance(encoded, torch.Tensor):
        x = encoded.unsqueeze(0).to(device)
    elif isinstance(encoded, dict):
        x = {}
        for k, v in encoded.items():
            if not torch.is_tensor(v):
                x[k] = v
            elif k in ('edge_index', 'edge_attr'):
                x[k] = v.to(device)
            else:
                x[k] = v.unsqueeze(0).to(device)
    else:
        x = encoded

    with torch.no_grad():
        output = agent.model(x)

    result = {}

    if 'value' in output:
        result['value'] = output['value'][0].item()

    if 'policy' in output:
        policy_logits = output['policy'][0]

        legal_moves = list(board.legal_moves)
        legal_pairs = []
        for m in legal_moves:
            idx = UCI_MOVE_TO_INDEX.get(m.uci(), -1)
            if idx >= 0:
                legal_pairs.append((m, idx))

        if legal_pairs:
            legal_indices = [i for _, i in legal_pairs]
            mask = torch.full_like(policy_logits, float('-inf'))
            for i in legal_indices:
                mask[i] = 0
            masked_logits = policy_logits + mask
            probs = F.softmax(masked_logits, dim=-1)

            move_probs = []
            for m, idx in legal_pairs:
                move_probs.append({
                    'move': m,
                    'san': board.san(m),
                    'prob': probs[idx].item(),
                    'logit': policy_logits[idx].item(),
                })
            move_probs.sort(key=lambda x: x['prob'], reverse=True)
            result['move_probs'] = move_probs

            result['entropy'] = -(probs[probs > 0] * probs[probs > 0].log()).sum().item()
            result['top1_prob'] = move_probs[0]['prob'] if move_probs else 0
            result['num_legal'] = len(legal_moves)

    return result


def render_policy_bar(prob: float, width: int = 20) -> str:
    filled = int(prob * width)
    return "█" * filled + "░" * (width - filled)


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
):
    clear_screen()

    turn = "White" if board.turn == chess.WHITE else "Black"
    header = f"  ♔ Deep Dive — {bold(model_name)} │ Move {move_num} │ {turn} to play"
    if opponent_info:
        header += f" │ {opponent_info}"
    print(bold("═" * 72))
    print(header)
    print(bold("═" * 72))

    print(render_board(board, last_move))

    if last_san:
        print(f"  Last move: {bold(last_san)}  ({elapsed:.3f}s)")
    print()

    # Stockfish evaluation
    print(bold("  ── Stockfish Evaluation ──────────────────────────────────────"))
    cp = sf_eval.get('cp', 0)
    print(f"  Eval: {eval_bar(cp)}")
    for i, line in enumerate(sf_eval.get('lines', [])[:3]):
        label = format_eval(line['cp'])
        prefix = "  ►" if i == 0 else "   "
        # Convert UCI moves to SAN for readability
        pv_board = board.copy()
        san_moves = []
        for uci in line['pv'][:8]:
            try:
                m = chess.Move.from_uci(uci)
                if m in pv_board.legal_moves:
                    san_moves.append(pv_board.san(m))
                    pv_board.push(m)
                else:
                    break
            except ValueError:
                break
        # Format with move numbers
        pv_parts = []
        ply = board.ply()
        for j, san in enumerate(san_moves):
            cur_ply = ply + j
            if cur_ply % 2 == 0:
                pv_parts.append(f"{cur_ply // 2 + 1}. {san}")
            elif j == 0:
                pv_parts.append(f"{cur_ply // 2 + 1}... {san}")
            else:
                pv_parts.append(san)
        pv_str = " ".join(pv_parts)
        print(f"{prefix} PV{i+1}: {label:>8}  {pv_str}")
    print()

    # Model debug info
    print(bold("  ── Model Debug ───────────────────────────────────────────────"))
    if 'value' in model_debug:
        v = model_debug['value']
        v_pct = (v + 1) / 2 * 100
        if v > 0.1:
            v_color = 32
        elif v < -0.1:
            v_color = 31
        else:
            v_color = 33
        print(f"  Value head: {colorize(f'{v:+.4f}', v_color)}  (White win prob ≈ {v_pct:.1f}%)")

    if 'entropy' in model_debug:
        print(f"  Entropy:    {model_debug['entropy']:.3f}  │  Legal moves: {model_debug['num_legal']}")
    print()

    if 'move_probs' in model_debug:
        top_n = model_debug['move_probs'][:10]
        print(bold("  ── Policy Distribution (top 10) ──────────────────────────────"))
        print(f"  {'#':<4} {'Move':<8} {'Prob':>8}  {'Logit':>8}  Bar")
        print(f"  {'─'*4} {'─'*8} {'─'*8}  {'─'*8}  {'─'*22}")
        for i, mp in enumerate(top_n):
            prob_str = f"{mp['prob']*100:.2f}%"
            logit_str = f"{mp['logit']:.2f}"
            bar = render_policy_bar(mp['prob'])
            marker = " ◄" if i == 0 else ""
            print(f"  {i+1:<4} {mp['san']:<8} {prob_str:>8}  {logit_str:>8}  {bar}{marker}")

        remaining = model_debug['move_probs'][10:]
        if remaining:
            rest_prob = sum(m['prob'] for m in remaining) * 100
            print(f"  ...  ({len(remaining)} more moves, {rest_prob:.2f}% combined)")
    print()

    # Move history
    if move_history:
        print(bold("  ── Move History ──────────────────────────────────────────────"))
        moves_line = "  "
        for i, san in enumerate(move_history):
            if i % 2 == 0:
                moves_line += f"{i//2 + 1}. "
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


def load_agent(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
    use_mcts: bool = False,
    mcts_sims: int = 200,
) -> LearningAgent | MCTSAgent:
    config = load_config("config/default.yaml")
    config.model.backbone = model_name
    config.model.head = "dual"

    model = create_model(config.model)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    encoder_factory = get_encoder_for_model(model_name)
    encoder = encoder_factory() if callable(encoder_factory) else encoder_factory

    if use_mcts:
        return MCTSAgent(
            model=model,
            encoder=encoder,
            device=device,
            num_simulations=mcts_sims,
            c_puct=1.4,
            temperature=0.0,
        )
    return LearningAgent(
        model=model,
        encoder=encoder,
        device=device,
        temperature=0.0,
    )


def interactive_loop(
    agent: LearningAgent | MCTSAgent,
    evaluator: chess.engine.SimpleEngine,
    model_name: str,
    device: torch.device,
    opponent_depth: int = 5,
    model_color: str = "white",
    skill_level: int | None = None,
    eval_depth: int = EVAL_DEPTH,
):
    board = chess.Board()
    move_num = 0
    last_move = None
    last_san = None
    elapsed = 0.0
    move_history: list[str] = []

    opp_info = f"SF d{opponent_depth}"
    if skill_level is not None:
        opp_info += f" skill {skill_level}"

    # Show initial state
    sf_eval = evaluate_position(evaluator, board, eval_depth)
    model_debug = get_model_debug_info(agent, board, device)
    display_state(board, move_num, last_move, last_san, model_debug, sf_eval, model_name, elapsed, move_history, opp_info)

    model_is_white = model_color == "white"

    while not board.is_game_over():
        is_model_turn = (board.turn == chess.WHITE) == model_is_white

        if is_model_turn:
            print(bold("  Controls: ") + "[enter] model plays │ [m <uci>] force move │ [u] undo │ [q] quit")
        else:
            print(bold("  Controls: ") + "[enter] stockfish plays │ [m <uci>] manual move │ [u] undo │ [q] quit")

        try:
            cmd = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            return

        if cmd == 'q':
            print("  Goodbye!")
            return

        if cmd == 'u':
            if move_history:
                board.pop()
                move_history.pop()
                move_num = len(move_history)
                last_move = board.peek() if board.move_stack else None
                last_san = move_history[-1] if move_history else None
                sf_eval = evaluate_position(evaluator, board, eval_depth)
                model_debug = get_model_debug_info(agent, board, device)
                display_state(board, move_num, last_move, last_san, model_debug, sf_eval, model_name, elapsed, move_history, opp_info)
            continue

        if cmd.startswith('m '):
            uci_str = cmd[2:].strip()
            try:
                move = chess.Move.from_uci(uci_str)
                if move not in board.legal_moves:
                    print(colorize("  Illegal move!", 31))
                    continue
                san = board.san(move)
                board.push(move)
                move_num += 1
                last_move = move
                last_san = san
                elapsed = 0.0
                move_history.append(san)
                sf_eval = evaluate_position(evaluator, board, eval_depth)
                model_debug = get_model_debug_info(agent, board, device) if not board.is_game_over() else {}
                display_state(board, move_num, last_move, last_san, model_debug, sf_eval, model_name, elapsed, move_history, opp_info)
                continue
            except ValueError:
                print(colorize(f"  Invalid UCI format: {uci_str}", 31))
                continue

        # Default: auto-play
        start = time.time()
        if is_model_turn:
            move = agent.get_move(board)
        else:
            result = evaluator.play(board, chess.engine.Limit(depth=opponent_depth))
            move = result.move

        elapsed = time.time() - start
        san = board.san(move)
        board.push(move)
        move_num += 1
        last_move = move
        last_san = san
        move_history.append(san)

        sf_eval = evaluate_position(evaluator, board, eval_depth)
        model_debug = get_model_debug_info(agent, board, device) if not board.is_game_over() else {}
        display_state(board, move_num, last_move, last_san, model_debug, sf_eval, model_name, elapsed, move_history, opp_info)

    print("\n  Press enter to exit...")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass


def main():
    parser = argparse.ArgumentParser(description="Interactive deep-dive chess benchmark")
    parser.add_argument("--model", type=str, required=True, choices=TRAINED_MODELS, help="Model architecture")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--stockfish", type=str, default="/opt/homebrew/bin/stockfish", help="Stockfish binary path")
    parser.add_argument("--opponent-depth", type=int, default=5, help="Stockfish opponent search depth")
    parser.add_argument("--eval-depth", type=int, default=EVAL_DEPTH, help="Stockfish evaluation depth")
    parser.add_argument("--color", type=str, default="white", choices=["white", "black"], help="Model plays as")
    parser.add_argument("--skill-level", type=int, default=None, help="Stockfish skill level (0-20, None for full strength)")
    parser.add_argument("--mcts", action="store_true", help="Use MCTS agent")
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations")
    parser.add_argument("--name", type=str, default=None, help="Run name (loads from runs/<name>/training/<model>/final.pt)")

    args = parser.parse_args()

    if not args.checkpoint and not args.name:
        parser.error("Either --checkpoint or --name is required")

    eval_depth = args.eval_depth

    if args.name and not args.checkpoint:
        checkpoint_path = Path(f"runs/{args.name}/training/{args.model}/final.pt")
    else:
        checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    device = get_device()
    print(f"Loading {args.model} from {checkpoint_path} on {device}...")

    agent = load_agent(args.model, checkpoint_path, device, args.mcts, args.sims)

    evaluator = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    if args.skill_level is not None:
        try:
            evaluator.configure({"Skill Level": args.skill_level})
        except chess.engine.EngineError:
            pass
    try:
        interactive_loop(
            agent=agent,
            evaluator=evaluator,
            model_name=args.model,
            device=device,
            opponent_depth=args.opponent_depth,
            model_color=args.color,
            skill_level=args.skill_level,
            eval_depth=eval_depth,
        )
    finally:
        evaluator.quit()


if __name__ == "__main__":
    main()
