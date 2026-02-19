"""
Test how well PV-inherited evals hold up vs fresh evals.

Plays Stockfish vs Stockfish games, then at each position compares:
  - The eval inherited from the root position's PV (what PV unrolling would use)
  - A fresh independent eval at the same depth

This validates whether re-evaluating PV-unrolled positions is necessary.
"""
import argparse
import chess
import chess.engine
from pathlib import Path
import statistics


def play_and_compare(
    engine_path: str,
    play_depth: int,
    eval_depth: int,
    num_games: int,
    max_moves: int,
) -> list[dict]:
    games = []

    player = chess.engine.SimpleEngine.popen_uci(engine_path)
    evaluator = chess.engine.SimpleEngine.popen_uci(engine_path)

    for game_idx in range(num_games):
        board = chess.Board()
        positions = []
        move_num = 0

        while not board.is_game_over() and move_num < max_moves:
            result = player.analyse(board, chess.engine.Limit(depth=play_depth))
            pv = result.get("pv", [])
            root_score = result["score"].white()

            fresh = evaluator.analyse(board, chess.engine.Limit(depth=eval_depth))
            fresh_score = fresh["score"].white()

            pv_inherited = []
            sim_board = board.copy()
            for ply_offset, pv_move in enumerate(pv):
                if not sim_board.is_legal(pv_move):
                    break
                sim_board.push(pv_move)
                if sim_board.is_game_over():
                    break

                pv_eval = evaluator.analyse(
                    sim_board, chess.engine.Limit(depth=eval_depth)
                )
                pv_fresh_score = pv_eval["score"].white()

                pv_inherited.append({
                    "ply_offset": ply_offset + 1,
                    "fen": sim_board.fen(),
                    "inherited_cp": _score_to_cp(root_score),
                    "fresh_cp": _score_to_cp(pv_fresh_score),
                })

            positions.append({
                "move_num": move_num + 1,
                "fen": board.fen(),
                "root_cp": _score_to_cp(root_score),
                "fresh_cp": _score_to_cp(fresh_score),
                "pv_length": len(pv),
                "pv_positions": pv_inherited,
            })

            best_move = pv[0] if pv else list(board.legal_moves)[0]
            board.push(best_move)
            move_num += 1

        games.append({
            "game": game_idx + 1,
            "moves": move_num,
            "result": board.result() if board.is_game_over() else "truncated",
            "positions": positions,
        })
        print(f"Game {game_idx + 1}/{num_games} complete: {move_num} moves")

    player.quit()
    evaluator.quit()
    return games


def _score_to_cp(score: chess.engine.Score) -> float:
    if score.is_mate():
        mate_in = score.mate()
        return 10000.0 if mate_in > 0 else -10000.0
    cp = score.score()
    return float(cp) if cp is not None else 0.0


def analyse_drift(games: list[dict]):
    print("\n" + "=" * 70)
    print("EVAL DRIFT ANALYSIS: Inherited PV Eval vs Fresh Eval")
    print("=" * 70)

    by_ply = {}
    root_diffs = []

    for game in games:
        for pos in game["positions"]:
            root_diffs.append(abs(pos["root_cp"] - pos["fresh_cp"]))

            for pv_pos in pos["pv_positions"]:
                ply = pv_pos["ply_offset"]
                diff = abs(pv_pos["inherited_cp"] - pv_pos["fresh_cp"])
                by_ply.setdefault(ply, []).append(diff)

    print(f"\nRoot eval consistency (play_depth eval vs eval_depth eval):")
    print(f"  Mean |diff|: {statistics.mean(root_diffs):.1f} cp")
    print(f"  Median |diff|: {statistics.median(root_diffs):.1f} cp")
    print(f"  Samples: {len(root_diffs)}")

    print(f"\nPV-inherited eval drift by ply offset:")
    print(f"{'Ply':>4} | {'Mean |diff|':>12} | {'Median':>8} | {'P90':>8} | {'P99':>8} | {'Samples':>8}")
    print("-" * 65)

    all_diffs = []
    for ply in sorted(by_ply.keys()):
        diffs = by_ply[ply]
        all_diffs.extend(diffs)
        sorted_d = sorted(diffs)
        p90 = sorted_d[int(len(sorted_d) * 0.9)] if len(sorted_d) >= 10 else max(sorted_d)
        p99 = sorted_d[int(len(sorted_d) * 0.99)] if len(sorted_d) >= 100 else max(sorted_d)
        print(
            f"{ply:>4} | {statistics.mean(diffs):>10.1f}cp | {statistics.median(diffs):>6.1f}cp"
            f" | {p90:>6.1f}cp | {p99:>6.1f}cp | {len(diffs):>8}"
        )

    if all_diffs:
        print(f"\nOverall PV drift:")
        print(f"  Mean |diff|: {statistics.mean(all_diffs):.1f} cp")
        print(f"  Median |diff|: {statistics.median(all_diffs):.1f} cp")

        threshold = 50
        within = sum(1 for d in all_diffs if d <= threshold)
        print(f"  Within ±{threshold}cp: {within}/{len(all_diffs)} ({100*within/len(all_diffs):.1f}%)")

        threshold = 100
        within = sum(1 for d in all_diffs if d <= threshold)
        print(f"  Within ±{threshold}cp: {within}/{len(all_diffs)} ({100*within/len(all_diffs):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Test PV eval inheritance drift")
    parser.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--play-depth", type=int, default=18, help="Depth for playing moves")
    parser.add_argument("--eval-depth", type=int, default=18, help="Depth for fresh evaluation")
    parser.add_argument("--games", type=int, default=3, help="Number of games to play")
    parser.add_argument("--max-moves", type=int, default=80, help="Max moves per game")
    args = parser.parse_args()

    print(f"Engine: {args.engine}")
    print(f"Play depth: {args.play_depth}, Eval depth: {args.eval_depth}")
    print(f"Games: {args.games}, Max moves: {args.max_moves}")

    games = play_and_compare(
        engine_path=args.engine,
        play_depth=args.play_depth,
        eval_depth=args.eval_depth,
        num_games=args.games,
        max_moves=args.max_moves,
    )

    analyse_drift(games)


if __name__ == "__main__":
    main()
