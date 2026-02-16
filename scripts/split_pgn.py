"""Split all_evaluation_games.pgn into per-model per-difficulty PGN files.

Usage:
    python scripts/split_pgn.py --input runs/50_10M/evaluation/pgn/all_evaluation_games.pgn \
                                --output-dir runs/50_10M/evaluation/pgn/split
"""

import argparse
import re
from pathlib import Path


def parse_games(pgn_path: Path) -> dict[str, list[str]]:
    games: dict[str, list[str]] = {}
    current_lines: list[str] = []
    current_key: str | None = None

    with open(pgn_path) as f:
        for line in f:
            stripped = line.strip()
            event_match = re.match(r'^\[Event "(.+)"\]$', stripped)
            if event_match:
                if current_key and current_lines:
                    games.setdefault(current_key, []).append("".join(current_lines))
                current_lines = [line]
                event = event_match.group(1)
                model, rest = event.split(" vs SF ")
                elo = rest.split("-")[-1]
                current_key = f"{model}_{elo}"
            else:
                current_lines.append(line)

    if current_key and current_lines:
        games.setdefault(current_key, []).append("".join(current_lines))

    return games


def main():
    parser = argparse.ArgumentParser(description="Split evaluation PGN by model and difficulty")
    parser.add_argument("--input", type=Path, required=True, help="Path to all_evaluation_games.pgn")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for split PGNs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    games = parse_games(args.input)

    for key, game_list in sorted(games.items()):
        out_path = args.output_dir / f"{key}.pgn"
        with open(out_path, "w") as f:
            f.write("\n".join(game_list))
        print(f"{out_path.name}: {len(game_list)} games")


if __name__ == "__main__":
    main()
