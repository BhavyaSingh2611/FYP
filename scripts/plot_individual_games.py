"""Plot individual game evaluation trajectories and compile into markdown.

Usage:
    python scripts/plot_individual_games.py \
        --input-dir runs/50_10M/evaluation/pgn/split \
        --output-dir runs/50_10M/evaluation/figures/individual_games
"""

import argparse
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


EVAL_PATTERN = re.compile(r'\[%eval ([^\]]+)\]')
EVAL_CAP = 10.0


def parse_game_info(game_text: str) -> dict:
    result = "unknown"
    result_match = re.search(r'\[Result "([^"]+)"\]', game_text)
    if result_match:
        result = result_match.group(1)

    white_match = re.search(r'\[White "([^"]+)"\]', game_text)
    black_match = re.search(r'\[Black "([^"]+)"\]', game_text)
    white_name = white_match.group(1) if white_match else "?"
    black_name = black_match.group(1) if black_match else "?"
    model_is_white = "Learning_" in white_name

    term_match = re.search(r'\[Termination "([^"]+)"\]', game_text)
    termination = term_match.group(1) if term_match else "unknown"

    lines = game_text.split("\n")
    move_line = ""
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("["):
            move_line += " " + stripped

    raw_evals = EVAL_PATTERN.findall(move_line)

    all_evals: list[float] = []
    for i, raw in enumerate(raw_evals):
        is_white_move = (i % 2 == 0)
        if raw.startswith("#"):
            val = EVAL_CAP if is_white_move else -EVAL_CAP
        elif raw.startswith("-#"):
            val = -EVAL_CAP
        else:
            val = float(raw)
            val = max(-EVAL_CAP, min(EVAL_CAP, val))
        all_evals.append(val)

    model_side = "White" if model_is_white else "Black"
    if result == "1-0":
        outcome = "Model wins" if model_is_white else "Model loses"
    elif result == "0-1":
        outcome = "Model loses" if model_is_white else "Model wins"
    else:
        outcome = "Draw"

    return {
        "evals": all_evals,
        "result": result,
        "outcome": outcome,
        "model_side": model_side,
        "white": white_name,
        "black": black_name,
        "termination": termination,
        "model_is_white": model_is_white,
    }


def parse_games(pgn_path: Path) -> list[str]:
    games: list[str] = []
    current: list[str] = []

    with open(pgn_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('[Event ') and current:
                games.append("\n".join(current))
                current = []
            current.append(line.rstrip())

    if current:
        games.append("\n".join(current))
    return games


def plot_game(info: dict, game_num: int, model_name: str, elo: str, out_path: Path) -> None:
    evals = info["evals"]
    if not evals:
        return

    half_moves = np.arange(1, len(evals) + 1) / 2.0
    evals_arr = np.array(evals)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.fill_between(half_moves, 0, evals_arr, where=evals_arr >= 0,
                    color="#2ecc71", alpha=0.2, interpolate=True)
    ax.fill_between(half_moves, 0, evals_arr, where=evals_arr < 0,
                    color="#e74c3c", alpha=0.2, interpolate=True)

    ax.plot(half_moves, evals_arr, color="#2c3e50", linewidth=1.5)

    ax.axhline(0, color="#333333", linewidth=0.6, alpha=0.4, linestyle="--")
    ax.set_xlim(0, half_moves[-1])
    ax.set_ylim(-EVAL_CAP - 0.5, EVAL_CAP + 0.5)
    ax.set_xlabel("Move Number", fontsize=11)
    ax.set_ylabel("Eval (pawns)", fontsize=11)

    outcome_color = "#2ecc71" if "wins" in info["outcome"] else (
        "#e74c3c" if "loses" in info["outcome"] else "#95a5a6"
    )
    ax.set_title(
        f"Game {game_num}: {info['outcome']} ({info['result']}, {info['termination']})"
        f" — Model as {info['model_side']}",
        fontsize=12, fontweight="bold", color=outcome_color,
    )
    ax.grid(True, alpha=0.2, color="#cccccc")

    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_pdf(
    pdf_path: Path,
    title: str,
    summary: str,
    game_infos: list[dict],
    img_paths: list[Path],
) -> None:
    page_w, page_h = A4
    margin = 20 * mm
    usable_w = page_w - 2 * margin

    c = canvas.Canvas(str(pdf_path), pagesize=A4)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, page_h - margin - 10, title)
    c.setFont("Helvetica", 12)
    c.drawString(margin, page_h - margin - 30, summary)

    y = page_h - margin - 55

    for i, (info, img_path) in enumerate(zip(game_infos, img_paths), 1):
        text_height = 100
        img_height = usable_w * 0.4
        needed = text_height + img_height + 15

        if y - needed < margin:
            c.showPage()
            y = page_h - margin

        c.setFont("Helvetica-Bold", 13)
        outcome = info["outcome"]
        if "wins" in outcome:
            c.setFillColorRGB(0.18, 0.80, 0.44)
        elif "loses" in outcome:
            c.setFillColorRGB(0.91, 0.30, 0.24)
        else:
            c.setFillColorRGB(0.58, 0.65, 0.65)
        c.drawString(margin, y, f"Game {i}: {outcome}")
        c.setFillColorRGB(0, 0, 0)

        y -= 18
        c.setFont("Helvetica", 10)
        details = [
            f"Result: {info['result']}    Termination: {info['termination']}    "
            f"Moves: {len(info['evals']) // 2}",
            f"Model plays: {info['model_side']}    "
            f"White: {info['white']}    Black: {info['black']}",
        ]
        for line in details:
            c.drawString(margin, y, line)
            y -= 14

        y -= 5
        if img_path.exists():
            c.drawImage(
                str(img_path), margin, y - img_height,
                width=usable_w, height=img_height,
                preserveAspectRatio=True, anchor="nw",
            )
            y -= img_height + 15

    c.save()


def process_pgn(pgn_path: Path, output_dir: Path) -> None:
    games = parse_games(pgn_path)
    if not games:
        return

    stem = pgn_path.stem
    parts = stem.rsplit("_", 1)
    model_name = parts[0].replace("_", " ").title()
    elo = parts[1] if len(parts) == 2 else "?"

    img_dir = output_dir / stem
    img_dir.mkdir(parents=True, exist_ok=True)

    game_infos: list[dict] = []
    img_paths: list[Path] = []
    wins = losses = draws = 0

    for i, game_text in enumerate(games, 1):
        info = parse_game_info(game_text)
        if "wins" in info["outcome"]:
            wins += 1
        elif "loses" in info["outcome"]:
            losses += 1
        else:
            draws += 1

        img_path = img_dir / f"game_{i:02d}.png"
        plot_game(info, i, model_name, elo, img_path)
        game_infos.append(info)
        img_paths.append(img_path)

    title = f"{model_name} vs Stockfish (Elo {elo})"
    summary = f"{len(games)} games — {wins}W / {losses}L / {draws}D"

    pdf_path = output_dir / f"{stem}.pdf"
    build_pdf(pdf_path, title, summary, game_infos, img_paths)

    shutil.rmtree(img_dir)

    md_path = output_dir / f"{stem}.md"
    if md_path.exists():
        md_path.unlink()

    print(f"  {stem}: {len(games)} games → {pdf_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Plot individual game eval trajectories")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pgn_files = sorted(args.input_dir.glob("*.pgn"))
    if not pgn_files:
        print(f"No PGN files found in {args.input_dir}")
        return

    print(f"Processing {len(pgn_files)} PGN files...")
    for pgn_file in pgn_files:
        process_pgn(pgn_file, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
