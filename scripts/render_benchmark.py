from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

matplotlib.use("Agg")

EVAL_CAP = 10.0
SMOOTH_WINDOW = 5

MODEL_LABELS = {
    "convnet": "ConvNet",
    "resnet": "ResNet",
    "square_transformer": "SqTransformer",
    "piece_transformer": "PcTransformer",
    "gcn": "GCN",
    "gat": "GAT",
}

OUTCOME_COLORS = {"win": "#2ecc71", "loss": "#e74c3c", "draw": "#95a5a6"}
OUTCOME_LABELS = {"win": "Model wins", "loss": "Model loses", "draw": "Draw"}


def smooth_series(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Return a moving-average smoothed series while preserving edges."""
    if len(values) < window:
        return values

    kernel = np.ones(window) / window
    smoothed_values = np.convolve(values, kernel, mode="same")
    edge_width = window // 2

    smoothed_values[:edge_width] = values[:edge_width]
    smoothed_values[-edge_width:] = values[-edge_width:]

    return smoothed_values


def _series_model_perspective(game_record: dict[str, Any]) -> np.ndarray:
    evaluations = np.array(game_record["evaluations"], dtype=float) / 100.0
    if game_record["model_color"] == "white":
        return np.clip(evaluations, -EVAL_CAP, EVAL_CAP)

    return np.clip(-evaluations, -EVAL_CAP, EVAL_CAP)


def _plot_level_cumulative_graph(
    level_name: str,
    elo: int,
    games: list[dict[str, Any]],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 5))
    cumulative_series_by_game: list[np.ndarray] = []

    for game_number, game in enumerate(games, start=1):
        series = _series_model_perspective(game)
        if len(series) == 0:
            continue

        cumulative_series = np.cumsum(series)
        cumulative_series_by_game.append(cumulative_series)

        x_axis = np.arange(1, len(cumulative_series) + 1) / 2.0
        axis.plot(
            x_axis,
            cumulative_series,
            alpha=0.2,
            linewidth=1.0,
            label=f"Game {game_number}" if game_number == 1 else None,
        )

    if cumulative_series_by_game:
        max_length = max(len(series) for series in cumulative_series_by_game)
        padded = np.full((len(cumulative_series_by_game), max_length), np.nan)

        for index, series in enumerate(cumulative_series_by_game):
            padded[index, : len(series)] = series

        mean_cumulative = np.nanmean(padded, axis=0)
        valid_points = ~np.isnan(mean_cumulative)

        x_mean = np.arange(1, len(mean_cumulative) + 1) / 2.0
        axis.plot(
            x_mean[valid_points],
            mean_cumulative[valid_points],
            color="#1f77b4",
            linewidth=2.8,
            label="Average cumulative eval",
        )

    axis.axhline(0, color="#333333", linewidth=0.8, alpha=0.5, linestyle="--")
    axis.set_title(
        f"Cumulative Evaluation — {level_name} (Elo {elo})",
        fontsize=13,
        fontweight="bold",
    )

    axis.set_xlabel("Move Number", fontsize=11)
    axis.set_ylabel("Cumulative eval (model perspective, pawns)", fontsize=11)
    axis.grid(True, alpha=0.25)

    if axis.get_legend_handles_labels()[0]:
        axis.legend(loc="best", fontsize=9)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_level_eval_overlay_graph(
    level_name: str,
    elo: int,
    games: list[dict[str, Any]],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 5))
    traces_by_outcome: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "win": [],
        "loss": [],
        "draw": [],
    }

    for game in games:
        series = _series_model_perspective(game)
        if len(series) == 0:
            continue

        move_numbers = np.arange(1, len(series) + 1) / 2.0
        traces_by_outcome[game["outcome"]].append((move_numbers, smooth_series(series)))

    for outcome in ("win", "loss", "draw"):
        traces = traces_by_outcome[outcome]
        if not traces:
            continue

        color = OUTCOME_COLORS[outcome]
        for move_numbers, series in traces:
            axis.plot(move_numbers, series, color=color, alpha=0.18, linewidth=1.0)

        max_length = max(len(series) for _, series in traces)
        padded = np.full((len(traces), max_length), np.nan)
        for index, (_, series) in enumerate(traces):
            padded[index, : len(series)] = series

        mean_series = np.nanmean(padded, axis=0)
        valid_points = ~np.isnan(mean_series)

        mean_x = np.arange(1, len(mean_series) + 1) / 2.0
        axis.plot(
            mean_x[valid_points],
            mean_series[valid_points],
            color=color,
            linewidth=2.6,
            label=f"{OUTCOME_LABELS[outcome]} (avg)",
        )

    axis.axhline(0, color="#333333", linewidth=0.8, alpha=0.5, linestyle="--")
    axis.set_ylim(-EVAL_CAP - 0.5, EVAL_CAP + 0.5)
    axis.set_title(
        f"Evaluation Trajectories — {level_name} (Elo {elo})",
        fontsize=13,
        fontweight="bold",
    )

    axis.set_xlabel("Move Number", fontsize=11)
    axis.set_ylabel("Eval (model perspective, pawns)", fontsize=11)
    axis.grid(True, alpha=0.25)

    if axis.get_legend_handles_labels()[0]:
        axis.legend(loc="best", fontsize=9)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _build_level_pdf(
    pdf_path: Path,
    title: str,
    summary_lines: list[str],
    cumulative_image_path: Path,
    eval_image_path: Path,
) -> None:
    page_width, page_height = A4
    margin = 18 * mm
    usable_width = page_width - 2 * margin

    pdf_canvas = canvas.Canvas(str(pdf_path), pagesize=A4)
    pdf_canvas.setFont("Helvetica-Bold", 17)
    pdf_canvas.drawString(margin, page_height - margin - 8, title)

    pdf_canvas.setFont("Helvetica", 11)
    y_position = page_height - margin - 28
    for line in summary_lines:
        pdf_canvas.drawString(margin, y_position, line)
        y_position -= 14

    y_position -= 6
    image_height = usable_width * 0.42
    image_sections = [
        ("Cumulative eval graph", cumulative_image_path),
        ("All-game eval trajectories", eval_image_path),
    ]

    for section_label, image_path in image_sections:
        if y_position - image_height - 18 < margin:
            pdf_canvas.showPage()
            y_position = page_height - margin

        pdf_canvas.setFont("Helvetica-Bold", 12)
        pdf_canvas.drawString(margin, y_position, section_label)
        y_position -= 14

        if image_path.exists():
            pdf_canvas.drawImage(
                str(image_path),
                margin,
                y_position - image_height,
                width=usable_width,
                height=image_height,
                preserveAspectRatio=True,
                anchor="nw",
            )

        y_position -= image_height + 18

    pdf_canvas.save()


def _write_cumulative_markdown(
    model_name: str,
    levels_data: dict[str, dict[str, Any]],
    report_path: Path,
) -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Cumulative Benchmark Report",
        "",
        f"- Generated: {generated_at}",
        f"- Model: {model_name}",
        "",
    ]

    for level_name, payload in levels_data.items():
        summary = payload["summary"]
        elo = payload["elo"]
        games = payload["games"]
        slug = f"{model_name}_{elo}"

        cumulative_image = f"figures/levels/{slug}_cumulative.png"
        eval_overlay_image = f"figures/levels/{slug}_evals.png"

        lines.extend(
            [
                f"## {level_name} (Elo {elo})",
                "",
                f"- Games: {len(games)}",
                f"- W/D/L: {summary['wins']}/{summary['draws']}/{summary['losses']}",
                f"- Score: {summary['score_pct']:.1f}%",
                f"- Avg ACPL: {summary['avg_acpl']:.1f}",
                f"- Avg game length: {summary['avg_game_length']:.1f} moves",
                "",
                f"![Cumulative eval graph]({cumulative_image})",
                "",
                f"![All-game eval trajectories]({eval_overlay_image})",
                "",
                "| Game | Result | Outcome | Moves | Termination |",
                "|------|--------|---------|-------|-------------|",
            ]
        )

        for game_number, game in enumerate(games, start=1):
            lines.append(
                f"| {game_number} | {game['result']} | {game['outcome']} | "
                f"{game['total_moves']} | {game['termination']} |"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def render_artifacts_from_store(
    model_name: str,
    levels_data: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate PGN, image, PDF, and markdown artifacts for one model."""
    pgn_dir = output_dir / "pgn" / "by_level"
    figures_dir = output_dir / "figures" / "levels"
    pdf_dir = output_dir / "pdf"

    pgn_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    for level_name, payload in levels_data.items():
        elo = payload["elo"]
        games = payload["games"]
        summary = payload["summary"]
        slug = f"{model_name}_{elo}"

        pgn_path = pgn_dir / f"{slug}.pgn"
        pgn_text = "\n\n".join(game["pgn"] for game in games)
        pgn_path.write_text(pgn_text + ("\n" if pgn_text else ""), encoding="utf-8")

        cumulative_image_path = figures_dir / f"{slug}_cumulative.png"
        eval_overlay_image_path = figures_dir / f"{slug}_evals.png"

        _plot_level_cumulative_graph(level_name, elo, games, cumulative_image_path)
        _plot_level_eval_overlay_graph(level_name, elo, games, eval_overlay_image_path)

        model_label = MODEL_LABELS.get(model_name, model_name)
        title = f"{model_label} vs Stockfish — {level_name} ({elo})"
        summary_lines = [
            f"Games: {len(games)}",
            f"W/D/L: {summary['wins']}/{summary['draws']}/{summary['losses']}",
            f"Score: {summary['score_pct']:.1f}%",
            f"Avg ACPL: {summary['avg_acpl']:.1f}",
            f"Avg game length: {summary['avg_game_length']:.1f} moves",
        ]

        pdf_path = pdf_dir / f"{slug}.pdf"
        _build_level_pdf(
            pdf_path,
            title,
            summary_lines,
            cumulative_image_path,
            eval_overlay_image_path,
        )

    _write_cumulative_markdown(
        model_name=model_name,
        levels_data=levels_data,
        report_path=output_dir / "cumulative_report.md",
    )
