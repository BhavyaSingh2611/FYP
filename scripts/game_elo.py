"""
Estimate model Elo from benchmark game results using logistic curve fitting.

This mirrors the methodology in benchmark_elo.py:
- collect empirical win rates across Elo buckets
- fit a logistic curve
- report the 50% score point as estimated Elo
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def logistic_function(x: np.ndarray, k: float, x0: float) -> np.ndarray:
    """Logistic function for curve fitting."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def load_results(path: Path) -> dict[str, Any]:
    """Load benchmark JSON produced by scripts/benchmark.py."""
    if not path.exists():
        raise FileNotFoundError(f"Benchmark JSON not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def select_model(data: dict[str, Any], model_name: str | None) -> tuple[str, dict[str, Any]]:
    """Select model payload from benchmark JSON."""
    models = data.get("models", {})
    if not models:
        raise ValueError("No models found in benchmark JSON.")

    if model_name:
        if model_name not in models:
            available = ", ".join(sorted(models.keys()))
            raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
        return model_name, models[model_name]

    if len(models) == 1:
        chosen = next(iter(models))
        return chosen, models[chosen]

    available = ", ".join(sorted(models.keys()))
    raise ValueError(f"Multiple models found. Please pass --model. Available models: {available}")


def score_from_summary(summary: dict[str, Any], games: list[dict[str, Any]]) -> float:
    """Get score rate in [0, 1]. Prefer score_pct when present."""
    if "score_pct" in summary:
        return float(summary["score_pct"]) / 100.0

    wins = int(summary.get("wins", 0))
    draws = int(summary.get("draws", 0))
    total = len(games)
    if total == 0:
        return 0.0
    return (wins + 0.5 * draws) / total


def extract_points(levels: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Extract (elo, score_rate) points sorted by Elo."""
    points: list[tuple[float, float]] = []

    for _, level_data in levels.items():
        elo = float(level_data["elo"])
        summary = level_data.get("summary", {})
        games = level_data.get("games", [])
        score_rate = score_from_summary(summary, games)
        points.append((elo, score_rate))

    if not points:
        return np.array([]), np.array([])

    points.sort(key=lambda item: item[0])
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    return x, y


def fit_elo(x_data: np.ndarray, y_data: np.ndarray) -> tuple[float, float | None, np.ndarray | None]:
    """Fit logistic curve and return estimated Elo (50% point), k, and params."""
    if len(x_data) < 2:
        raise ValueError("Not enough data points to fit a curve.")

    if np.max(y_data) == 0.0:
        logging.error("Score rate is 0%% for all levels. Estimated Elo is likely below the tested range.")
        return 0.0, None, None

    if np.min(y_data) == 1.0:
        logging.error("Score rate is 100%% for all levels. Estimated Elo is likely above the tested range.")
        return 3000.0, None, None

    try:
        params, _ = optimize.curve_fit(
            logistic_function,
            x_data,
            y_data,
            p0=[-0.01, 1500.0],
        )
        k_val = float(params[0])
        estimated_elo = float(params[1])
        return estimated_elo, k_val, params
    except RuntimeError:
        logging.error("Failed to fit logistic curve. Returning fallback estimate.")
        return 1500.0, None, None


def save_plot(
    x_data: np.ndarray,
    y_data: np.ndarray,
    params: np.ndarray | None,
    estimated_elo: float,
    model_name: str,
    output_path: Path,
) -> None:
    """Create Elo fit visualization."""
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color="blue", label="Empirical Score Rate")

    if params is not None:
        x_smooth = np.linspace(float(np.min(x_data)), float(np.max(x_data)), 200)
        y_smooth = logistic_function(x_smooth, *params)
        plt.plot(x_smooth, y_smooth, color="red", label="Fitted Logistic Curve")
        plt.axvline(x=estimated_elo, color="green", linestyle="--", label=f"Estimated Elo: {estimated_elo:.0f}")
        plt.axhline(y=0.5, color="gray", linestyle=":")

    plt.title(f"Game Benchmark Elo Estimate ({model_name})")
    plt.xlabel("Stockfish Elo Level")
    plt.ylabel("Score Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    logging.info("Saved evaluation plot to %s", output_path)


def save_summary(
    output_path: Path,
    model_name: str,
    estimated_elo: float,
    k_val: float | None,
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> None:
    """Save Elo estimate summary as JSON."""
    summary = {
        "model": model_name,
        "estimated_elo": estimated_elo,
        "k": k_val,
        "x_elo": x_data.tolist(),
        "y_score_rate": y_data.tolist(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    logging.info("Saved Elo summary to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate Elo from benchmark.py JSON using logistic curve fitting.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="runs/evaluation/benchmark_results.json",
        help="Path to benchmark.py JSON output.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name under JSON 'models'. Required only if multiple models exist.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="runs/game_elo_evaluation.png",
        help="Path to output Elo plot image.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="runs/game_elo_summary.json",
        help="Path to output Elo summary JSON.",
    )
    args = parser.parse_args()

    data = load_results(Path(args.input))
    model_name, model_payload = select_model(data, args.model)

    levels = model_payload.get("levels", {})
    x_data, y_data = extract_points(levels)

    if len(x_data) < 2:
        raise ValueError("Need at least 2 Elo levels from benchmark results to estimate Elo.")

    for elo, score in zip(x_data, y_data, strict=False):
        logging.info("Level %.0f: Score Rate %.1f%%", elo, score * 100.0)

    estimated_elo, k_val, params = fit_elo(x_data, y_data)

    logging.info("=== Final Result ===")
    logging.info("Estimated 50%% game Elo: %.1f", estimated_elo)
    if k_val is not None:
        logging.info("Logistic falloff param k: %.4f", k_val)

    save_plot(
        x_data=x_data,
        y_data=y_data,
        params=params,
        estimated_elo=estimated_elo,
        model_name=model_name,
        output_path=Path(args.plot_path),
    )
    save_summary(
        output_path=Path(args.summary_path),
        model_name=model_name,
        estimated_elo=estimated_elo,
        k_val=k_val,
        x_data=x_data,
        y_data=y_data,
    )


if __name__ == "__main__":
    main()
