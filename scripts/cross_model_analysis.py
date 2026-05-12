import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODEL_COLORS = {
    "convnet": "#1f77b4",
    "resnet": "#ff7f0e",
    "square_transformer": "#2ca02c",
    "piece_transformer": "#d62728",
    "gcn": "#9467bd",
    "gat": "#8c564b"
}

RUN_SIZES = ["10M", "100M", "200M", "500M", "1000M"]
OUTCOME_COLORS = {"win": "#2ecc71", "loss": "#e74c3c", "draw": "#95a5a6"}
OUTCOME_LABELS = {"win": "Model wins", "loss": "Model loses", "draw": "Draw"}

EVAL_CAP = 10.0
SMOOTH_WINDOW = 5

def smooth_series(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed_values = np.convolve(values, kernel, mode="same")
    edge_width = window // 2
    smoothed_values[:edge_width] = values[:edge_width]
    smoothed_values[-edge_width:] = values[-edge_width:]
    return smoothed_values

def _series_model_perspective(game: dict[str, Any]) -> np.ndarray:
    evaluations = np.array(game["evaluations"], dtype=float)
    if game.get("termination") == "checkmate":
        result = game.get("result")
        if result == "1-0":
            evaluations[-1] = 10000.0
        elif result == "0-1":
            evaluations[-1] = -10000.0
            
    evaluations = evaluations / 100.0
    if game["model_color"] == "white":
        return np.clip(evaluations, -EVAL_CAP, EVAL_CAP)
    return np.clip(-evaluations, -EVAL_CAP, EVAL_CAP)

def get_run_size(run_name: str) -> str:
    for size in RUN_SIZES:
        if size in run_name:
            return size
    return run_name

# --- ELO ESTIMATION ---
def logistic_function(x: np.ndarray, k: float, x0: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

def score_from_summary(summary: dict[str, Any], games: list[dict[str, Any]]) -> float:
    if "score_pct" in summary:
        return float(summary["score_pct"]) / 100.0
    wins = int(summary.get("wins", 0))
    draws = int(summary.get("draws", 0))
    total = len(games)
    if total == 0:
        return 0.0
    return (wins + 0.5 * draws) / total

def extract_points(levels: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
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
    if len(x_data) < 2:
        return 800.0, None, None
    if np.max(y_data) == 0.0:
        return 800.0, None, None
    if np.min(y_data) == 1.0:
        return 3000.0, None, None
    try:
        params, _ = optimize.curve_fit(
            logistic_function,
            x_data,
            y_data,
            p0=[-0.01, 1500.0],
            maxfev=10000
        )
        k_val = float(params[0])
        estimated_elo = float(params[1])
        return estimated_elo, k_val, params
    except RuntimeError:
        return 800.0, None, None

# --- DATA LOADING ---
def load_all_runs(benchmarks_dir: Path) -> dict:
    all_data = {}
    for model_dir in benchmarks_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name in ["figures", "pdf", "evals"]:
            continue
        model_name = model_dir.name
        all_data[model_name] = {}
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
            json_file = run_dir / "benchmark_results.json"
            if not json_file.exists():
                continue
            with open(json_file, "r") as f:
                data = json.load(f)
            
            models_data = data.get("models", {})
            if model_name in models_data:
                levels = models_data[model_name].get("levels", {})
                
                x_data, y_data = extract_points(levels)
                est_elo, k_val, params = fit_elo(x_data, y_data)
                
                if est_elo > 3500: est_elo = 3500.0
                if est_elo < 0: est_elo = 800.0
                
                all_data[model_name][run_dir.name] = {
                    "levels": levels,
                    "elo_estimate": est_elo,
                    "elo_k": k_val,
                    "elo_params": params,
                    "x_data": x_data,
                    "y_data": y_data,
                    "run_name": run_dir.name,
                    "data_size": get_run_size(run_dir.name)
                }
    return all_data

# --- PLOTS ---

def sort_runs(runs: list[str]) -> list[str]:
    def rank(r):
        s = get_run_size(r)
        if "10M" in s and "100M" not in s: return 1
        if "100M" in s: return 2
        if "200M" in s: return 3
        if "500M" in s: return 4
        if "1000M" in s: return 5
        return 99
    return sorted(runs, key=lambda r: (rank(r), r))

def plot_acpl_boxplot(all_data: dict, output_dir: Path):
    cmap = plt.get_cmap("viridis")
    for model, runs_data in all_data.items():
        runs = sort_runs(list(runs_data.keys()))
        if not runs: continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        elo_levels = set()
        for run in runs:
            for lvl_data in runs_data[run]["levels"].values():
                elo_levels.add(lvl_data["elo"])
        elo_levels = sorted(list(elo_levels))
        
        positions = []
        box_data = []
        colors = []
        run_colors = {run: cmap(i / max(1, len(runs)-1)) for i, run in enumerate(runs)}
        
        pos = 1
        x_ticks = []
        x_tick_labels = []
        
        for elo in elo_levels:
            x_ticks.append(pos + (len(runs)-1)/2)
            x_tick_labels.append(str(elo))
            for run in runs:
                acpls = []
                for lvl_data in runs_data[run]["levels"].values():
                    if lvl_data["elo"] == elo:
                        if "acpl_list" in lvl_data.get("summary", {}):
                            acpls.extend(lvl_data["summary"]["acpl_list"])
                box_data.append(acpls if acpls else [0])
                positions.append(pos)
                colors.append(run_colors[run])
                pos += 1
            pos += 1 
            
        bplot = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.8, showfliers=False)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_title(f"ACPL Distribution - {model}")
        ax.set_xlabel("Opponent Elo")
        ax.set_ylabel("Centipawn Loss")
        
        handles = [plt.Rectangle((0,0),1,1, color=run_colors[r], alpha=0.7) for r in runs]
        labels = [get_run_size(r) for r in runs]
        ax.legend(handles, labels, title="Data Size")
        
        fig.tight_layout()
        plt.savefig(output_dir / f"acpl_boxplot_{model}.png", dpi=150)
        plt.close()

def plot_score_heatmaps_by_size(all_data: dict, output_dir: Path):
    models = sorted(list(all_data.keys()))
    elo_set = set()
    for model in models:
        for run_data in all_data[model].values():
            for lvl in run_data["levels"].values():
                elo_set.add(lvl["elo"])
    elo_levels = sorted(list(elo_set))
    
    for size in RUN_SIZES:
        has_data = False
        for model in models:
            for run, run_data in all_data[model].items():
                if run_data["data_size"] == size:
                    has_data = True
        if not has_data: continue
        
        heatmap_data = np.zeros((len(models), len(elo_levels)))
        for i, model in enumerate(models):
            target_run = None
            for run, run_data in all_data[model].items():
                if run_data["data_size"] == size:
                    target_run = run
                    break
                    
            for j, elo in enumerate(elo_levels):
                score = 0
                if target_run:
                    for lvl_data in all_data[model][target_run]["levels"].values():
                        if lvl_data["elo"] == elo:
                            score = lvl_data.get("summary", {}).get("score_pct", 0)
                heatmap_data[i, j] = score
                
        fig, ax = plt.subplots(figsize=(10, len(models)*0.6 + 2))
        im = ax.imshow(heatmap_data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        
        ax.set_xticks(np.arange(len(elo_levels)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(elo_levels)
        ax.set_yticklabels(models)
        
        for i in range(len(models)):
            for j in range(len(elo_levels)):
                val = heatmap_data[i, j]
                color = "black" if 30 < val < 70 else "white"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontweight="bold")
                
        ax.set_title(f"Score % Heatmap by Model - {size} Data Size")
        fig.colorbar(im, ax=ax, label="Score %")
        fig.tight_layout()
        plt.savefig(output_dir / f"score_heatmap_{size}.png", dpi=150)
        plt.close()

def plot_score_heatmaps_by_model(all_data: dict, output_dir: Path):
    for model, runs_data in all_data.items():
        runs = sort_runs(list(runs_data.keys()))
        if not runs: continue
        
        elo_set = set()
        for run in runs:
            for lvl in runs_data[run]["levels"].values():
                elo_set.add(lvl["elo"])
        elo_levels = sorted(list(elo_set))
        
        sizes_present = [get_run_size(r) for r in runs]
        
        heatmap_data = np.zeros((len(runs), len(elo_levels)))
        for i, run in enumerate(runs):
            for j, elo in enumerate(elo_levels):
                score = 0
                for lvl_data in runs_data[run]["levels"].values():
                    if lvl_data["elo"] == elo:
                        score = lvl_data.get("summary", {}).get("score_pct", 0)
                heatmap_data[i, j] = score
                
        fig, ax = plt.subplots(figsize=(10, len(runs)*0.6 + 2))
        im = ax.imshow(heatmap_data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        
        ax.set_xticks(np.arange(len(elo_levels)))
        ax.set_yticks(np.arange(len(runs)))
        ax.set_xticklabels(elo_levels)
        ax.set_yticklabels(sizes_present)
        
        for i in range(len(runs)):
            for j in range(len(elo_levels)):
                val = heatmap_data[i, j]
                color = "black" if 30 < val < 70 else "white"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontweight="bold")
                
        ax.set_title(f"Score % Heatmap by Data Size - {model}")
        ax.set_ylabel("Data Size")
        ax.set_xlabel("Opponent Elo")
        fig.colorbar(im, ax=ax, label="Score %")
        fig.tight_layout()
        plt.savefig(output_dir / f"model_size_heatmap_{model}.png", dpi=150)
        plt.close()

def plot_wdl_stacked(all_data: dict, output_dir: Path):
    for model, runs_data in all_data.items():
        runs = sort_runs(list(runs_data.keys()))
        if not runs: continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        elo_levels = set()
        for run in runs:
            for lvl_data in runs_data[run]["levels"].values():
                elo_levels.add(lvl_data["elo"])
        elo_levels = sorted(list(elo_levels))
        
        pos = 1
        x_ticks = []
        x_tick_labels = []
        
        for elo in elo_levels:
            x_ticks.append(pos + (len(runs)-1)/2)
            x_tick_labels.append(str(elo))
            for run in runs:
                w, d, l = 0, 0, 0
                for lvl_data in runs_data[run]["levels"].values():
                    if lvl_data["elo"] == elo:
                        summary = lvl_data.get("summary", {})
                        w = summary.get("wins", 0)
                        d = summary.get("draws", 0)
                        l = summary.get("losses", 0)
                tot = w + d + l
                if tot > 0:
                    wp, dp, lp = w/tot*100, d/tot*100, l/tot*100
                else:
                    wp, dp, lp = 0, 0, 0
                    
                ax.bar(pos, wp, color="#2ecc71", width=0.8)
                ax.bar(pos, dp, bottom=wp, color="#95a5a6", width=0.8)
                ax.bar(pos, lp, bottom=wp+dp, color="#e74c3c", width=0.8)
                
                # Move the text lower to prevent overlap
                ax.text(pos, -5, get_run_size(run), ha='center', va='top', rotation=90, fontsize=8)
                
                pos += 1
            pos += 1
            
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_title(f"Win / Draw / Loss Breakdown - {model}")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        ax.set_xlabel("Opponent Elo")
        
        handles = [
            plt.Rectangle((0,0),1,1, color="#2ecc71"),
            plt.Rectangle((0,0),1,1, color="#95a5a6"),
            plt.Rectangle((0,0),1,1, color="#e74c3c")
        ]
        ax.legend(handles, ["Win", "Draw", "Loss"], loc='upper right')
        
        # Adjust bottom to accommodate the rotated data size text + label cleanly
        plt.subplots_adjust(bottom=0.25)
        plt.savefig(output_dir / f"result_stacked_{model}.png", dpi=150)
        plt.close()

def plot_elo_logistic_fits(all_data: dict, output_dir: Path):
    cmap = plt.get_cmap("viridis")
    for model, runs_data in all_data.items():
        runs = sort_runs(list(runs_data.keys()))
        if not runs: continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        run_colors = {run: cmap(i / max(1, len(runs)-1)) for i, run in enumerate(runs)}
        
        for run in runs:
            run_data = runs_data[run]
            x_data = run_data["x_data"]
            y_data = run_data["y_data"]
            params = run_data["elo_params"]
            
            if len(x_data) > 0:
                color = run_colors[run]
                size_label = get_run_size(run)
                
                ax.scatter(x_data, y_data, color=color, alpha=0.8, marker="o")
                
                if params is not None:
                    x_smooth = np.linspace(max(0, float(np.min(x_data))-200), float(np.max(x_data))+200, 200)
                    y_smooth = logistic_function(x_smooth, *params)
                    ax.plot(x_smooth, y_smooth, color=color, alpha=0.8, label=size_label)
                else:
                    ax.plot(x_data, y_data, color=color, alpha=0.8, linestyle="--", label=f"{size_label} (no fit)")
                    
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"Logistic Curve Fits - {model}")
        ax.set_xlabel("Stockfish Elo")
        ax.set_ylabel("Score Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(title="Data Size")
            
        fig.tight_layout()
        plt.savefig(output_dir / f"elo_logistic_fits_{model}.png", dpi=150)
        plt.close()

def plot_estimated_elo_bar(all_data: dict, output_dir: Path):
    models = sorted(list(all_data.keys()))
    runs_sizes = RUN_SIZES
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.15
    index = np.arange(len(models))
    
    cmap = plt.get_cmap("viridis")
    size_colors = {size: cmap(i / (len(runs_sizes)-1)) for i, size in enumerate(runs_sizes)}
    
    max_elo = 0
    for i, size in enumerate(runs_sizes):
        elos = []
        for model in models:
            target_run = None
            for run, run_data in all_data[model].items():
                if run_data["data_size"] == size:
                    target_run = run
                    break
            if target_run:
                elo = all_data[model][target_run]["elo_estimate"]
            else:
                elo = 0
            elos.append(elo)
            if elo > max_elo: max_elo = elo
            
        x_pos = index + (i - len(runs_sizes)/2) * bar_width + bar_width/2
        bars = ax.bar(x_pos, elos, bar_width, label=size, color=size_colors[size], alpha=0.8)
        
        for bar, elo in zip(bars, elos):
            if elo > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f"{elo:.0f}", 
                        ha='center', va='bottom', rotation=90, fontsize=8)
                
    ax.set_title("Estimated Elo (Logistic Curve Fit 50% Point)")
    ax.set_ylabel("Estimated Elo")
    ax.set_xticks(index)
    ax.set_xticklabels(models)
    
    ax.set_ylim(0, max_elo + 500)
    ax.legend(title="Data Size")
    
    fig.tight_layout()
    plt.savefig(output_dir / "estimated_elo_bar.png", dpi=150)
    plt.close()

def plot_score_line(all_data: dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in sorted(all_data.keys()):
        color = MODEL_COLORS.get(model, "#000000")
        runs = sort_runs(list(all_data[model].keys()))
        for i, run in enumerate(runs):
            x_data = all_data[model][run]["x_data"]
            y_data = all_data[model][run]["y_data"] * 100
            if len(x_data) > 0:
                alpha = 0.4 + 0.6 * (i / max(1, len(runs)-1))
                ax.plot(x_data, y_data, marker="o", color=color, alpha=alpha, label=f"{model}-{get_run_size(run)}")
    ax.set_title("Score % vs Opponent Elo")
    ax.set_xlabel("Opponent Elo")
    ax.set_ylabel("Score %")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    fig.tight_layout()
    plt.savefig(output_dir / "score_line.png", dpi=150)
    plt.close()

def plot_acpl_line(all_data: dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in sorted(all_data.keys()):
        color = MODEL_COLORS.get(model, "#000000")
        runs = sort_runs(list(all_data[model].keys()))
        for i, run in enumerate(runs):
            x_data = []
            y_data = []
            levels = all_data[model][run]["levels"]
            for lvl_data in levels.values():
                x_data.append(lvl_data["elo"])
                y_data.append(lvl_data.get("summary", {}).get("avg_acpl", 0))
            pts = sorted(zip(x_data, y_data))
            if pts:
                xs, ys = zip(*pts)
                alpha = 0.4 + 0.6 * (i / max(1, len(runs)-1))
                ax.plot(xs, ys, marker="o", color=color, alpha=alpha, label=f"{model}-{get_run_size(run)}")
    ax.set_title("Average Centipawn Loss vs Opponent Elo")
    ax.set_xlabel("Opponent Elo")
    ax.set_ylabel("ACPL")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    fig.tight_layout()
    plt.savefig(output_dir / "acpl_line.png", dpi=150)
    plt.close()

def generate_eval_graphs(all_data: dict, evals_dir: Path):
    indiv_dir = evals_dir / "individual"
    indiv_dir.mkdir(exist_ok=True)
    
    for model, runs_data in all_data.items():
        for run, run_data in runs_data.items():
            for lvl_name, lvl_data in run_data["levels"].items():
                elo = lvl_data["elo"]
                games = lvl_data.get("games", [])
                
                # 1. Cumulative (Overlay)
                fig, ax = plt.subplots(figsize=(11, 5))
                traces_by_outcome: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
                    "win": [], "loss": [], "draw": [],
                }
                
                for game_idx, game in enumerate(games):
                    series = _series_model_perspective(game)
                    if len(series) == 0:
                        continue
                    move_numbers = np.arange(1, len(series) + 1) / 2.0
                    traces_by_outcome[game["outcome"]].append((move_numbers, smooth_series(series)))
                    
                    # 2. Individual Game graph
                    fig_indiv, ax_indiv = plt.subplots(figsize=(8, 4))
                    ax_indiv.plot(move_numbers, series, color=OUTCOME_COLORS[game["outcome"]], linewidth=1.5)
                    ax_indiv.axhline(0, color="#333333", linewidth=0.8, alpha=0.5, linestyle="--")
                    ax_indiv.set_ylim(-EVAL_CAP - 0.5, EVAL_CAP + 0.5)
                    ax_indiv.set_title(f"Game {game_idx+1}: {model} ({get_run_size(run)}) vs {lvl_name} ({game['result']})")
                    ax_indiv.set_xlabel("Move Number")
                    ax_indiv.set_ylabel("Eval (model perspective)")
                    ax_indiv.grid(True, alpha=0.25)
                    fig_indiv.tight_layout()
                    
                    indiv_path = indiv_dir / f"eval_indiv_{model}_{get_run_size(run)}_{elo}_g{game_idx+1}.png"
                    fig_indiv.savefig(indiv_path, dpi=100, bbox_inches="tight")
                    plt.close(fig_indiv)
                    
                # Finish Cumulative graph
                for outcome in ("win", "loss", "draw"):
                    traces = traces_by_outcome[outcome]
                    if not traces:
                        continue
                    color = OUTCOME_COLORS[outcome]
                    for move_numbers, series in traces:
                        ax.plot(move_numbers, series, color=color, alpha=0.18, linewidth=1.0)
                        
                    max_length = max(len(series) for _, series in traces)
                    padded = np.full((len(traces), max_length), np.nan)
                    for index, (_, series) in enumerate(traces):
                        padded[index, : len(series)] = series
                    mean_series = np.nanmean(padded, axis=0)
                    valid_points = ~np.isnan(mean_series)
                    mean_x = np.arange(1, len(mean_series) + 1) / 2.0
                    ax.plot(
                        mean_x[valid_points],
                        mean_series[valid_points],
                        color=color,
                        linewidth=2.6,
                        label=f"{OUTCOME_LABELS[outcome]} (avg)",
                    )
                    
                ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.5, linestyle="--")
                ax.set_ylim(-EVAL_CAP - 0.5, EVAL_CAP + 0.5)
                ax.set_title(
                    f"Cumulative Eval Trajectories — {model} ({get_run_size(run)}) vs {lvl_name}",
                    fontsize=13, fontweight="bold",
                )
                ax.set_xlabel("Move Number", fontsize=11)
                ax.set_ylabel("Eval (model perspective, pawns)", fontsize=11)
                ax.grid(True, alpha=0.25)
                
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(loc="best", fontsize=9)
                    
                fig.tight_layout()
                fig_path = evals_dir / f"eval_cumulative_{model}_{get_run_size(run)}_{elo}.png"
                fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
                plt.close(fig)

def generate_markdown_report(all_data: dict, figures_dir: Path, output_path: Path):
    lines = [
        "# Cross-Model Benchmark & Elo Estimation Report\n",
        "## Summary of Estimated Elos\n",
        "| Model | Run | Estimated Elo | Logistic `k` |",
        "|---|---|---|---|"
    ]
    for model in sorted(all_data.keys()):
        for run in sort_runs(list(all_data[model].keys())):
            elo = all_data[model][run]["elo_estimate"]
            k_val = all_data[model][run]["elo_k"]
            k_str = f"{k_val:.4f}" if k_val is not None else "N/A"
            lines.append(f"| {model} | {get_run_size(run)} | **{elo:.0f}** | {k_str} |")
            
    lines.extend([
        "\n## Visualizations\n",
        "### Estimated Elo Bar Chart\n![Estimated Elo](figures/estimated_elo_bar.png)\n",
        "### Score % Line Graph\n![Score %](figures/score_line.png)\n",
        "### ACPL Line Graph\n![ACPL Line Graph](figures/acpl_line.png)\n",
    ])
    
    lines.append("\n### Score % Heatmaps (Comparing Models by Data Size)\n")
    for size in RUN_SIZES:
        if (figures_dir / f"score_heatmap_{size}.png").exists():
            lines.append(f"#### {size}\n![Score % Heatmap {size}](figures/score_heatmap_{size}.png)\n")
            
    lines.append("\n### Per-Model Visualizations\n")
    for model in sorted(all_data.keys()):
        lines.extend([
            f"#### {model}\n",
            f"**Score % by Data Size (Heatmap)**\n![Model Heatmap {model}](figures/model_size_heatmap_{model}.png)\n",
            f"**ACPL Distribution**\n![ACPL Boxplot {model}](figures/acpl_boxplot_{model}.png)\n",
            f"**Win/Draw/Loss Breakdown**\n![WDL Stacked {model}](figures/result_stacked_{model}.png)\n",
            f"**Elo Logistic Fits**\n![Logistic Fits {model}](figures/elo_logistic_fits_{model}.png)\n"
        ])
        
    lines.append("\n## Cumulative Eval Trajectory Graphs\n")
    lines.append("*(Note: Individual game trajectory plots are saved in `figures/evals/individual/`)*\n")
    for model in sorted(all_data.keys()):
        lines.append(f"### {model}\n")
        for run in sort_runs(list(all_data[model].keys())):
            size = get_run_size(run)
            lines.append(f"#### {size}\n")
            elos = sorted(list(set([lvl["elo"] for lvl in all_data[model][run]["levels"].values()])))
            for elo in elos:
                lines.append(f"![Eval {model} {size} {elo}](figures/evals/eval_cumulative_{model}_{size}_{elo}.png)\n")
                
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

def generate_pdf_report(all_data: dict, figures_dir: Path, output_path: Path):
    page_width, page_height = A4
    margin = 18 * mm
    usable_width = page_width - 2 * margin

    pdf_canvas = canvas.Canvas(str(output_path), pagesize=A4)
    pdf_canvas.setFont("Helvetica-Bold", 18)
    
    y_position = page_height - margin - 10
    pdf_canvas.drawString(margin, y_position, "Cross-Model Benchmark & Elo Estimation Report")
    
    y_position -= 25
    pdf_canvas.setFont("Helvetica-Bold", 14)
    pdf_canvas.drawString(margin, y_position, "Summary of Estimated Elos")
    
    y_position -= 20
    pdf_canvas.setFont("Helvetica-Bold", 11)
    pdf_canvas.drawString(margin, y_position, "Model")
    pdf_canvas.drawString(margin + 100, y_position, "Run")
    pdf_canvas.drawString(margin + 250, y_position, "Estimated Elo")
    pdf_canvas.drawString(margin + 350, y_position, "Logistic 'k'")
    
    y_position -= 15
    pdf_canvas.setFont("Helvetica", 11)
    
    for model in sorted(all_data.keys()):
        for run in sort_runs(list(all_data[model].keys())):
            elo = all_data[model][run]["elo_estimate"]
            k_val = all_data[model][run]["elo_k"]
            k_str = f"{k_val:.4f}" if k_val is not None else "N/A"
            
            pdf_canvas.drawString(margin, y_position, model)
            pdf_canvas.drawString(margin + 100, y_position, get_run_size(run))
            pdf_canvas.drawString(margin + 250, y_position, f"{elo:.0f}")
            pdf_canvas.drawString(margin + 350, y_position, k_str)
            
            y_position -= 15
            if y_position < margin:
                pdf_canvas.showPage()
                y_position = page_height - margin
                pdf_canvas.setFont("Helvetica", 11)

    y_position -= 10
    
    # 1. Global Figures
    figures = [
        ("Estimated Elo", "estimated_elo_bar.png", 0.4),
        ("Score % Line Graph", "score_line.png", 0.5),
        ("ACPL Line Graph", "acpl_line.png", 0.5),
    ]
    for size in RUN_SIZES:
        if (figures_dir / f"score_heatmap_{size}.png").exists():
            figures.append((f"Score % Heatmap - {size}", f"score_heatmap_{size}.png", 0.5))

    for title, filename, aspect_ratio in figures:
        img_path = figures_dir / filename
        if not img_path.exists(): continue
        img_height = usable_width * aspect_ratio
        if y_position - img_height - 30 < margin:
            pdf_canvas.showPage()
            y_position = page_height - margin
        pdf_canvas.setFont("Helvetica-Bold", 14)
        pdf_canvas.drawString(margin, y_position, title)
        y_position -= 15
        pdf_canvas.drawImage(str(img_path), margin, y_position - img_height,
                             width=usable_width, height=img_height, preserveAspectRatio=True, anchor="nw")
        y_position -= img_height + 25

    # 2. Per-Model Figures
    for model in sorted(all_data.keys()):
        model_figs = [
            (f"Score % Heatmap by Data Size - {model}", f"model_size_heatmap_{model}.png", 0.4),
            (f"ACPL Distribution - {model}", f"acpl_boxplot_{model}.png", 0.4),
            (f"Win/Draw/Loss Breakdown - {model}", f"result_stacked_{model}.png", 0.4),
            (f"Logistic Curve Fits - {model}", f"elo_logistic_fits_{model}.png", 0.4)
        ]
        for title, filename, aspect_ratio in model_figs:
            img_path = figures_dir / filename
            if not img_path.exists(): continue
            img_height = usable_width * aspect_ratio
            if y_position - img_height - 30 < margin:
                pdf_canvas.showPage()
                y_position = page_height - margin
            pdf_canvas.setFont("Helvetica-Bold", 14)
            pdf_canvas.drawString(margin, y_position, title)
            y_position -= 15
            pdf_canvas.drawImage(str(img_path), margin, y_position - img_height,
                                 width=usable_width, height=img_height, preserveAspectRatio=True, anchor="nw")
            y_position -= img_height + 25

    # 3. Eval Trajectories (Cumulative Only)
    pdf_canvas.showPage()
    y_position = page_height - margin
    pdf_canvas.setFont("Helvetica-Bold", 16)
    pdf_canvas.drawString(margin, y_position, "Cumulative Evaluation Trajectories")
    y_position -= 25

    for model in sorted(all_data.keys()):
        for run in sort_runs(list(all_data[model].keys())):
            size = get_run_size(run)
            elos = sorted(list({lvl["elo"] for lvl in all_data[model][run]["levels"].values()}))
            for elo in elos:
                img_path = figures_dir / "evals" / f"eval_cumulative_{model}_{size}_{elo}.png"
                if not img_path.exists(): continue
                img_height = usable_width * 0.4
                if y_position - img_height - 30 < margin:
                    pdf_canvas.showPage()
                    y_position = page_height - margin
                title = f"{model} ({size}) vs Elo {elo}"
                pdf_canvas.setFont("Helvetica-Bold", 12)
                pdf_canvas.drawString(margin, y_position, title)
                y_position -= 15
                pdf_canvas.drawImage(str(img_path), margin, y_position - img_height,
                                     width=usable_width, height=img_height, preserveAspectRatio=True, anchor="nw")
                y_position -= img_height + 25

    pdf_canvas.save()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks-dir", type=str, default="benchmarks")
    parser.add_argument("--output-dir", type=str, default="benchmarks")
    args = parser.parse_args()
    
    benchmarks_dir = Path(args.benchmarks_dir)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    evals_dir = figures_dir / "evals"
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("Loading all runs from %s", benchmarks_dir)
    all_data = load_all_runs(benchmarks_dir)
    if not all_data:
        logging.error("No benchmark data found.")
        return
        
    logging.info("Generating ACPL boxplots per model...")
    plot_acpl_boxplot(all_data, figures_dir)
    logging.info("Generating Score Heatmaps by size...")
    plot_score_heatmaps_by_size(all_data, figures_dir)
    logging.info("Generating Score Heatmaps by model...")
    plot_score_heatmaps_by_model(all_data, figures_dir)
    logging.info("Generating WDL stacked bars per model...")
    plot_wdl_stacked(all_data, figures_dir)
    logging.info("Generating Score line graph...")
    plot_score_line(all_data, figures_dir)
    logging.info("Generating ACPL line graph...")
    plot_acpl_line(all_data, figures_dir)
    logging.info("Generating Elo estimated bar chart...")
    plot_estimated_elo_bar(all_data, figures_dir)
    logging.info("Generating Elo logistic fits per model...")
    plot_elo_logistic_fits(all_data, figures_dir)
    logging.info("Generating Eval graphs with checkmate fix...")
    generate_eval_graphs(all_data, evals_dir)
    
    logging.info("Generating Markdown report...")
    generate_markdown_report(all_data, figures_dir, output_dir / "cross_model_report.md")
    logging.info("Generating PDF report...")
    generate_pdf_report(all_data, figures_dir, output_dir / "cross_model_report.pdf")
    logging.info("Done! Visualizations saved to %s", figures_dir)

if __name__ == "__main__":
    main()
