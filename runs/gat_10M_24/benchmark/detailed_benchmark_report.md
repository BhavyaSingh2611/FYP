# Detailed Chess Model Benchmark

**Date:** 2026-02-14 15:51
**Opponent:** Stockfish depth 24
**Evaluation Depth:** 18

## Results Summary

| Model | W | D | L | Score |
|-------|---|---|---|-------|
| gat | 0 | 0 | 4 | 0.0/4 |

## Move Timing — Model

| Model | Avg (s) | Median (s) | Min (s) | Max (s) | Moves |
|-------|---------|------------|---------|---------|-------|
| gat | 0.072 | 0.074 | 0.037 | 0.352 | 86 |

## Move Timing — Opponent (Stockfish)

| Model | Avg (s) | Median (s) | Min (s) | Max (s) | Moves |
|-------|---------|------------|---------|---------|-------|
| gat | 3.053 | 2.866 | 0.001 | 13.156 | 88 |

## Game Files

- **Combined PGN:** [all_games.pgn](all_games.pgn) - Open in Lichess, Chess.com, or any chess software
- **Individual PGNs:** `pgn/` directory

## Evaluation Charts

![All Games Comparison](figures/all_games_comparison.png)

### Individual Game Charts

#### gat

![Game 1](figures/gat_game_1_eval.png)

![Game 2](figures/gat_game_2_eval.png)

![Game 3](figures/gat_game_3_eval.png)

![Game 4](figures/gat_game_4_eval.png)

## How to View Games

1. **Lichess:** Go to lichess.org/paste and paste the PGN content
2. **Chess.com:** Use chess.com/analysis and import PGN
3. **Desktop Apps:** Open with ChessBase, Arena, SCID, or Lucas Chess
4. **Command Line:** Use `python-chess` to parse the PGN files
