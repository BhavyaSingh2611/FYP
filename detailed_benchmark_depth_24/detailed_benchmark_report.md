# Detailed Chess Model Benchmark

**Date:** 2026-01-27 01:39
**Opponent:** Stockfish depth 24
**Evaluation Depth:** 18

## Results Summary

| Model | W | D | L | Score |
|-------|---|---|---|-------|
| convnet | 0 | 0 | 2 | 0.0/2 |
| resnet | 0 | 0 | 2 | 0.0/2 |
| square_transformer | 0 | 0 | 2 | 0.0/2 |
| piece_transformer | 0 | 0 | 2 | 0.0/2 |

## Game Files

- **Combined PGN:** [all_games.pgn](all_games.pgn) - Open in Lichess, Chess.com, or any chess software
- **Individual PGNs:** `pgn/` directory

## Evaluation Charts

![All Games Comparison](figures/all_games_comparison.png)

### Individual Game Charts

#### convnet

![Game 1](figures/convnet_game_1_eval.png)

![Game 2](figures/convnet_game_2_eval.png)

#### resnet

![Game 1](figures/resnet_game_1_eval.png)

![Game 2](figures/resnet_game_2_eval.png)

#### square_transformer

![Game 1](figures/square_transformer_game_1_eval.png)

![Game 2](figures/square_transformer_game_2_eval.png)

#### piece_transformer

![Game 1](figures/piece_transformer_game_1_eval.png)

![Game 2](figures/piece_transformer_game_2_eval.png)

## How to View Games

1. **Lichess:** Go to lichess.org/paste and paste the PGN content
2. **Chess.com:** Use chess.com/analysis and import PGN
3. **Desktop Apps:** Open with ChessBase, Arena, SCID, or Lucas Chess
4. **Command Line:** Use `python-chess` to parse the PGN files
