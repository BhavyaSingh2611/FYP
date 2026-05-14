# Data Setup

## Folder Structure

Create the following structure inside the `data/` folder:

```text
data/
  chess_eval.parquet
  puzzles/
    puzzle.parquet
    puzzle2.parquet
    puzzle3.parquet
```

## Dataset Sources

Download the datasets from Kaggle and place the `.parquet` files into the structure above.

- Lichess evals (stripped): https://www.kaggle.com/datasets/bhavyasingh2611/lichess-evals-stripped
- Lichess chess puzzles: https://www.kaggle.com/datasets/lichess/chess-puzzles

## Notes

Both links provide zip archives. Extract the `.parquet` files and copy them into the folders shown above. Please make sure that the names of the folders and parquet files remain the same to be able to use predefined scripts for training/benchmarking.