"""
All possible UCI moves in chess (for policy output).
"""


def _generate_all_uci_moves() -> list[str]:
    """Generate all possible UCI move strings."""
    moves = []
    files = "abcdefgh"
    ranks = "12345678"
    promotion_pieces = ["q", "r", "b", "n"]

    for from_file in files:
        for from_rank in ranks:
            for to_file in files:
                for to_rank in ranks:
                    from_sq = from_file + from_rank
                    to_sq = to_file + to_rank

                    if from_sq == to_sq:
                        continue

                    # Regular move
                    moves.append(from_sq + to_sq)

                    # Pawn promotions (from rank 7 to 8 for white, 2 to 1 for black)
                    if (from_rank == "7" and to_rank == "8") or (
                        from_rank == "2" and to_rank == "1"
                    ):
                        for promo in promotion_pieces:
                            moves.append(from_sq + to_sq + promo)

    return moves


ALL_UCI_MOVES = _generate_all_uci_moves()
UCI_MOVE_TO_INDEX = {move: idx for idx, move in enumerate(ALL_UCI_MOVES)}
INDEX_TO_UCI_MOVE = dict(enumerate(ALL_UCI_MOVES))
NUM_MOVES = len(ALL_UCI_MOVES)
