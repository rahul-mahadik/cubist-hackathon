from __future__ import annotations

import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]


def evaluate_material(board: chess.Board) -> int:
    if board.is_checkmate():
        return -100_000 if board.turn == chess.WHITE else 100_000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score


def evaluate_handcrafted(board: chess.Board) -> int:
    score = evaluate_material(board)
    for square in board.pieces(chess.KNIGHT, chess.WHITE):
        score += KNIGHT_TABLE[square]
    for square in board.pieces(chess.KNIGHT, chess.BLACK):
        score -= KNIGHT_TABLE[chess.square_mirror(square)]
    turn = board.turn
    board.turn = chess.WHITE
    white_mobility = board.legal_moves.count()
    board.turn = chess.BLACK
    black_mobility = board.legal_moves.count()
    board.turn = turn
    score += 2 * (white_mobility - black_mobility)
    if board.is_check():
        score += -25 if board.turn == chess.WHITE else 25
    return score


def score_for_side(board: chess.Board) -> int:
    score = evaluate_handcrafted(board)
    return score if board.turn == chess.WHITE else -score

