import data_util
import chess
b = chess.Board()
b.push_uci('a2a4')
b.push_uci('a7a5')
b.push_uci('a1a3')
b.push_uci('a8a6')
s = data_util.board_to_state(b)
b_ = data_util.state_to_board(s)

print(b_.fen())
print (b.fen())