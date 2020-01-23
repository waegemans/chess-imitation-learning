import dataset
import chess
import data_util

b = chess.Board()

s = data_util.board_to_state(b)

c = data_util.state_to_cnn(s)

print(s[:768].reshape((12,64)))

for i in range(c.shape[0]):
  print(c[i,:,:])

print(data_util.move_to_action(chess.Move.from_uci("e2e4")).argmax())