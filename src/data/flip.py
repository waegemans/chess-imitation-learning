def swaprows(c):
  if c.isdigit():
    return str(9-(ord(c)-ord('0')))
  return c

while 1:
  try:
    board,color,castl,rest = input().split(' ')
  except EOFError:
    break
  enpass,mv = rest.split(',')
  if color == 'b':
    board = board.swapcase()
    board = '/'.join(list(board.split('/')[::-1]))
    color = 'w'
    castl = castl.swapcase()
    castl = ''.join(map(chr,sorted(map(ord,castl))))
    enpass = ''.join(map(swaprows,enpass))
    mv = ''.join(map(swaprows,mv))

  print(' '.join([board,color,castl,enpass])+','+mv)
