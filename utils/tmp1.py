import math


initial_lambda = 0.5
alp = 0.002
f = lambda frame_idx: initial_lambda + (1.0 - initial_lambda) * math.tanh(alp*frame_idx)

for i in range(192*5):
    print(f(i))