import numpy as np
import sys

if len(sys.argv) >= 3 and sys.argv[2] == 'all':
    size = np.inf
else:
    size = 1000

with np.printoptions(threshold=size):
    print(np.load(sys.argv[1], allow_pickle=True))