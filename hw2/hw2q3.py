import numpy as np

widthMaze = 5
heightMaze = 5
numStates = widthMaze * heightMaze
numActions = 4

qValues = np.zeros((numStates, numActions))
eTraces = np.zeros((numStates, numActions))

