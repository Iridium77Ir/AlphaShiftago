from __future__ import print_function
from shiftago.Shiftago import ExtremeShiftagoGame as Sf
import numpy as np
import copy

def getValidMoves(board, player):
        game = Sf()
        game.board = board
        game.turn = True if player == 1 else False

        valids = [0]*28
        for i in range(28):
            if i//7 == 0:
                if game.checkCol(i%7) == True:
                    valids[i] = 1
            elif i//7 == 1:
                if game.checkRow(i%7) == True:
                    valids[i] = 1
            elif i//7 == 2:
                if game.checkCol((7-1)-i%7) == True:
                    valids[i] = 1
            elif i//7 == 3:
                if game.checkRow((7-1)-i%7) == True:
                    valids[i] = 1

        return np.array(valids)

board = np.array([[1.,1.,1.,-1.,-1.,1.,1.]
,[-1.,0.,0.,0.,0.,-1.,1.]
,[-1.,-1.,1.,0.,0.,0.,1.]
,[1.,-1.,0.,0.,1.,0.,1.]
,[1.,0.,1.,0.,-1.,-1.,-1.]
,[1.,-1.,0.,0.,0.,-1.,1.]
,[-1.,1.,1.,1.,1.,-1.,1.]])

counts = getValidMoves(board, 1)

bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
print(bestAs)
bestA = np.random.choice(bestAs)
probs = [0] * len(counts)
probs[bestA] = 1
print(probs)