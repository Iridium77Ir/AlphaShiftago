import numpy as np

class HumanShiftagoPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, scores):
        move = input()
        return int(move)

class RandomShiftagoPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, scores):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a