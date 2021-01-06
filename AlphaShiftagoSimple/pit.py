import Arena
from MCTS import MCTS
from shiftago.shiftagoGame import shiftagoGame as Game
from shiftago.shiftagoGamePlayers import *
from shiftago.keras.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True

g = Game()

# all players
hp = HumanShiftagoPlayer(g).play
rp = RandomShiftagoPlayer(g).play



# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 100, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


player2 = rp

arena = Arena.Arena(n1p, player2, g, display=Game.display)

print(arena.playGames(10, verbose=True))
