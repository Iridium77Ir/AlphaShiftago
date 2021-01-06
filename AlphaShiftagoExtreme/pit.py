import Arena
from MCTS import MCTS
from shiftago.shiftagoGame import shiftagoGame
from shiftago.shiftagoGamePlayers import *
from shiftago.keras.NNet import NNetWrapper as NNet
#from shiftago.keras1.NNet import NNetWrapper as NNet1
#from shiftago.keras2.NNet import NNetWrapper as NNet2


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True

g = shiftagoGame()

# all players
hp = HumanShiftagoPlayer(g).play
rp = RandomShiftagoPlayer(g).play


# nnet players
""" n1 = NNet1(g)
n1.load_checkpoint('./temp/','n1.pth.tar')

args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)

n1p = lambda x, y: np.argmax(mcts1.getActionProb(x, y, temp=0))


n2 = NNet2(g)
n2.load_checkpoint('./temp/', 'n2.pth.tar')
mcts2 = MCTS(g, n2, args1)
player2 = lambda x, y: np.argmax(mcts1.getActionProb(x, y, temp=0))

arena = Arena.Arena(n1p, player2, g, display=shiftagoGame.display)

print(arena.playGames(20, verbose=True)) """

n1 = NNet(g)
n1.load_checkpoint('./temp/','checkpoint2_5.pth.tar')

args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)

n1p = lambda x, y: np.argmax(mcts1.getActionProb(x, y, temp=0))

player2 =  rp

arena = Arena.Arena(n1p, player2, g, display=shiftagoGame.display)

print(arena.playGames(20, verbose=True))
