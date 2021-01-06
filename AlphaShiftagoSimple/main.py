import logging

import coloredlogs

from Coach import Coach
from shiftago.shiftagoGame import shiftagoGame as Game
from shiftago.keras.NNet import NNetWrapper as nn
""" from vierGewinnt.vierGewinntGame import vierGewinntGame as Game
from vierGewinnt.keras.NNet import NNetWrapper as nn """
""" from connect4.Connect4Game import Connect4Game as Game
from connect4.keras.NNet import NNetWrapper as nn """
from utils import *
import faulthandler

import sys
sys.setrecursionlimit(10**9)
faulthandler.enable()

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 100,            # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.5,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 40000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 30,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 20,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp/','checkpoint_100.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], 'best.pth.tar')
        nnet.load_checkpoint(args.load_folder_file[0], 'best.pth.tar')
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process ')
    c.learn()

if __name__ == "__main__":
    main()
