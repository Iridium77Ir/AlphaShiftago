import argparse
from operator import concat
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../..')
from utils import *
from NeuralNet import NeuralNet

import argparse

from .OthelloNNet import OthelloNNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.1,
    'epochs': 15,
    'batch_size': 64,
    'cuda': True,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs, input_scores = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        input_scores = np.asarray(input_scores)

        concatBoards = []
        for i in range(len(input_boards)):
            scores1 = np.zeros((7,7))
            scores1.fill(input_scores[i][0])
            scores2 = np.zeros((7,7))
            scores2.fill(input_scores[i][1])
            tempBoard = np.dstack((input_boards[i], scores1, scores2))
            concatBoards.append(tempBoard)
            del tempBoard, scores1, scores2

        concatBoards = np.asarray(concatBoards)

        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x = concatBoards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)

    def predict(self, board, scores):
        """
        board: np array with board
        """

        # preparing input
        scores1 = np.zeros((7,7))
        scores2 = np.zeros((7,7))

        scores1.fill(scores[0])
        scores2.fill(scores[1])

        concatBoards = np.dstack((board, scores1, scores2))
        concatBoards = concatBoards[np.newaxis, :, :, :]

        # run
        pi, v = self.nnet.model.predict(concatBoards)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        """ if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath)) """
        self.nnet.model.load_weights(filepath)
