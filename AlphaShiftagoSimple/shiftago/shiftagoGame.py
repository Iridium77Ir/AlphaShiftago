from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .Shiftago import SimpleShiftagoGame as Sf
import numpy as np
import copy


def printChar(num):
    endc = '\033[0m'
    blue = '\033[91m'
    green = '\033[92m'
    if num == 0:
        print(' 0  ', end='')
    elif num == 1:
        print(f"{blue} {str(num)}{endc}", end='')
    elif num == -1:
        print(f"{green}{str(num)}{endc}", end='')

class shiftagoGame(Game):
    def __init__(self):
        pass

    def getInitBoard(self):
        return np.zeros((7,7))

    def getBoardSize(self):
        return (7, 7)

    def getActionSize(self):
        return 28

    def getNextState(self, board, player, action):
        game = Sf()
        game.board = np.copy(board)
        game.turn = True if player == 1 else False
        game.move(action)

        return (game.board, 1 if game.turn == True else -1)

    def getValidMoves(self, board, player):
        game = Sf()
        game.board = board
        game.turn = True if player == 1 else False

        valids = [0]*self.getActionSize()
        for i in range(self.getActionSize()):
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

    def getGameEnded(self, board, player):
        game = Sf()
        game.board = board
        game.turn = True if player == 1 else False
        return game.checkEnd() * player

    def getCanonicalForm(self, board, player):
        return player*board

    def rotate(self, arr, num):
        temp = copy.deepcopy(arr)
        for i in range(num):
            temp = np.rot90(temp)
        return temp

    def mapToArr(self, arr):
        temp = np.zeros((7+2,7+2))
        for i in range(7):
            temp[0,i+1] = arr[i]
            temp[(7+1),(7+1)-(i+1)] = arr[i+(7*2)]
            temp[(7+1)-(i+1),0] = arr[i+(7*3)]
            temp[i+1,(7+1)] = arr[i+7]

        return temp

    def mapToFlat(self, arr):
        temp = np.zeros((self.getActionSize(),))
        for i in range(7):
            temp[i] = arr[0,i+1]
            temp[i+(7*2)] = arr[(7+1),(7+1)-(i+1)]
            temp[i+(7*3)] = arr[(7+1)-(i+1),0]
            temp[i+(7*1)] = arr[i+1,(7+1)]

        return temp

    def getSymmetries(self, board, pi):
        l = []
        #rotation
        l.append((board, pi))
        #flip lr
        l.append((np.fliplr(board), self.mapToFlat(np.fliplr(self.mapToArr(pi)))))
        #flip ud
        l.append((np.flipud(board), self.mapToFlat(np.flipud(self.mapToArr(pi)))))
        #flip ud lr
        l.append((np.fliplr(np.flipud(board)), self.mapToFlat(np.fliplr(np.flipud(self.mapToArr(pi))))))

        return l

    def stringRepresentation(self, board):
        return board.tostring()

    @staticmethod
    def display(board):
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                printChar(board[i,j])
                print(' - ', end='')
            print('')
