import numpy as np

class SimpleShiftagoGame():
    def __init__(self):
        self.board = np.zeros((7,7), int)
        self.turn = True
        self.gameEnd = False
        self.winner = 0

    def move(self, num):
        if self.gameEnd == True:
            return None
        if 0 in self.board:
            pass
        else:
            return False
        if num >= 28:
            return False
        else:
            if num//7 == 0:
                if self.checkCol(num%7) == False:
                    return False
                else:
                    self.shiftColRight(num%7)
                    self.board[0,num%7] = 1 if self.turn == True else -1
            elif num//7 == 1:
                if self.checkRow(num%7) == False:
                    return False
                else:
                    self.shiftRowLeft(6-num%7)
                    self.board[num%7,6] = 1 if self.turn == True else -1
            elif num//7 == 2:
                if self.checkCol(6-num%7) == False:
                    return False
                else:
                    self.shiftColLeft(num%7)
                    self.board[6,6-num%7] = 1 if self.turn == True else -1
            elif num//7 == 3:
                if self.checkRow(6-num%7) == False:
                    return False
                else:
                    self.shiftRowRight(6-num%7)
                    self.board[6-num%7,0] = 1 if self.turn == True else -1

            self.turn = not self.turn

        res = self.checkEnd()
        if res != 0:
            self.winner = res
            self.gameEnd = True
            return None
        return True

    def checkEnd(self):
        if 0 in self.board:
            pass
        else:
            return 1e-4
            
        adjacent = 0
        adjacent2 = 0

        #Check Columns
        for i in range(7):
            for j in range(7):
                if self.board[i,j] != 0:
                    adjacent += self.board[i,j]
                if self.board[i,j] == 0:
                    adjacent = 0
                if adjacent == 4:
                    return 1
                if adjacent == -4:
                    return -1

                #Check rows
                if self.board[j,i] != 0:
                    adjacent2 += self.board[j,i]
                if self.board[j,i] == 0:
                    adjacent2 = 0
                if adjacent2 == 4:
                    return 1
                if adjacent2 == -4:
                    return -1
            adjacent = 0
            adjacent2 = 0
        
        #Check top-left to bottom-right diagonal
        res = self.checkDiagonal()
        if res != 0:
            return res
        #Check top-right to bottom-left diagonal
        self.rotate(1)
        res = self.checkDiagonal()
        self.rotate(3)
        if res != 0:
            return res
        #Return 0 because no one won
        return 0

    #Utility functions
    def checkCol(self, num):
        placed = 0
        for i in range(7):
            if self.board[i, num] != 0:
                placed += 1
            if self.board[i, num] == 0:
                return True
        if placed == 7:
            return False
        else:
            return True

    def checkRow(self, num):
        placed = 0
        for i in range(7):
            if self.board[num, i] != 0:
                placed += 1
            if self.board[num, i] == 0:
                return True
        if placed == 7:
            return False
        else:
            return True

    def shiftRowRight(self, num):
        temp = []
        for i in range(7):
            if self.board[num, i] == 0 and i == 0:
                return
            elif self.board[num, i] == 1 or self.board[num, i] == -1:
                temp.append(self.board[num, i])
                self.board[num, i] = 0
            elif self.board[num, i] == 0 and i != 0:
                temp.reverse()
                for j in range(len(temp)):
                    self.board[num, i-j] = temp[j]
                return
        return

    def shiftColRight(self, num):
        temp = []
        for i in range(7):
            if self.board[i, num] == 0 and i == 0:
                return
            elif self.board[i, num] == 1 or self.board[i, num] == -1:
                temp.append(self.board[i, num])
                self.board[i, num] = 0
            elif self.board[i, num] == 0 and i != 0:
                temp.reverse()
                for j in range(len(temp)):
                    self.board[i-j, num] = temp[j]
                return
        return

    def shiftRowLeft(self, num):
        self.rotate(2)
        temp = []
        for i in range(7):
            if self.board[num, i] == 0 and i == 0:
                self.rotate(2)
                return
            elif self.board[num, i] == 1 or self.board[num, i] == -1:
                temp.append(self.board[num, i])
                self.board[num, i] = 0
            elif self.board[num, i] == 0 and i != 0:
                temp.reverse()
                for j in range(len(temp)):
                    self.board[num, i-j] = temp[j]
                self.rotate(2)
                return
        self.rotate(2)
        return

    def shiftColLeft(self, num):
        temp = []
        self.rotate(2)
        for i in range(7):
            if self.board[i, num] == 0 and i == 0:
                self.rotate(2)
                return
            elif self.board[i, num] == 1 or self.board[i, num] == -1:
                temp.append(self.board[i, num])
                self.board[i, num] = 0
            elif self.board[i, num] == 0 and i != 0:
                temp.reverse()
                for j in range(len(temp)):
                    self.board[i-j, num] = temp[j]
                self.rotate(2)
                return
        self.rotate(2)
        return
    
    def rotate(self, num):
        self.board = np.rot90(self.board, num)
        return

    def checkDiagonal(self):
        adjacent1 = 0
        adjacent2 = 0
        for i in range(4):
            for j in range(7-i):
                if self.board[i+j, j] != 0:
                    adjacent1 += self.board[i+j, j]
                if self.board[i+j, j] == 0:
                    adjacent1 = 0
                if adjacent1 == 4:
                    return 1
                if adjacent1 == -4:
                    return -1

                if self.board[j, j+i] != 0:
                    adjacent2 += self.board[j, j+i]
                if self.board[j, j+i] == 0:
                    adjacent2 = 0
                if adjacent2 == 4:
                    return 1
                if adjacent2 == -4:
                    return -1
            adjacent1 = 0
            adjacent2 = 0
        return 0