from plark_game.classes import Panther_Agent

class PantherNN(Panther_Agent):

    def __init__(self):
        pass

    def getAction(self, state):
        print("PantherNN action!")
        return '1'
