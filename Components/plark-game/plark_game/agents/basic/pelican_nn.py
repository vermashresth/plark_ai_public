from plark_game.classes import Pelican_Agent

class PelicanNN(Pelican_Agent):

    def __init__(self):
        pass

    def getAction(self, state):
        print("PelicanNN action!")
        return '1'
