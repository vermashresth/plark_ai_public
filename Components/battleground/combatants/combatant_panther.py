from abc import abstractmethod

class combatant_panther():
    @abstractmethod
    def __load_agent(self):
        self.panther_agent = None

    def __init__(self):
        self.__load_agent()