from plark_game.classes.pelicanAgentFixedSBs import PelicanAgentFixedSBs

class Pelican_Agent_5_Buoys(PelicanAgentFixedSBs):

	def __init__(self):
		super(Pelican_Agent_5_Buoys, self).__init__()
		self.sb_locations = [
			{'col':1, 'row':1},
			{'col':1, 'row':6},
			{'col':4, 'row':4},
			{'col':8, 'row':1},
			{'col':8, 'row':7}
		]
