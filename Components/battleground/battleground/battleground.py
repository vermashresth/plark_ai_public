from classes.environment import Environment
from classes.newgame import NewGame

class Battle(NewGame):
    def __init__(self,game_config, **kwargs):
		# load the game configurations
		self.load_configurations(game_config, **kwargs)

		# Create required game objects
		self.create_game_objects()

		# Load agents
		if self.driving_agent == 'panther':
			if not self.output_view_all:
				self.gamePlayerTurn = "PANTHER"
		
##### NOT FOR US			self.pelicanAgent = self.load_agent(self.pelican_parameters['agent_filepath'], self.pelican_parameters['agent_name'])

		else:
			if not self.output_view_all:
				self.gamePlayerTurn = "PELICAN"

##### NOT FOR US			self.pantherAgent = self.load_agent(self.panther_parameters['agent_filepath'], self.panther_parameters['agent_name'])

		# Game state variables
		self.default_game_variables()

		# Create UI objects and render game. This must be the last thing in the __init__
		if self.driving_agent == 'pelican':
			self.render_height = self.pelican_parameters['render_height']
			self.render_width = self.pelican_parameters['render_width']
		else:
			self.render_height = self.panther_parameters['render_height']
			self.render_width = self.panther_parameters['render_width']

		self.reset_game()
		self.render(self.render_width,self.render_height,self.gamePlayerTurn)        


	def pelicanPhase(self):
		# This is what is called if the panther is the playable agent.
		self.pelicanMove = Move()
		while True:
			### CHANGEME pelican_action = self.pelicanAgent.getAction(self._state("PELICAN"))
            pelican_action = None ### rabbitMQ call
			self.perform_pelican_action(pelican_action)

			if self.pelican_move_in_turn >= self.pelican_parameters['move_limit'] or pelican_action == 'end':
				break

	def pantherPhase(self):
		# This is what is called if the pelican is the playable agent.
		self.pantherMove = Move()
		while True:
			### CHANGEME panther_action = self.pantherAgent.getAction(self._state("PANTHER"))
            panther_action = None ### rabbitMQ call

			self.perform_panther_action(panther_action)
			if self.gameState == 'ESCAPE' or self.panther_move_in_turn >= self.panther_parameters['move_limit'] or panther_action == 'end':
				break

if __name__ == "__main__":
    print("Such nice work")