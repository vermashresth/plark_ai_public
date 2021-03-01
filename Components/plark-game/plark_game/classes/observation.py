import gym
from gym import error, spaces, utils
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Observation():

    def __init__(self, game, **kwargs):
        self.game = game
        self.kwargs = kwargs
        if self.kwargs.get('driving_agent'):
            self.driving_agent = self.kwargs.get('driving_agent')
        else:
            self.driving_agent = 'pelican'
            self.kwargs['driving_agent'] = self.driving_agent

        #A boolean flag to indicate whether to add the domain parameters to the
        #observation space
        self.domain_params_in_obs = kwargs.get('domain_params_in_obs', True)

        #A boolean flag to determine whether to normalise the observations
        self.normalise = self.kwargs.get('normalise', True)

        #self.num_remaining_domain_params = len(self._determine_remaining_domain_parameters())

        #Old max maxs - before Simon's upper bounds
        #self.max_grid_width = 30
        #self.max_grid_height = 30
        #self.max_sonobuoys = 25
        #self.max_turns = 40
        #self.max_pelican_moves = 25
        #self.max_panther_moves = 5
        #self.max_torpedoes = 25
        #self.max_torpedo_turns = 3
        #self.max_torpedo_speed = 3

        #Simon's upper bounds
        self.max_grid_width = 35
        self.max_grid_height = 35
        self.max_sonobuoys = 25
        self.max_turns = 50
        self.max_pelican_moves = 25
        self.max_panther_moves = 3
        self.max_torpedoes = 5
        self.max_torpedo_turns = 3
        self.max_torpedo_speed = 4
        self.max_torp_search_range = 4
        self.max_sonobuoy_active_range = 4

        self.pelican_col = 0
        self.pelican_row = 0
        self.panther_col = 0
        self.panther_row = 0

        #Obs shape is a list of the MAX values that all of the states will ever take
        #The size of this list is required by stable baslines in order to determine
        #the number of inputs - I think
        obs_shape = []

        # Represent the chosen grid width
        assert game.map_width <= self.max_grid_width, "Game width must not be greater than max grid width: {}".format(self.max_grid_width)
        assert game.map_height <= self.max_grid_height, "Game height must not be greater than max grid height: {}".format(self.max_grid_width)
        obs_shape.append(self.max_grid_width)
        obs_shape.append(self.max_grid_height)
        # Number of turns remaining
        assert game.maxTurns <= self.max_turns, "Game max turns must not be greater than max possible turns: {}".format(self.max_turns)
        obs_shape.append(self.max_turns)

        if self.driving_agent == 'pelican':
            # Number of moves remaining per turn
            assert game.pelican_parameters['move_limit'] <= self.max_pelican_moves, "Game move limit must not be greater than max move limit: {}".format(self.max_pelican_moves)
            obs_shape.append(self.max_pelican_moves)
            # Pelican location
            obs_shape.append(self.max_grid_width)
            obs_shape.append(self.max_grid_height)

            # Madman status - this is a boolean
            obs_shape.append(2)

            # sonobuoy range
            self.max_sb_range = int(max(self.max_grid_height, self.max_grid_width)/2)
            assert game.sonobuoy_parameters['active_range'] <= self.max_sb_range, "Sonobuoy active range must not be greater than max range: {}".format(self.max_sb_range)
            obs_shape.append(self.max_sb_range)

            # Number of sonobuoys remaining
            assert game.pelican_parameters['default_sonobuoys'] <= self.max_sonobuoys, "Starting number of sonobuoys must not be greater than maximum possible sonobuoys: {}".format(self.max_sonobuoys)
            obs_shape.append(self.max_sonobuoys)

            # sonobuoy locations and activations
            for i in range(self.max_sonobuoys):
                # location (max_height+1, max_width+1) represents undeployed
                obs_shape.append(self.max_grid_width+1)
                obs_shape.append(self.max_grid_height+1)
                # binary: 0=inactive, 1=active
                obs_shape.append(2)

            # Number of torpedoes remaining
            assert game.pelican_parameters['default_torps'] <= self.max_torpedoes, "Starting number of sonobuoys must not be greater than maximum possible sonobuoys: {}".format(self.max_torpedoes)
            obs_shape.append(self.max_torpedoes)

            # Torpedo hunt enabled - boolean
            obs_shape.append(2)

            # Torpedo speeds for each turn
            assert game.torpedo_parameters['turn_limit'] <= self.max_torpedo_turns, "Torpedo turn limit must not be greater than the maximum torpedo turns: {}".format(self.max_torpedo_turns)
            for speed in game.torpedo_parameters['speed']:
                assert speed <= self.max_torpedo_speed, "Torpedo speed for each turn must not be greater than the maximum torpedo speed: {}".format(self.max_torpedo_speed)
            for turn in range(self.max_torpedo_turns):
                # speed can also be zero:
                # e.g. max speed of 3 leads to possible values [0,1,2,3]
                obs_shape.append(self.max_torpedo_speed+1)

            # torpedo locations
            for i in range(self.max_torpedoes):
                # location (max_height+1, max_width+1) represents undeployed
                obs_shape.append(self.max_grid_width+1)
                obs_shape.append(self.max_grid_height+1)

        else:
            # Number of moves remaining per turn
            assert game.panther_parameters['move_limit'] <= self.max_panther_moves, "Game move limit must not be greater than max move limit: {}".format(self.max_panther_moves)
            obs_shape.append(self.max_panther_moves)
            # Pelican location
            obs_shape.append(self.max_grid_width)
            obs_shape.append(self.max_grid_height)

            # Panther location
            obs_shape.append(self.max_grid_width)
            obs_shape.append(self.max_grid_height)

            # Madman status - this is a boolean
            obs_shape.append(2)

            # sonobuoy range
            self.max_sb_range = int(max(self.max_grid_height, self.max_grid_width)/2)
            assert game.sonobuoy_parameters['active_range'] <= self.max_sb_range, "Sonobuoy active range must not be greater than max range: {}".format(self.max_sb_range)
            obs_shape.append(self.max_sb_range)

            # Number of sonobuoys remaining
            assert game.pelican_parameters['default_sonobuoys'] <= self.max_sonobuoys, "Starting number of sonobuoys must not be greater than maximum possible sonobuoys: {}".format(self.max_sonobuoys)
            obs_shape.append(self.max_sonobuoys)


            # sonobuoy locations and activations
            for i in range(self.max_sonobuoys):
                # location (max_height+1, max_width+1) represents undeployed
                obs_shape.append(self.max_grid_width+1)
                obs_shape.append(self.max_grid_height+1)
                # binary: 0=inactive, 1=active
                obs_shape.append(2)

            # Number of torpedoes remaining
            assert game.pelican_parameters['default_torps'] <= self.max_torpedoes, "Starting number of sonobuoys must not be greater than maximum possible sonobuoys: {}".format(self.max_torpedoes)
            obs_shape.append(self.max_torpedoes)

            # Torpedo hunt enabled - boolean
            obs_shape.append(2)

            # Torpedo speeds for each turn
            assert game.torpedo_parameters['turn_limit'] <= self.max_torpedo_turns, "Torpedo turn limit must not be greater than the maximum torpedo turns: {}".format(self.max_torpedo_turns)
            for speed in game.torpedo_parameters['speed']:
                assert speed <= self.max_torpedo_speed, "Torpedo speed for each turn must not be greater than the maximum torpedo speed: {}".format(self.max_torpedo_speed)
            for turn in range(self.max_torpedo_turns):
                # speed can also be zero:
                # e.g. max speed of 3 leads to possible values [0,1,2,3]
                obs_shape.append(self.max_torpedo_speed+1)

            # torpedo locations
            for i in range(self.max_torpedoes):
                # location (max_height+1, max_width+1) represents undeployed
                obs_shape.append(self.max_grid_height+1)
                obs_shape.append(self.max_grid_width+1)

        #Add max domain params
        self.domain_params_obs_shape = []

        #Map width, map height, active range are already given so leave them out
        self.domain_params_obs_shape.append(self.max_sonobuoys)
        self.domain_params_obs_shape.append(self.max_turns)
        self.domain_params_obs_shape.append(self.max_pelican_moves)
        self.domain_params_obs_shape.append(self.max_panther_moves)
        self.domain_params_obs_shape.append(self.max_torpedoes)
        self.domain_params_obs_shape.append(self.max_torpedo_turns)
        for i in range(self.max_torpedo_turns):
            self.domain_params_obs_shape.append(self.max_torpedo_speed)
        self.domain_params_obs_shape.append(self.max_torp_search_range)

        #Add max domain params to obs_shape if flag is true
        if self.domain_params_in_obs:
            obs_shape += self.domain_params_obs_shape

        #Goes through and adds 1 to everything
        #I just end up taking this away later anyway :/
        obs_shape_new = []
        for i in obs_shape:
            obs_shape_new.append(i+1)
        obs_shape = obs_shape_new
        self.domain_params_obs_shape = np.array(self.domain_params_obs_shape)+1

        self.observation_max_for_normalisation = obs_shape
        self.observation_space = spaces.MultiDiscrete(obs_shape)

    def get_observation_space(self):
        return self.observation_space

    def _get_location(self, board, item):
        for col in range(board.cols):
            for row in range(board.rows):
                if board.is_item_type_in_cell(item, col, row):
                    return (col, row)
        return (None,None)

    #Do not call! Private function
    def __determine_pelican_specific_observations(self, state, obs):
        # Remaining moves
        remaining_pelican_moves = state['pelican_max_moves']  - state['pelican_move_in_turn']
        obs.append(remaining_pelican_moves)
        # Pelican location
        obs += [state['pelican_location']['col'], state['pelican_location']['row']]
        # Madman status
        obs.append(int(state['madman_status']))
        # Sonobuoy range - fixed per game
        obs.append(state['sonobuoy_range'])
        # Remaining Sonobuoys
        obs.append(state['remaining_sonobuoys'])
        # sonobuoy locations & activations
        for i in range(self.max_sonobuoys):
            if i < len(state['deployed_sonobuoys']):
                buoy = state['deployed_sonobuoys'][i]
                active = 1 if buoy['state'] == "HOT" else 0
                obs += [buoy['col'], buoy['row'], active]
            else:
                obs += [self.max_grid_width+1, self.max_grid_height+1, 0]

        # Remaining Torpedoes
        obs.append(state['remaining_torpedoes'])
        # Torpedo hunt enabled
        obs.append(int(state['torpedo_hunt_enabled']))
        # Torpedo speeds per turn
        for i in range (self.max_torpedo_turns):
            if  i < len(state['torpedo_speeds']):
                speed = state['torpedo_speeds'][i]
                obs.append(speed)
            else:
                obs.append(0)

        # torpedo locations & activations
        for i in range(self.max_torpedoes):
            if i < len(state['deployed_torpedoes']):
                torp = state['deployed_torpedoes'][i]
                obs += [torp['col'], torp['row']]
            else:
                obs += [self.max_grid_width+1, self.max_grid_height+1]

        return obs

    #Do not call! Private function
    def __determine_panther_specific_observations(self, state, obs):
        # Remaining moves
        remaining_panther_moves = state['panther_max_moves']  - state['panther_move_in_turn']
        obs.append(remaining_panther_moves)
        # Pelican location
        obs += [state['pelican_location']['col'], state['pelican_location']['row']]
        # Panther location
        obs += [state['panther_location']['col'], state['panther_location']['row']]
        # Madman status
        obs.append(int(state['madman_status']))
        # Sonobuoy range - fixed per game
        obs.append(state['sonobuoy_range'])
        # Remaining Sonobuoys
        obs.append(state['remaining_sonobuoys'])
        # sonobuoy locations & activations
        for i in range(self.max_sonobuoys):
            if i < len(state['deployed_sonobuoys']):
                buoy = state['deployed_sonobuoys'][i]
                active = 1 if buoy['state'] == "HOT" else 0
                obs += [buoy['col'], buoy['row'], active]
            else:
                obs += [self.max_grid_width+1, self.max_grid_height+1, 0]

        # Remaining Torpedoes
        obs.append(state['remaining_torpedoes'])
        # Torpedo hunt enabled
        obs.append(int(state['torpedo_hunt_enabled']))

        # Torpedo speeds per turn
        for i in range (self.max_torpedo_turns):
            if  i < len(state['torpedo_speeds']):
                speed = state['torpedo_speeds'][i]
                obs.append(speed)
            else:
                obs.append(0)

        # torpedo locations & activations
        for i in range(self.max_torpedoes):
            if i < len(state['deployed_torpedoes']):
                torp = state['deployed_torpedoes'][i]
                obs += [torp['col'], torp['row']]
            else:
                obs += [self.max_grid_width+1, self.max_grid_height+1]

        return obs

    #Get numpy array of remaining domain parameters - by remaining domain parameters, I
    #mean the domain parameters that have not yet been added to observation (map_width,
    #map_height and sonobuoy_range/active_range have already been added)
    def _determine_remaining_domain_parameters(self, normalise=False):

        domain_params = []
        domain_params.append(self.game.pelican_parameters['default_sonobuoys'])
        domain_params.append(self.game.maxTurns)
        domain_params.append(self.game.pelican_parameters['move_limit'])
        domain_params.append(self.game.panther_parameters['move_limit'])
        domain_params.append(self.game.pelican_parameters['default_torps'])
        domain_params.append(self.game.torpedo_parameters['turn_limit'])
        #Default value for torpedo speed is 0 if there are not maximum
        #torpedo turns
        torp_speeds = [0] * self.max_torpedo_turns
        for i in range(len(torp_speeds)):
            if i < len(self.game.torpedo_parameters['speed']):
                torp_speeds[i] = self.game.torpedo_parameters['speed'][i]
        domain_params += torp_speeds
        domain_params.append(self.game.torpedo_parameters['search_range'])

        domain_params = np.array(domain_params, dtype=float)

        #Normalise remaining domain parameters
        if normalise:
            domain_params = np.divide(domain_params, self.domain_params_obs_shape-1)

        return domain_params

    def _determine_observation(self, state, norm_obs=False, domain_params=False):

        new_pelican_col,new_pelican_row = self._get_location(self.game.gameBoard, "PELICAN")
        if new_pelican_col is not None:
            self.pelican_col = new_pelican_col
            self.pelican_row = new_pelican_row
        state['pelican_location'] =  {'col': self.pelican_col, 'row': self.pelican_row}

        new_panther_col, new_panther_row = self._get_location(self.game.gameBoard, "PANTHER")
        if new_panther_col is not None:
            self.panther_col = new_panther_col
            self.panther_row = new_panther_row
        state['panther_location'] =  {'col': self.panther_col, 'row': self.panther_row}
        state['madman_status'] = self.game.pelicanPlayer.madmanStatus
        state['sonobuoy_range'] = self.game.sonobuoy_parameters['active_range']

        pelican_payload = self.game.pelicanPlayer.payload
        remaining_sbs = len([obj for obj in pelican_payload if obj.type == "SONOBUOY"])
        remaining_torps = len([obj for obj in pelican_payload if obj.type == "TORPEDO"])
        state['remaining_sonobuoys'] = remaining_sbs
        state['deployed_sonobuoys'] = [{'col': b.col, 'row': b.row, 'state': b.state} for b in self.game.globalSonobuoys]
        state['remaining_torpedoes'] = remaining_torps
        state['torpedo_hunt_enabled'] = self.game.torpedo_parameters['hunt']
        state['torpedo_speeds'] = self.game.torpedo_parameters['speed']
        state['deployed_torpedoes'] = [{'col': t.col, 'row': t.row} for t in self.game.globalTorps]

        obs = []
        # Current game dimensions - fixed per game
        obs += [state['map_width'], state['map_height']]
        # Remaining turns
        remaining_turns = state['maxTurns'] - state['turn_count']
        obs.append(remaining_turns)
        if self.driving_agent == 'pelican':
            obs = self.__determine_pelican_specific_observations(state, obs)
        else:
            obs = self.__determine_panther_specific_observations(state, obs)

        #If flag is true, add domain params to the observation space
        #Max width, max height and active range are already added above
        if domain_params:
            obs += self._determine_remaining_domain_parameters().tolist()

        #To normalise the observations we need to divide them all by the max values
        #that a state can be - we construct this in obs_maxs
        if norm_obs:
            #These if conditions just deal with different circumstances for how to
            #build the normalisation constants
            obs_maxs = np.array(self.observation_max_for_normalisation)
            if domain_params and not self.domain_params_in_obs:
                obs_maxs += np.array(self.domain_params_obs_shape)
            if not domain_params and self.domain_params_in_obs:
                slice_index = len(self.observation_max_for_normalisation) - \
                              len(self.domain_params_obs_shape)
                obs_maxs = obs_maxs[:slice_index]

            #Divide by normalisation constants
            return np.divide(np.array(obs), obs_maxs-1)
        else:
            return np.array(obs, dtype=float)

    #This function returns a numpy array with the full observation. It is modified
    #according to the member variables self.domain_params_in_obs and self.normalise
    def get_observation(self, state):
        return self._determine_observation(state, norm_obs=self.normalise,
                                           domain_params=self.domain_params_in_obs)

    #Get numpy array of unmodified observation with no normalisation or domain parameters
    def get_original_observation(self, state):
        return self._determine_observation(state, norm_obs=False, domain_params=False)

    #Get numpy array of normalised observation with no domain parameters
    def get_normalised_observation(self, state):
        return self._determine_observation(state, norm_obs=True, domain_params=False)

    #Get numpy array of remaining domain parameters - by remaining domain parameters, I
    #mean the domain parameters that have not yet been added to observation (map_width,
    #map_height and sonobuoy_range/active_range have already been added)
    def get_remaining_domain_parameters(self):
        return self._determine_remaining_domain_parameters(normalise=False)

    #Get numpy array of normalised domain parameters
    def get_normalised_remaining_domain_parameters(self):
        return self._determine_remaining_domain_parameters(normalise=True)
