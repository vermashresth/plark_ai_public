import os, subprocess, time, signal
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np 

import sys
from plark_game.classes.environment import Environment
from plark_game.classes.observation import Observation
import random
import math

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PlarkEnv(gym.Env):
        metadata = {'render.modes': ['human']}

        def __init__(self,config_file_path=None,verbose=False, **kwargs):
                self.kwargs = kwargs


                self.random_panther_start_position = kwargs.get('random_panther_start_position', False)
                self.random_pelican_start_position = kwargs.get('random_pelican_start_position', False)

                self.render_height = kwargs.get('render_height', None)
                if self.render_height is None:
                        self.render_height = 250
                        self.kwargs['render_height'] = self.render_height
                self.render_width = kwargs.get('render_width', None)
                if self.render_width is None:
                        self.render_width = 310
                        self.kwargs['render_width'] = self.render_width

                self.driving_agent = kwargs.get('driving_agent', None)
                if self.driving_agent is None:
                        self.driving_agent = 'pelican'
                        self.kwargs['driving_agent'] = self.driving_agent

                #logger.info('plark.kwargs :'+ str(self.kwargs))

                self.verbose = verbose
                self.viewer = None
                self.server_process = None
                self.server_port = None

                self.image_based = kwargs.get('image_based', False)

                #logger.info('self.image_based :'+ str(self.image_based))
                self.env = Environment()
                self.config_file_path = config_file_path

                self.illegal_move_reward = -0.1
                self.buoy_too_close_reward = -0.2
                self.buoy_far_apart_reward = 0.5

                #1 UP
                #2 UP RIGHT
                #3 DOWN RIGHT
                #4 DOWN
                #5 DOWN LEFT
                #6 UP LEFT

                if self.driving_agent == 'panther':
                        self.view = 'PANTHER'
                        self.ACTION_LOOKUP = {
                                0 : '1',
                                1 : '2',
                                2 : '3',
                                3 : '4', 
                                4 : '5',  
                                5 : '6',  
                                6 : 'end'  
                        }
                elif self.driving_agent == 'pelican':
                        self.view = 'PELICAN'
                        self.ACTION_LOOKUP = {
                                0 : '1',
                                1 : '2',
                                2 : '3',
                                3 : '4', 
                                4 : '5',  
                                5 : '6',  
                                6 : 'drop_buoy',  
                                7 : 'drop_torpedo',  
                                8 : 'end'  
                        }
                else:
                        raise ValueError('driving_agent not set correctly')

                # Inverse action lookup for looking up specific actions
                self.action_index = dict((val, key) for key, val in self.ACTION_LOOKUP.items())

                

                if self.image_based:
                        self.reset()
                        self.game = self.env.activeGames[len(self.env.activeGames)-1]
                        N_CHANNELS = 3
                        logger.info('observation space: Height:'+str(self.render_height)+', Width:'+str(self.render_width)+', Channels:'+str(N_CHANNELS))
                        self.observation_space = spaces.Box(low=0, high=255,
                                                                                        shape=(self.render_height, self.render_width, N_CHANNELS), dtype=np.uint8)
                        logger.info('Image observations created')                                                               
                        self.normalise = None
                        self.domain_params_in_obs = None
                else:
                        if len(self.env.activeGames) > 0:
                                self.env.activeGames[len(self.env.activeGames)-1].reset_game()
                        else:    
                                if self.config_file_path:
                                        #logger.info('config filepath: ' +str(self.config_file_path))
                                        self.env.createNewGame(config_file_path=self.config_file_path, **self.kwargs)
                                else:
                                        self.env.createNewGame(**self.kwargs)
                        self.game = self.env.activeGames[len(self.env.activeGames)-1]           
                        self.observation = Observation(self.game, **kwargs)
                        self.observation_space = self.observation.get_observation_space() 

                        self.normalise = self.observation.normalise
                        self.domain_params_in_obs = self.observation.domain_params_in_obs
                        #logger.info('Non image observations created')

                        
                        
                if self.driving_agent == 'panther':
                        self.action_space = spaces.Discrete(7)
                elif self.driving_agent == 'pelican':    
                        self.action_space = spaces.Discrete(9)
                self.status = self.env.activeGames[len(self.env.activeGames)-1].gameState

        def close(self):
                self.env.stopAllGames()

        def _get_location(self, board, item):
                for col in range(board.cols):
                        for row in range(board.rows):
                                if board.is_item_type_in_cell(item, col, row):
                                        return (col, row)
                raise ValueError("Could not find {}".format(item))


        def _observation(self):
                if self.image_based:    
                        pil_image = self.env.activeGames[len(self.env.activeGames)-1].render(self.render_width,self.render_height,self.view)
                        np_image = np.array(pil_image, dtype=np.uint8)
                        return np_image
                else:
                        obs = self.observation.get_observation(self.env.activeGames[len(self.env.activeGames)-1]._state(self.view))
                        #return obs
                        return np.array(obs, dtype=np.float)


        def step(self, action):
                action = self.ACTION_LOOKUP[action]
                game = self.env.activeGames[len(self.env.activeGames)-1]

                if self.verbose:
                        logger.info('Action:'+action)
                gameState,uioutput = game.game_step(action)
                self.status = gameState
                self.uioutput = uioutput 
                
                ob = self._observation()

                #print(self.driving_agent)
                #print(ob)

                reward = 0
                done = False
                _info = { 'turn': game.turn_count }

                if self.driving_agent == 'pelican':
                        illegal_move = game.illegal_pelican_move
                else:
                        illegal_move = game.illegal_panther_move
                _info['illegal_move'] = illegal_move

                if illegal_move == True:
                        reward = reward + self.illegal_move_reward
                if self.driving_agent == 'pelican': #If it wasn't an illegal move.
                        ## Reward for droping a sonobouy 
                        if action == 'drop_buoy' and illegal_move == False:
                                self.globalSonobuoys = game.globalSonobuoys
                                if len(self.globalSonobuoys)>1: 
                                        sonobuoy = self.globalSonobuoys[-1]
                                        sbs_in_range = game.gameBoard.searchRadius(sonobuoy.col, sonobuoy.row, sonobuoy.range, "SONOBUOY")
                                        sbs_in_range.remove(sonobuoy) # remove itself from search results
                                        if len(sbs_in_range) > 0:
                                                reward = reward + self.buoy_too_close_reward
                                        else:
                                                reward = reward + self.buoy_far_apart_reward 
                                else:
                                        reward = reward + self.buoy_far_apart_reward        

                #  PELICANWIN = Pelican has won    
                #  ESCAPE     = Panther has won
                #  BINGO      = Panther has won, Pelican has reached it's turn limit and run out of fuel 
                #  WINCHESTER = Panther has won, All torpedoes dropped and stopped running. Panther can't be stopped   
                if self.status == "PELICANWIN" or self.status == "BINGO" or self.status == "WINCHESTER" or self.status == "ESCAPE": 
                        done = True
                        if self.verbose:
                                logger.info("GAME STATE IS " + self.status)    
                        if self.status in ["ESCAPE","BINGO","WINCHESTER"]:
                                if self.driving_agent == 'pelican':
                                        reward = -1 
                                        _info['result'] = "LOSE"
                                elif self.driving_agent == 'panther':  
                                        reward = 1 
                                        _info['result'] = "WIN"
                                else:
                                        raise ValueError('driving_agent not set correctly')
                        if self.status == "PELICANWIN":
                                if self.driving_agent == 'pelican':
                                        reward = 1 
                                        _info['result'] = "WIN"
                                elif self.driving_agent == 'panther':  
                                        reward = -1 
                                        _info['result'] = "LOSE"
                                else:
                                        raise ValueError('driving_agent not set correctly')
                
                return ob, reward, done, _info

        def set_pelican(self, pelican):
            self.env.activeGames[len(self.env.activeGames)-1].set_pelican(pelican)

        def set_panther(self, panther):
            self.env.activeGames[len(self.env.activeGames)-1].set_panther(panther)

        def set_pelican_using_path(self, pelican):
            self.env.activeGames[len(self.env.activeGames)-1].load_pelican_using_path(pelican)

        def set_panther_using_path(self, panther):
            self.env.activeGames[len(self.env.activeGames)-1].load_panther_using_path(panther)

        def clear_memory(self):
            self.env.activeGames[len(self.env.activeGames)-1].clear_memory()


        def wrapper_test_function(self):
            print("hello there!")

        def reset(self):
                #If a game already exists. reset
                if len(self.env.activeGames) > 0:
                        #On reset randomly places panther in a different location.
                        if self.random_panther_start_position: 
                                map_width = self.env.activeGames[len(self.env.activeGames)-1].map_width
                                map_height = self.env.activeGames[len(self.env.activeGames)-1].map_height
                                #Taken out of domain_parameters.py (Simon's bounds)
                                panther_start_col_lb = int(math.floor(0.33 * map_width))
                                panther_start_col_ub = int(math.floor(0.66 * map_width))
                                panther_start_row_lb = int(math.floor(0.8 * map_height))
                                panther_start_row_ub = map_height-1

                                panther_start_col = random.choice(range(panther_start_col_lb,
                                                                        panther_start_col_ub+1))
                                panther_start_row = random.choice(range(panther_start_row_lb,
                                                                        panther_start_row_ub+1)) 
                                self.env.activeGames[len(self.env.activeGames)-1].panther_start_col = panther_start_col
                                self.env.activeGames[len(self.env.activeGames)-1].panther_start_row = panther_start_row

                        #On reset randomly places pelican in a different location.
                        if self.random_pelican_start_position: 
                                map_width = self.env.activeGames[len(self.env.activeGames)-1].map_width
                                map_height = self.env.activeGames[len(self.env.activeGames)-1].map_height
                                #Taken out of domain_parameters.py (Simon's bounds)
                                pelican_start_col_lb = 0 
                                pelican_start_col_ub = int(math.floor(0.33 * map_width))
                                pelican_start_row_lb = 0
                                pelican_start_row_ub = int(math.floor(0.2 * map_height))

                                pelican_start_col = random.choice(range(pelican_start_col_lb,
                                                                        pelican_start_col_ub+1))
                                pelican_start_row = random.choice(range(pelican_start_row_lb,
                                                                        pelican_start_row_ub+1))
                                self.env.activeGames[len(self.env.activeGames)-1].pelican_start_col = pelican_start_col
                                self.env.activeGames[len(self.env.activeGames)-1].pelican_start_row = pelican_start_row

                        self.env.activeGames[len(self.env.activeGames)-1].reset_game()
                else:    
                        if self.config_file_path:
                                logger.info('config filepath: ' +str(self.config_file_path))
                                self.env.createNewGame(config_file_path=self.config_file_path, **self.kwargs)
                        else:
                                self.env.createNewGame(**self.kwargs)   

                self.game = self.env.activeGames[len(self.env.activeGames)-1]           

                if self.driving_agent == 'pelican':
                        self.render_width = self.game.pelican_parameters['render_width']
                        self.render_height = self.game.pelican_parameters['render_height']

                elif self.driving_agent == 'panther':
                        self.render_width = self.game.panther_parameters['render_width']
                        self.render_height = self.game.panther_parameters['render_height']

                return self._observation()


        def render(self, mode='human', close=False, view=None):
                if view is None:
                        view = self.view
                pil_image = self.env.activeGames[len(self.env.activeGames)-1].render(self.render_width,self.render_height,view)
                return pil_image
