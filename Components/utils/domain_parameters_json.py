import json
import math

def compute_torpedo_speeds(turn_limit, init_speed):

    #The speed at turn n+1 is equal to half the speed at turn n rounded up to the nearest
    #whole number
    torp_speeds = [init_speed]
    for i in range(1, turn_limit):
        torp_speeds.append(math.ceil(torp_speeds[-1]/2))
    
    return torp_speeds

#Integrate random param instance into a readable game config json
def create_game_config_json(param_instance):

    game_config = {
        "game_settings" : {
            "maximum_turns" : param_instance["max_turns"].item(),
            "map_width" : param_instance["map_width"].item(),
            "map_height" : param_instance["map_height"].item(),
            "driving_agent" : "pelican",
            "max_illegal_moves_per_turn" : 10
        },
        "game_rules" : {
            "bingo_limit" : 0,
            "winchester_rule" : True,
            "escape_rule" : True,
            "panther" : {
                "move_limit" : param_instance["move_limit_panther"].item(),
                "start_col" : param_instance["start_col_panther"].item(),
                "start_row" : param_instance["start_row_panther"].item(),
                "agent_filepath" : "pantherAgent_random_walk.py",
                "agent_name" : "Panther_Agent_Random_Walk",
                "render_height" : 250,
                "render_width" : 310
            },
            "pelican" : {
                "move_limit" : param_instance["move_limit_pelican"].item(),
                "start_col" : param_instance["start_col_pelican"].item(),
                "start_row" : param_instance["start_row_pelican"].item(),
                "madman_range" : 1,
                "agent_filepath" : "pelicanAgent_3_buoys.py",
                "agent_name" : "Pelican_Agent_3_Buoys",
                "default_torps" : param_instance["default_torpedos"].item(),
                "default_sonobuoys" : param_instance["default_sonobuoys"].item(),
                "render_height" : 250 ,
                "render_width" : 310
            },
            "torpedo" : {
                "speed" : compute_torpedo_speeds(param_instance["turn_limit"].item(),
                                                 param_instance["speed"].item()),
                "turn_limit" : param_instance["turn_limit"].item(),
                "hunt" : True,
                "search_range" : param_instance["search_range"].item()
            },
            "sonobuoy" : {
                "active_range" : param_instance["active_range"].item(),
                "display_range" : True
            }
        },
        "render_settings" : {
            "hex_scale" : 40,
            "output_view_all" : True
        }
    }

    return game_config

#Dump param instance dictionary of the form defined in domain_parameters.py
def dump_param_instance_json(param_instance, file_path):

    game_config = create_game_config_json(param_instance)

    with open(file_path, 'w') as outfile:
        json.dump(game_config, outfile, indent=4)
