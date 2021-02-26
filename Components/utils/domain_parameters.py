#This script provides a number of functions for enumerating different types of permutations
#of domain parameter values accoring to parameter upper and lower bounds
#This script also provides functionality to randomly generate a number of different types
#of domain parameter instances

import numpy as np
import sys
import math
import random

map_width_lb = 25
map_width_ub = 35
map_height_lb = 25
map_height_ub = 35
max_turns_lb = 25
max_turns_ub = 50
move_limit_panther_lb = 1
move_limit_panther_ub = 3
move_limit_pelican_lb = 15
move_limit_pelican_ub = 25
default_torpedos_lb = 1
default_torpedos_ub = 5
default_sonobuoys_lb = 15
default_sonobuoys_ub = 25
turn_limit_lb = 1
turn_limit_ub = 3
#Torpedo speed is the speed in the first turn, the rest of the turn speeds are computed
speed_lb = 2
speed_ub = 4
search_range_lb = 2
search_range_ub = 4
active_range_lb = 1
active_range_ub = 4

def start_col_panther_lb(map_width):
    return int(math.floor(0.33 * map_width))
def start_col_panther_ub(map_width):
    return int(math.floor(0.66 * map_width))
def start_row_panther_lb(map_height):
    return 0
def start_row_panther_ub(map_height):
    return int(math.floor(0.2 * map_height))

def start_col_pelican_lb(map_width):
    return 0 
def start_col_pelican_ub(map_width):
    return int(math.floor(0.33 * map_width))
def start_row_pelican_lb(map_height):
    return int(math.floor(0.8 * map_height))
def start_row_pelican_ub(map_height):
    return map_height-1

def compute_range(lb, ub):
    return list(range(lb, ub+1))

domain_parameter_ranges = {
    "map_width" : compute_range(map_width_lb, map_width_ub),
    "map_height" : compute_range(map_height_lb, map_height_ub),
    "max_turns" : compute_range(max_turns_lb, max_turns_ub),
    "move_limit_panther" : compute_range(move_limit_panther_lb, move_limit_panther_ub),
    "move_limit_pelican" : compute_range(move_limit_pelican_lb, move_limit_pelican_ub),
    "default_torpedos" : compute_range(default_torpedos_lb, default_torpedos_ub),
    "default_sonobuoys" : compute_range(default_sonobuoys_lb, default_sonobuoys_ub),
    "turn_limit" : compute_range(turn_limit_lb, turn_limit_ub),
    "speed" : compute_range(speed_lb, speed_ub),
    "search_range" : compute_range(search_range_lb, search_range_ub),
    "active_range" : compute_range(active_range_lb, active_range_ub),
    "start_col_panther" : [],
    "start_row_panther" : [],
    "start_col_pelican" : [],
    "start_row_pelican" : []
}

#Create a parameter dictionary from a numpy array of parameters
def create_param_instance(np_params):

    #Check numpy array is correct size
    if(len(np_params) != 15):
        print("Cannot create a parameter dictionary from the following numpy array", 
            np_params, "\nThe length of this array should be 15!")
        sys.exit()

    parameter_instance = {
        "map_width" : np_params[0],
        "map_height" : np_params[1],
        "max_turns" : np_params[2],
        "move_limit_panther" : np_params[3],
        "move_limit_pelican" : np_params[4],
        "default_torpedos" : np_params[5],
        "default_sonobuoys" : np_params[6],
        "turn_limit" : np_params[7],
        "speed" : np_params[8],
        "search_range" : np_params[9],
        "active_range" : np_params[10],
        "start_col_panther" : np_params[11],
        "start_row_panther" : np_params[12],
        "start_col_pelican" : np_params[13],
        "start_row_pelican" : np_params[14]
    }

    return parameter_instance

#Computes start positions in domain parameter ranges according to map height and map width
def compute_start_positions(map_width, map_height):
    domain_parameter_ranges["start_col_panther"] = compute_range(start_col_panther_lb(map_width),
                                                                 start_col_panther_ub(map_width))
    domain_parameter_ranges["start_row_panther"] = compute_range(start_row_panther_lb(map_height),
                                                                 start_row_panther_ub(map_height))
    domain_parameter_ranges["start_col_pelican"] = compute_range(start_col_pelican_lb(map_width),
                                                                 start_col_pelican_ub(map_width))
    domain_parameter_ranges["start_row_pelican"] = compute_range(start_row_pelican_lb(map_height),
                                                                 start_row_pelican_ub(map_height))

#Check dictionary for map height and map width being in the correct place because this is
#essential for these algorithms to work
def check_domain_parameter_ranges_validity():
    if(not ("map_width" in domain_parameter_ranges)):
        print("map_width not in domain_parameter_ranges!")
        sys.exit()
    if(not ("map_height" in domain_parameter_ranges)):
        print("map_height not in domain_parameter_ranges!")
        sys.exit()
    if(list(domain_parameter_ranges)[0] != "map_width"):
        print("map_width needs to be the first item in the dictionary!")
        sys.exit()
    if(list(domain_parameter_ranges)[1] != "map_height"):
        print("map_height needs to be the second item in the dictionary!")
        sys.exit()

#Utility function to calculate whether the number of endpoints is 1 or 2
def calculate_num_endpoints(param_range):
    return 2 if len(param_range) > 1 else 1

#Utility function to calculate endpoints
def calculate_endpoints(param_range):
    if len(param_range) > 1:
        return [param_range[0], param_range[-1]]
    else:
        return [param_range[0]]

#Recursive function that enumerates over the n-dimensional (where n is the number of domain 
#parameter types) "hyperlattice"?
def recursively_enumerate_params(param_index, param_instance, all_permutations, 
                                 return_dict, endpoints):
    if(param_index == len(domain_parameter_ranges)):
        #print(param_instance)
        #Either create a list of dictionaries
        if return_dict:
            param_dict = create_param_instance(param_instance)
            all_permutations.append(param_dict)
        #Or a list of numpy arrays
        else:
            all_permutations.append(np.copy(param_instance))
        return all_permutations
    else:
        if endpoints:
            #Only get endpoints from parameter range
            range_values = calculate_endpoints(list(domain_parameter_ranges.values())[param_index])
        else:
            range_values = list(domain_parameter_ranges.values())[param_index]
        for v in range_values:
            param_instance[param_index] = v
            #Compute values for start positions once map width and map height have been decided
            #Hacky as hell but I know the first two dictionary entries are map width and
            #map height
            if(param_index == 1):
               compute_start_positions(param_instance[0], param_instance[1]) 
            all_permutations = recursively_enumerate_params(param_index+1, param_instance, 
                                                            all_permutations, return_dict,
                                                            endpoints)
        return all_permutations
        
#Returns a list of numpy arrays with each numpy array being a particular permutation of the
#domain parameters
#OR if you set return_dict to True, it returns a list of dictionaries instead
#If endpoints is set to True, only the endpoints of the parameter ranges are considered,
#not the full parameter range
def compute_all_permutations(domain_parameter_ranges, return_dict = False, endpoints = False):

    #Check map width and map height are in the dictionary and in the right place
    check_domain_parameter_ranges_validity()

    all_permutations = []
    param_instance = np.zeros(len(domain_parameter_ranges), dtype=np.uint8) 
    recursively_enumerate_params(0, param_instance, all_permutations, return_dict, endpoints)

    return all_permutations

#Generate random parameter instance according to lower and upper bounds at the top of
#this script
#One can choose whether to return a dictionary or a numpy array
#If endpoints is set to True, it considers only the endpoints of the parameter ranges
def generate_random_param_instance(return_dict = True, endpoints = False):

    #Check map width and map height are in the dictionary and in the right place
    check_domain_parameter_ranges_validity()

    rand_instance = np.zeros(len(domain_parameter_ranges), dtype=np.uint8) 
    for i in range(0, len(domain_parameter_ranges)):
        if endpoints:
            rand_instance[i] = \
                random.choice(calculate_endpoints(list(domain_parameter_ranges.values())[i]))
        else:
            rand_instance[i] = random.choice(list(domain_parameter_ranges.values())[i])
        #Calculate starting positions now that map width and map height have been chosen
        if i == 1:
           compute_start_positions(rand_instance[0], rand_instance[1]) 

    if return_dict:
        return create_param_instance(rand_instance)
    else:
        return rand_instance

#Calculate the number of parameter permutations
#Only counts endpoints of parameter ranges if endpoints = True
def calculate_num_param_permutations(endpoints = False):

    num_params = 1

    #If only calculating the number of endpoint permutations
    if endpoints:
        num_params *= calculate_num_endpoints(domain_parameter_ranges["map_width"])
        num_params *= calculate_num_endpoints(domain_parameter_ranges["map_height"])
        num_params *= calculate_num_endpoints(domain_parameter_ranges["max_turns"])
        num_params *= calculate_num_endpoints(domain_parameter_ranges["move_limit_panther"])
        num_params *= calculate_num_endpoints(domain_parameter_ranges["move_limit_pelican"])
        num_params *= calculate_num_endpoints(domain_parameter_ranges["default_torpedos"])
        num_params *= calculate_num_endpoints(domain_parameter_ranges["default_sonobuoys"])
        num_params *= calculate_num_endpoints(domain_parameter_ranges["turn_limit"])
        num_params *= calculate_num_endpoints(domain_parameter_ranges["speed"])
        num_params *= calculate_num_endpoints(domain_parameter_ranges["search_range"])
        num_params *= calculate_num_endpoints(domain_parameter_ranges["active_range"])

        start_col_products_sum = 0
        for width in calculate_endpoints(domain_parameter_ranges["map_width"]):
            start_col_panther_params = compute_range(start_col_panther_lb(width),
                                                     start_col_panther_ub(width))
            start_col_pelican_params = compute_range(start_col_pelican_lb(width),
                                                     start_col_pelican_ub(width))
            start_col_products_sum += calculate_num_endpoints(start_col_panther_params) * \
                                      calculate_num_endpoints(start_col_pelican_params)

        num_params *= start_col_products_sum / \
                      calculate_num_endpoints(domain_parameter_ranges["map_width"])

        start_row_products_sum = 0
        for height in calculate_endpoints(domain_parameter_ranges["map_height"]):
            start_row_panther_params = compute_range(start_row_panther_lb(height),
                                                     start_row_panther_ub(height))
            start_row_pelican_params = compute_range(start_row_pelican_lb(height),
                                                     start_row_pelican_ub(height))
            start_row_products_sum += calculate_num_endpoints(start_row_panther_params) * \
                                      calculate_num_endpoints(start_row_pelican_params)

        num_params *= start_row_products_sum / \
                      calculate_num_endpoints(domain_parameter_ranges["map_height"])

    else:
        num_params *= len(domain_parameter_ranges["map_width"])
        num_params *= len(domain_parameter_ranges["map_height"])
        num_params *= len(domain_parameter_ranges["max_turns"])
        num_params *= len(domain_parameter_ranges["move_limit_panther"])
        num_params *= len(domain_parameter_ranges["move_limit_pelican"])
        num_params *= len(domain_parameter_ranges["default_torpedos"])
        num_params *= len(domain_parameter_ranges["default_sonobuoys"])
        num_params *= len(domain_parameter_ranges["turn_limit"])
        num_params *= len(domain_parameter_ranges["speed"])
        num_params *= len(domain_parameter_ranges["search_range"])
        num_params *= len(domain_parameter_ranges["active_range"])

        #For the start column and rows, the range size is variable depending on the width and
        #height respectively so we can't just multiply by a fixed range size
        #We have to split it up an multiply accordingly
        #Ask James for the derived combinatorics equation if you have further questions
        start_col_products_sum = 0
        for width in domain_parameter_ranges["map_width"]:
            start_col_panther_params = compute_range(start_col_panther_lb(width),
                                                     start_col_panther_ub(width))
            start_col_pelican_params = compute_range(start_col_pelican_lb(width),
                                                     start_col_pelican_ub(width))
            start_col_products_sum += len(start_col_panther_params) * \
                                      len(start_col_pelican_params)

        num_params *= start_col_products_sum / len(domain_parameter_ranges["map_width"])

        start_row_products_sum = 0
        for height in domain_parameter_ranges["map_height"]:
            start_row_panther_params = compute_range(start_row_panther_lb(height),
                                                     start_row_panther_ub(height))
            start_row_pelican_params = compute_range(start_row_pelican_lb(height),
                                                     start_row_pelican_ub(height))
            start_row_products_sum += len(start_row_panther_params) * \
                                      len(start_row_pelican_params)

        num_params *= start_row_products_sum / len(domain_parameter_ranges["map_height"])

    return int(num_params)

#Call examples

#rand_params = generate_random_param_instance(return_dict=True, endpoints=False)
#print(rand_params)

num_param_perms = calculate_num_param_permutations(endpoints=False)
print("Number of parameter permutations:", num_param_perms)

#all_permutations = compute_all_permutations(domain_parameter_ranges, 
#                                            return_dict=False, endpoints=False)
