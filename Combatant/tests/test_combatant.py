#!/usr/bin/env python

import os
import sys
import json
from combatant import load_combatant
from schema import deserialize_state
import numpy as np

ALLOWED_ACTIONS = {
    "PANTHER": [
        "1",  # Up
        "2",  # Up right
        "3",  # Down right
        "4",  # Down
        "5",  # Down left
        "6",  # Up left
        "end",
    ],
    "PELICAN": [
        "1",  # Up
        "2",  # Up right
        "3",  # Down right
        "4",  # Down
        "5",  # Down left
        "6",  # Up left
        "drop_buoy",
        "drop_torpedo",
        "end",
    ],
}

AGENT_NAME = ""

agent_type = sys.argv[1]

basic_agents_path = os.path.join(
    "/plark_ai_public",
    "Components",
    "plark-game",
    "plark_game",
    "agents",
    "basic",
)

agent_path = os.path.join(
    "/plark_ai_public", "data", "agents", "models", "move_north_nn"
)

if agent_type == "PELICAN":
    # subdirs = os.listdir(os.path.join(agent_path, "pelican"))
    # for subdir in subdirs:
    #     agent_path = os.path.join(agent_path, "pelican", subdir)
    #     break

    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pelican")
    
elif agent_type == "PANTHER":

    # subdirs = os.listdir(os.path.join(agent_path, "panther"))
    # for subdir in subdirs:
    #     agent_path = os.path.join(agent_path, "panther", subdir)
    #     break
    
    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panther")

else:
    raise RuntimeError("Unknown agent_type - must be 'PELICAN' or 'PANTHER'")

state = deserialize_state(
    json.load(open(os.path.join(test_path, "10x10_state.json")))
)

obs = np.loadtxt(os.path.join(test_path, "10x10_obs.txt"))
obs_norm = np.loadtxt(os.path.join(test_path, "10x10_obs_norm.txt"))
d_params = np.loadtxt(os.path.join(test_path, "10x10_domain_params.txt"))
d_params_norm = np.loadtxt(os.path.join(test_path, "10x10_domain_params_norm.txt"))


agent = load_combatant(agent_path, AGENT_NAME, basic_agents_path)

action = agent.getTournamentAction(
    obs,
    obs_norm,
    d_params,
    d_params_norm,
    state
)

if action not in ALLOWED_ACTIONS[agent_type]:
    raise RuntimeError("NO!")
