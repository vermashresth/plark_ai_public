#!/usr/bin/env python

import os
import sys
import json
from combatant import load_combatant
from schema import deserialize_state

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

TEST_PATH = os.path.join("/plark_ai_public", "Combatant", "tests", "states")

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
    "/plark_ai_public", "data", "agents", "models", "latest"
)


if agent_type == "PELICAN":
    subdirs = os.listdir(os.path.join(agent_path, "pelican"))
    for subdir in subdirs:
        agent_path = os.path.join(agent_path, "pelican", subdir)
        break

    state = deserialize_state(
        json.load(open(os.path.join(TEST_PATH, "state_10x10_pelican.json")))
    )
elif agent_type == "PANTHER":

    subdirs = os.listdir(os.path.join(agent_path, "panther"))
    for subdir in subdirs:
        agent_path = os.path.join(agent_path, "panther", subdir)
        break

    state = deserialize_state(
        json.load(open(os.path.join(TEST_PATH, "state_10x10_panther.json")))
    )
else:
    raise RuntimeError("Unknown agent_type - must be 'PELICAN' or 'PANTHER'")

print("agent_path: ", agent_path)

agent = load_combatant(agent_path, AGENT_NAME, basic_agents_path)

action = agent.getAction(state)

if action not in ALLOWED_ACTIONS[agent_type]:
    raise RuntimeError("NO!")
