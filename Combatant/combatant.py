import os
import sys
import json
import logging
import pika
import uuid

from plark_game.classes.newgame import load_agent
from plark_game.classes.pantherAgent_load_agent import Panther_Agent_Load_Agent
from plark_game.classes.pelicanAgent_load_agent import Pelican_Agent_Load_Agent

from schema import deserialize_state

##########################################################
# set AGENTS_PATH to point to the directory containg your agents, relative
# to the base `plark_ai_public` directory (i.e. keep "/plark_ai_public" at
# the start of the path below - this is the directory as it will be seen
# in the docker image).
#
# This directory should have sub-directories 'panther' and/or 'pelican', and
# those should have a sub-directory containing your agent.
# Agent can be either a .zip file and metadata json file, or a .py file.

AGENTS_PATH = os.path.join(
#    "/plark_ai_public", "data", "agents", "models", "latest"
    "/Users/nbarlow/plark_ai_public", "data", "greg_test"
)

##########################################################
# You shouldn't need to change BASIC_AGENTS_PATH, unless you
# alter the directory structure of this package.

BASIC_AGENTS_PATH = os.path.normpath(
    os.path.join(
        "/plark_ai_public",
        "Components",
        "plark-game",
        "plark_game",
        "agents",
        "basic",
    )
)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_combatant(
    agent_path, agent_name, basic_agents_path, game=None, **kwargs
):
    """
    Loads agent as a combatant.
        agent_path:
        agent_name:
        basic_agents_path:
        game:
        kwargs:
    Returns:
        agent
    """

    if ".py" in agent_path:
        return load_agent(
            agent_path, agent_name, basic_agents_path, game, **kwargs
        )
    else:

        files = os.listdir(agent_path)

        for f in files:
            if ".zip" not in f:
                # ignore non agent files
                pass

            elif ".zip" in f:
                # load model
                metadata_filepath = os.path.join(agent_path, "metadata.json")
                agent_filepath = os.path.join(agent_path, f)

                with open(metadata_filepath) as f:
                    metadata = json.load(f)

                if (
                    "image_based" in metadata
                    and metadata["image_based"] is False
                ):
                    return load_agent(
                        agent_path,
                        agent_name,
                        basic_agents_path,
                        game,
                        **kwargs
                    )

                observation = None
                image_based = True
                algorithm = metadata["algorithm"]
                print("algorithm: ", algorithm)

                if metadata["agentplayer"] == "pelican":
                    return Pelican_Agent_Load_Agent(
                        agent_filepath,
                        algorithm,
                        observation,
                        image_based,
                    )
                elif metadata["agentplayer"] == "panther":
                    return Panther_Agent_Load_Agent(
                        agent_filepath,
                        algorithm,
                        observation,
                        image_based,
                    )

    return None


class Combatant:
    def __init__(
        self, agent_path, agent_name, basic_agents_path, game=None, **kwargs
    ):
        self.agent = load_combatant(
            agent_path, agent_name, basic_agents_path, game=None, **kwargs
        )
        self.ready = False

    def get_action(self, ch, method, props, body):

        state = json.loads(body)
        # convert json objects back into e.g. Torpedo instances
        deserialized_state = deserialize_state(state)
        # ask the agent for the action, given this state
        response = self.agent.getAction(deserialized_state)
        ch.basic_publish(
            exchange="",
            routing_key=props.reply_to,
            properties=pika.BasicProperties(
                correlation_id=props.correlation_id
            ),
            body=str(response),
        )

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def send_ready(self, agent_type, conn, channel, queue):
        """
        Send a message to the battleground indicating we are ready.
        """
        channel.basic_publish(
            exchange="",
            routing_key=queue,
            properties=pika.BasicProperties(
            ),
            body="{}_READY".format(agent_type.upper()),
        )
        self.ready = True
        return True


def run_combatant(agent_type, agent_path, agent_name, basic_agents_path):
    """
    Run a combatant agent in the tournament environment.
    Load the agent from the agent path, and subscribe to a RabbitMQ queue.
    """

    agent = Combatant(agent_path, agent_name, basic_agents_path)

    agent_queue = "rpc_queue_{}".format(agent_type)

    ready_queue = "rpc_queue_ready"

    if "RABBITMQ_HOST" in os.environ.keys():

        hostname = os.environ["RABBITMQ_HOST"]
    else:
        hostname = "localhost"

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=hostname)
    )

    channel = connection.channel()

    channel.queue_declare(queue=agent_queue)
    channel.queue_declare(queue=ready_queue)

    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=agent_queue, on_message_callback=agent.get_action
    )

    agent.send_ready(agent_type, connection, channel, ready_queue)
    print(" [x] {} Awaiting RPC requests".format(agent_type.upper()))
    channel.start_consuming()


if __name__ == "__main__":
    """"""

    agent_type = sys.argv[1].lower()

    if agent_type not in ["pelican", "panther"]:
        raise RuntimeError(
            "First argument to combatant.py must be 'pelican' or 'panther'"
        )

    subdirs = os.listdir(os.path.join(AGENTS_PATH, agent_type))
    for subdir in subdirs:
        agent_path = os.path.join(AGENTS_PATH, agent_type, subdir)
        break

    agent_name = "comb_%s" % (agent_type)

    run_combatant(agent_type, agent_path, agent_name, BASIC_AGENTS_PATH)
