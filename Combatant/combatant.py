import os
import sys
import json
import logging
import pika

from plark_game.classes.newgame import load_agent
from plark_game.classes.pantherAgent_load_agent import Panther_Agent_Load_Agent
from plark_game.classes.pelicanAgent_load_agent import Pelican_Agent_Load_Agent

from schema import deserialize_state

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


AGENTS_PATH = os.path.join(
    "/plark_ai_public", "data", "agents", "models", "latest"
)

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


def run_combatant(agent_type, agent_path, agent_name, basic_agents_path):
    """
    Run a combatant agent in the tournament environment.
    Load the agent from the agent path, and subscribe to a RabbitMQ queue.
    """

    agent = Combatant(agent_path, agent_name, basic_agents_path)

    agent_queue = "rpc_queue_{}".format(agent_type)

    if "RABBITMQ_HOST" in os.environ.keys():
        hostname = os.environ["RABBITMQ_HOST"]
    else:
        hostname = "localhost"

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=hostname)
    )

    channel = connection.channel()

    channel.queue_declare(queue=agent_queue)

    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=agent_queue, on_message_callback=agent.get_action
    )

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
