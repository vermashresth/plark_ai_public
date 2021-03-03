.PHONY: login pelican_build pelican_push pelican_test panther_build panther_push panther_test

REGISTRY = turingrldsg.azurecr.io

login:
	docker login $(REGISTRY) -u ${RL_TOKEN_NAME} -p ${RL_TOKEN}

# PELICAN
pelican_build:
	docker build -t $(REGISTRY)/${RL_TEAM_ID}:pelican_latest -f Combatant/Dockerfile .

pelican_push:
	docker push $(REGISTRY)/${RL_TEAM_ID}:pelican_latest

pelican_test:
	docker run $(REGISTRY)/${RL_TEAM_ID}:pelican_latest Combatant/tests/test_pelican.sh

# PANTHER
panther_build:
	docker build -t $(REGISTRY)/${RL_TEAM_ID}:panther_latest -f Combatant/Dockerfile .

panther_push:
	docker push $(REGISTRY)/${RL_TEAM_ID}:panther_latest

panther_test:
	docker run $(REGISTRY)/${RL_TEAM_ID}:panther_latest Combatant/tests/test_panther.sh
