.PHONY: combatant combatant_test

TEAM_ID = test# <<change me>>

combatant:
	docker build -t plark_hunt/team_$(TEAM_ID):latest -f Combatant/Dockerfile .

combatant_test:
	docker run plark_hunt/team_$(TEAM_ID):latest Combatant/tests/test_combatant.sh