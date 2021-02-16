FROM ubuntu:18.04

### get pip git etc

RUN apt-get update; apt-get install -y python3
RUN apt-get update; apt-get install -y python3-pip
RUN apt-get update; apt-get install -y locales

RUN apt-get update; apt-get install -y git
RUN apt-get update; apt-get install -y git-lfs
RUN apt-get update; apt-get install -y libgl1-mesa-glx
RUN apt-get update; apt-get install -y libsm6 libxrender1 libfontconfig1
# Set the locale
RUN sed -i -e 's/# en_GB.UTF-8 UTF-8/en_GB.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_GB.UTF-8
ENV LANGUAGE en_GB:en
ENV LC_ALL en_GB.UTF-8

# Temporary fix
RUN python3 -m pip install --upgrade pip
ADD Combatant/temp_requirements.txt /tmp/temp_requirements.txt
RUN python3 -m pip install -r /tmp/temp_requirements.txt


ADD Combatant/combatant_requirements.txt /tmp/combatant_requirements.txt

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /tmp/combatant_requirements.txt

ADD . /plark_ai_public

WORKDIR /plark_ai_public

RUN rm /tmp/combatant_requirements.txt

RUN python3 -m pip install -r requirements.txt

RUN cd Components/plark-game; python3 -m pip install .

### TOTAL HACK - MUST BE A BETTER WAY TO INSTALL resources/font.tff
RUN cp -r /plark_ai_public/Components/plark-game/plark_game/classes/resources /usr/local/lib/python3.6/dist-packages/plark_game/classes/

ENV PLARKAICOMPS /plark_ai_public/Components


ENV PYTHONPATH "${PYTHONPATH}:/plark_ai_public/Combatant"