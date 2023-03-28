FROM python:3.11-slim

RUN apt-get update && apt-get upgrade -y
RUN /usr/local/bin/python -m pip install --upgrade pip

COPY ./requirements.txt /tmp/project_requirements_file/
RUN pip install -r /tmp/project_requirements_file/requirements.txt

RUN mkdir -p /project_code/app
WORKDIR /project_code/app

# Create the user
# src: https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ARG USERNAME=docker_user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    # && apt-get update \
    # && apt-get install -y sudo \
    # && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    # && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************
RUN chown -R $USER_UID:$USER_UID /project_code/app
RUN chmod 770 /project_code/app

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME