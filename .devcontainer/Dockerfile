FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
       libgl1 libglib2.0-0 build-essential protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

USER vscode
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/home/vscode/.cargo/bin:${PATH}"
