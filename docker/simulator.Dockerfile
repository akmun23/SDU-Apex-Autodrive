FROM ubuntu:22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG SIMULATOR_URL=https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing/releases/download/2026-iros/autodrive_simulator_practice_linux.zip
ARG SIMULATOR_SHA256=4ffa685fb83a48c3b9f2d2bb5f093aa9e6367f4eb98e2aa1365f57315ebd82f9

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      libasound2 \
      libatomic1 \
      libfontconfig1 \
      libfreetype6 \
      libgl1 \
      libglib2.0-0 \
      libglvnd0 \
      libgtk-3-0 \
      libnss3 \
      libvulkan1 \
      libx11-6 \
      libx11-xcb1 \
      libxcb1 \
      libxcb-glx0 \
      libxcb-keysyms1 \
      libxcb-randr0 \
      libxcb-shape0 \
      libxcb-xinerama0 \
      libxcursor1 \
      libxdamage1 \
      libxext6 \
      libxi6 \
      libxinerama1 \
      libxrandr2 \
      libxrender1 \
      libxss1 \
      libxtst6 \
      unzip \
      xauth \
      x11-utils \
      xdotool \
      xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/autodrive \
    && curl -fL --retry 3 --retry-delay 2 "$SIMULATOR_URL" -o /tmp/autodrive/simulator.zip \
    && echo "$SIMULATOR_SHA256  /tmp/autodrive/simulator.zip" | sha256sum -c - \
    && unzip -q /tmp/autodrive/simulator.zip -d /tmp/autodrive \
    && mv /tmp/autodrive/autodrive_simulator /home/autodrive_simulator \
    && chmod +x "/home/autodrive_simulator/AutoDRIVE Simulator.x86_64" \
    && rm -rf /tmp/autodrive

COPY docker/simulator-entrypoint.sh /usr/local/bin/autodrive-simulator-entrypoint
COPY docker/headless-simulator.sh /usr/local/bin/autodrive-headless-simulator
RUN chmod +x \
      /usr/local/bin/autodrive-simulator-entrypoint \
      /usr/local/bin/autodrive-headless-simulator

WORKDIR /home/autodrive_simulator
ENTRYPOINT ["/usr/local/bin/autodrive-simulator-entrypoint"]
CMD ["bash"]
