FROM autodriveecosystem/autodrive_roboracer_api:2026-iros-practice

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libeigen3-dev \
    python3-colcon-common-extensions \
    python3-matplotlib \
    python3-numpy \
    python3-opencv \
    python3-pil \
    python3-pip \
    python3-pytest \
    python3-scipy \
    python3-yaml \
    ros-humble-ackermann-msgs \
    ros-humble-ament-cmake-python \
    ros-humble-diagnostic-updater \
    ros-humble-nav2-amcl \
    ros-humble-nav2-lifecycle-manager \
    ros-humble-nav2-map-server \
    ros-humble-rclc \
    ros-humble-rclcpp-components \
    ros-humble-slam-toolbox \
    ros-humble-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace/src

RUN source /opt/ros/humble/setup.bash \
    && source /home/autodrive_devkit/install/setup.bash \
    && colcon build \
      --merge-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo 'source /home/autodrive_devkit/install/setup.bash' >> /root/.bashrc && \
    echo 'source /workspace/install/setup.bash' >> /root/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
