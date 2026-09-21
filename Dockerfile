FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS cuda_toolkit
FROM autodriveecosystem/autodrive_roboracer_api:2026-iros-compete@sha256:4ce4334657feb4c6760aa61f23a76f8962bf80e8e746e2547b082985a95a46a2

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-12.6
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64

COPY --from=cuda_toolkit /usr/local/cuda-12.6 /usr/local/cuda-12.6

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libeigen3-dev \
    python3-colcon-common-extensions \
    ros-humble-ackermann-msgs ros-humble-ament-cmake-python \
    ros-humble-nav2-lifecycle-manager ros-humble-nav2-map-server \
    ros-humble-rclc ros-humble-rclcpp-components \
    ros-humble-slam-toolbox ros-humble-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace/src

RUN source /opt/ros/humble/setup.bash \
    && source /home/autodrive_devkit/install/setup.bash \
    && CMAKE_BUILD_PARALLEL_LEVEL=1 colcon build --parallel-workers 1 --merge-install --symlink-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="61;89"

# The competition image has one fixed, self-contained startup path.
COPY docker/competition_entrypoint.sh /home/autodrive_devkit.sh
RUN chmod +x /home/autodrive_devkit.sh

ENTRYPOINT ["/home/autodrive_devkit.sh"]
