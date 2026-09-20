FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS cuda_toolkit
FROM autodriveecosystem/autodrive_roboracer_api:2026-iros-compete@sha256:4ce4334657feb4c6760aa61f23a76f8962bf80e8e746e2547b082985a95a46a2

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-12.6
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

COPY --from=cuda_toolkit /usr/local/cuda-12.6 /usr/local/cuda-12.6

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libeigen3-dev \
    python3-colcon-common-extensions python3-matplotlib python3-numpy python3-opencv python3-pandas python3-pip python3-pytest python3-scipy python3-yaml \
    ros-humble-ackermann-msgs ros-humble-ament-cmake-python \
    ros-humble-nav2-lifecycle-manager ros-humble-nav2-map-server \
    ros-humble-rclc ros-humble-rclcpp-components \
    ros-humble-slam-toolbox ros-humble-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# trajectory-planning-helpers 0.79 pins a prebuilt quadprog 0.1.7 extension
# that does not link on this Ubuntu/Python combination.  The newer compatible
# wheel exposes the same API and keeps the optimizer importable.
RUN python3 -m pip install --no-cache-dir --no-deps trajectory-planning-helpers==0.79 \
    && python3 -m pip install --no-cache-dir --no-deps quadprog==0.1.13

WORKDIR /workspace
COPY . /workspace/src

RUN source /opt/ros/humble/setup.bash \
    && source /home/autodrive_devkit/install/setup.bash \
    && CMAKE_BUILD_PARALLEL_LEVEL=1 colcon build --parallel-workers 1 --merge-install --symlink-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="61;89"

# The competition image has one fixed, self-contained startup path.  Compose
# explicitly overrides this with docker/entrypoint.sh for development only.
COPY docker/competition_entrypoint.sh /home/autodrive_devkit.sh
RUN chmod +x /home/autodrive_devkit.sh

ENTRYPOINT ["/home/autodrive_devkit.sh"]
