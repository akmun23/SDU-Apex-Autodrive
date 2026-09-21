import os
from glob import glob
from setuptools import setup

package_name = "sdu_apex_autodrive"
package_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(package_root)


def source_files(pattern):
    """Resolve package data from this file, not the temporary build cwd.

    Colcon may execute setup.py from a generated build directory.  Relative
    globs then omit newly added config/launch files and can leave an otherwise
    successful Python package without metadata or its runtime configuration.
    """
    return glob(pattern)


setup(
    name=package_name,
    version="0.3.0",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), source_files("config/*")),
        (
            os.path.join("share", package_name, "launch"),
            source_files("launch/*.launch.py"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Aksel Munksgaard-Ottosen",
    maintainer_email="akmun23@student.sdu.dk",
    description="AutoDRIVE RoboRacer single-launch racing integration.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "actuator_interface = sdu_apex_autodrive.actuator_interface:main",
            "lap_map_saver = sdu_apex_autodrive.lap_map_saver:main",
            "autodrive_bridge_40hz = sdu_apex_autodrive.bridge_40hz:main",
        ],
    },
)
