from glob import glob
import os
from setuptools import setup

package_name = "sdu_apex_autodrive"

setup(
    name=package_name,
    version="0.3.0",
    packages=[
        package_name,
        package_name + ".scripts",
        package_name + ".odometry_analysis",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (
            os.path.join("share", package_name, "artifacts", "calibration"),
            ["artifacts/calibration/MANIFEST.yaml"],
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
            "calibration = sdu_apex_autodrive.calibration:main",
            "create_open_map = sdu_apex_autodrive.create_open_map:main",
            "lap_map_saver = sdu_apex_autodrive.lap_map_saver:main",
            "ground_truth_amcl_monitor = sdu_apex_autodrive.ground_truth_amcl_monitor:main",
            "autodrive_bridge_40hz = sdu_apex_autodrive.bridge_40hz:main",
            "timing_validator = sdu_apex_autodrive.timing_validator:main",
            "analyze_calibration = sdu_apex_autodrive.scripts.analyze_calibration:main",
            "audit_calibration_csv = sdu_apex_autodrive.scripts.audit_calibration_csv:main",
            "validate_odometry_observer = sdu_apex_autodrive.scripts.validate_odometry_observer:main",
        ],
    },
)
