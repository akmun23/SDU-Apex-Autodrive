from glob import glob
import os

from setuptools import setup


package_name = 'sdu_apex_autodrive'

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aksel Munksgaard-Ottosen',
    maintainer_email='akmun23@student.sdu.dk',
    description=(
        'Single-launch AutoDRIVE controller and calibration integration.'
    ),
    license='MIT',
    entry_points={
        'console_scripts': [
            'actuator_interface = sdu_apex_autodrive.actuator_interface:main',
            'auto_global_localizer = sdu_apex_autodrive.auto_global_localizer:main',
            'throttle_characterization = sdu_apex_autodrive.throttle_characterization:main',
            'speed_tracking_test = sdu_apex_autodrive.speed_tracking_test:main',
            'steering_characterization = sdu_apex_autodrive.steering_characterization:main',
            'data_recorder = sdu_apex_autodrive.data_recorder:main',
            'create_open_map = sdu_apex_autodrive.create_open_map:main',
        ],
    },
)
