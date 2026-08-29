from glob import glob
import os

from setuptools import setup


package_name = 'sdu_apex_autodrive'

setup(
    name=package_name,
    version='0.1.0',
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
        'Safe native AutoDRIVE integration boundary for SDU Apex controllers.'
    ),
    license='MIT',
    entry_points={
        'console_scripts': [
            'command_adapter = sdu_apex_autodrive.command_adapter:main',
            'guarded_run = sdu_apex_autodrive.guarded_run:main',
        ],
    },
)
