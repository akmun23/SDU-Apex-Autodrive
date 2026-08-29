import os

from setuptools import setup

package_name = 'f1tenth_stack'

runtime_launch_files = [
    'launch/mapping.launch.py',
    'launch/localization_nav2.launch.py',
    'launch/perception.launch.py',
]

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), runtime_launch_files),
        (os.path.join('share', package_name, 'config'), [
            'config/slam_params.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aksel Munksgaard-Ottosen',
    maintainer_email='akmun23@student.sdu.dk',
    description='AutoDRIVE mapping, localization, and perception orchestration.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
