from pathlib import Path

from setuptools import find_packages, setup

package_name = 'tb4_voice_nav'

data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

for asset_directory in ('maps', 'config', 'launch'):
    files_by_destination = {}
    for path in Path(asset_directory).rglob('*'):
        if path.is_file():
            destination = str(Path('share') / package_name / path.parent)
            files_by_destination.setdefault(destination, []).append(str(path))
    data_files.extend(
        (destination, files)
        for destination, files in files_by_destination.items()
    )

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='engquan5',
    maintainer_email='engquan5@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'map_segmenter = tb4_voice_nav.map_segmenter:main',
            'voice_commander = tb4_voice_nav.voice_commander:main',
            'semantic_navigator = tb4_voice_nav.semantic_navigator:main',
        ],
    },
)
