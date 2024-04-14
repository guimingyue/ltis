import os 
from setuptools import setup, find_packages
from setuptools.command.install import install


cwd = os.path.dirname(os.path.abspath(__file__))

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)

setup(
    name='ltis',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=reqs,
    package_data={
        '': ['*.txt', 'cmudict_*'],
    },
    entry_points={
        "console_scripts": [
            "ltis = src.app:main",
        ],
    },
)
