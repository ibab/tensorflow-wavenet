from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['pandas', 'numpy', 'termcolor']

setup(name='trainer',
      version='0.1',
      description='EveNet Trainer for GCloud Engine',
      url='http://github.com/elggem/EveNet',
      author='Ralf Mayet',
      author_email='ralf.mayet@mindcloud.ai',
      license='Unlicense',
      packages=find_packages(),
      include_package_date=True)
