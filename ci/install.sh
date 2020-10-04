#!/usr/bin/env bash

wget --no-check-certificate \
    https://repo.anaconda.com/miniconda/Miniconda2-4.7.12.1-Linux-x86_64.sh -O miniconda.sh

chmod +x miniconda.sh && ./miniconda.sh -b -p ${HOME}/miniconda

export PATH="${HOME}/miniconda/bin:/usr/lib/llvm-7/bin:${PATH}"

conda config --set always_yes yes --set changeps1 no

conda create -q -n test${TRAVIS_PYTHON_VERSION} python=${TRAVIS_PYTHON_VERSION}
source activate test${TRAVIS_PYTHON_VERSION}

python -m pip install pip==19.3.1

python -m pip install -r requirements_test.txt
