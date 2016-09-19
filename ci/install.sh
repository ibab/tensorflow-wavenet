#!/bin/sh

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
conda update -q conda

conda create -q -n test python=$TRAVIS_PYTHON_VERSION numpy scipy
source activate test
pip install nose

pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-$TENSORFLOW-cp35-cp35m-linux_x86_64.whl

