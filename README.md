# EveNet: Expression of Emotion and Visimes Network

[![Build Status](https://travis-ci.org/elggem/EveNet.svg?branch=master)](https://travis-ci.org/elggem/EveNet)

This is a fork of ibab's excellent implementation of WaveNet. Here we are implementing changes for the generation of facial animations.

## Requirements

TensorFlow needs to be installed before running the training script.
Code is tested on TensorFlow version 1.0.1 for Python 2.7 and Python 3.5.

In addition, [librosa](https://github.com/librosa/librosa) must be installed for reading and writing audio.

To install the required python packages, run
```bash
pip install -r requirements.txt
```

For GPU support, use
```bash
pip install -r requirements_gpu.txt
```

## Running tests

Install the test requirements
```
pip install -r requirements_test.txt
```

Run the test suite
```
./ci/test.sh
```

## Related projects

- [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet), the WaveNet implementation this is based on.
