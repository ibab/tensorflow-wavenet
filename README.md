# A TensorFlow implementation of DeepMind's WaveNet paper

This is a TensorFlow implementation of the [WaveNet generative neural
network architecture](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) for audio generation.

The WaveNet architecture directly produces a raw audio waveform as its output,
and shows excellent results in TTS and general audio generation (see the
DeepMind blog post and paper for examples).

The network is a model of the conditional probability to generate the next
sample in the audio waveform, given all previous samples and possibly
additional parameters.
It is constructed from a number of *causal dilated layers*, each of which is a
dilated convolution with a filter of width 2.

This implementation is currently still under :construction:

## Training the network

The [VCTK corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) is currently used.
In order to train the network, you need to download the corpus and unpack it in the same directory as the `run.py` script.

Then, execute
```python
python run.py
```
to train the network.

**Disclaimer:** This repository is not affiliated with DeepMind or Google.

