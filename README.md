# A TensorFlow implementation of DeepMind's WaveNet paper

This is a TensorFlow implementation of the [WaveNet generative neural
network architecture](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) for audio generation.

<table style="border-collapse: collapse">
<tr>
<td>
<p>
The WaveNet neural network architecture directly generates a raw audio waveform,
showing excellent results in text-to-speech and general audio generation (see the
DeepMind blog post and paper for details).
</p>
<p>
The network models the conditional probability to generate the next
sample in the audio waveform, given all previous samples and possibly
additional parameters.
</p>
<p>
After an audio preprocessing step, the input waveform is quantized to a fixed integer range.
The integer amplitudes are then one-hot encoded to produce a tensor of shape <code>(num_samples, num_channels)</code>.
</p>
<p>
A convolutional layer that only accesses the current and previous inputs then reduces the channel dimension.
</p>
<p>
The core of the network is constructed as a stack of <em>causal dilated layers</em>, each of which is a
dilated convolution (convolution with holes), which only accesses the current and past audio samples.
</p>
<p>
The outputs of all layers are combined and extended back to the original number
of channels by a series of dense postprocessing layers.
</p>
<p>
The loss function is the cross-entropy between the output for each timestep and the input at the next timestep.
</p>
<p>
In this repository, the network implementation can be found in <a href="./wavenet.py">wavenet.py</a>.
</p>
</td>
<td width="300">
<img src="images/network.png" width="300"></img>
</td>
</tr>
</table>

## Requirements

TensorFlow needs to be installed before running the training script.
TensorFlow 0.10 and the current `master` version are supported.

In addition, the `ffmpeg` binary needs to be available on the command line.
It is needed by the TensorFlow ffmpeg contrib package that is used to decode the audio files.

## Training the network

The [VCTK corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) is currently used.
In order to train the network, you need to download the corpus and unpack it in the same directory as the `train.py` script.

Then, execute
```bash
python train.py
```
to train the network.

You can see documentation on the settings by by running
```bash
python train.py --help
```

## Generating audio

You can use the `generate.py` script to generate audio using a previously trained model.

Run
```
python generate.py --samples 16000 model.ckpt-1000
```
where `model.ckpt-1000` needs to be a previously saved model.
You can find these in the `logdir`.

The generated waveform can be played back using TensorBoard.

## Missing features

Currently, there is no conditioning on extra information like the speaker ID.

