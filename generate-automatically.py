import numpy as np
import os

logdir = "logdir/train/2times128-fw8-6464"
trailname = "6_4_optokoppler_unbound"
command = "python generate.py --samples 6500 --wav_out_path ../trails/{:s}-{:d} --fast_generation false {:s}/model.ckpt-{:d}"

for i in xrange(10):
    steps = (i+1)*10000
    print(command.format(trailname, steps, logdir, steps))
    os.system(command.format(trailname, steps, logdir, steps))
    
