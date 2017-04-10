import numpy as np
import os

logdir = "logdir/train/2times128-fw16"
trailname = "6_4_3232_optokoppler-fromseed"

seed = "./data/new_data/confident_L_03_4min.csv"
command = "python generate.py --samples 6500 --wav_out_path ../trails/{:s}-{:d} --fast_generation false --bound {:s} --wav_seed {:s} {:s}/model.ckpt-{:d}"

#for i in xrange(10):
#    steps = (i+1)*10000
    #os.system(command.format(trailname+"_unbound", steps, "false", seed, logdir, steps))
    #os.system(command.format(trailname+"_bound", steps, "true", seed, logdir, steps))

steps = 100000
os.system(command.format(trailname+"_unbound", steps, "false", seed, logdir, steps))
os.system(command.format(trailname+"_bound", steps, "true", seed, logdir, steps))
