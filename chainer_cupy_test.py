import os, time
from pathlib import Path
import shutil
import numpy as np
import argparse
import chainer
from chainer import cuda
from chainer.links import VGG16Layers as VGG
from chainer.training import extensions
import chainermn

import yaml
import source.yaml_utils as yaml_utils
from gen_models.ada_generator import AdaBIGGAN, AdaSNGAN
from dis_models.patch_discriminator import PatchDiscriminator
from updater import Updater
print(os.system(". ~/.bashrc"))
print(chainer.backends.cuda.available)
print(chainer.backends.cuda.cudnn_enabled)