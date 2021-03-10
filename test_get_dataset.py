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

import glob2
import yaml
import source.yaml_utils as yaml_utils
from gen_models.ada_generator import AdaBIGGAN, AdaSNGAN
from dis_models.patch_discriminator import PatchDiscriminator
from updater import Updater

# Modified to load the dataset specified via command line.
def get_dataset(image_size, config, dataset, rotate):
    # return an array of image shaped (config.datasize, 3, image_size, image_size)

    if dataset:
        # please define your dataset here if necessary
        import cv2
        print("Use rotations?: %s" % str(rotate))
        print("Dataset being used: %s" % dataset)


        img_dir = Path(f"{config.data_path}/{dataset}/**/*.png")
        print("Data location: ", img_dir)

        # Get all images in the chosen dataset directory.
        img_path = list(glob2.glob(str(img_dir)))

        img = []
        for i in range(len(img_path)):  # Normally 100, but try all images.
            if "_r" in str(img_path[i]) and not rotate:
                continue
            img_ = cv2.imread(str(img_path[i]))[:, :, ::-1]
            h, w = img_.shape[:2]
            size = min(h, w)
            img_ = img_[(h - size) // 2:(h - size) // 2 + size, (w - size) // 2:(w - size) // 2 + size]
            img.append(cv2.resize(img_, (image_size, image_size)))
        # This arranges the array so it has dimensions: N x C x H x W
        img = np.array(img).transpose(0, 3, 1, 2)
        print(img.shape)

        img = img.astype("float32") / 127.5 - 1


    # default dataset
    # images in {config.data_path}/{config.dataset} directory are loaded
    else:
        import cv2
        img_path = Path(f"{config.data_path}/{config.dataset}")
        img_path = list(img_path.glob("*"))[:config.datasize]
        img = []
        for i in range(config.datasize):
            img_ = cv2.imread(str(img_path[i]))[:, :, ::-1]
            h, w = img_.shape[:2]
            size = min(h, w)
            img_ = img_[(h - size) // 2:(h - size) // 2 + size, (w - size) // 2:(w - size) // 2 + size]
            img.append(cv2.resize(img_, (image_size, image_size)))
        img = np.array(img).transpose(0, 3, 1, 2)
        img = img.astype("float32") / 127.5 - 1

    
    print("Number of images to use: %i" % len(img))

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="configs/default.yml")
    # parser.add_argument("--resume", "-r", type=str, default="")
    parser.add_argument("--communicator", type=str, default="hierarchical")
    parser.add_argument("--suffix", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    # Specify dataset as command line argument.
    parser.add_argument("--dataset", type=str, default="anime")

    args = parser.parse_args()
    now = int(time.time()) * 10 + args.suffix
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    os.makedirs(f"{config.save_path}{now}", exist_ok=True)
    #shutil.copy(args.config_path, f"{config.save_path}{now}/config{now}.yml")
    #shutil.copy("./train.py", f"{config.save_path}{now}/train.py")
    print("snapshot->", now)

    # image size
    config.image_size = config.image_sizes[config.gan_type]
    image_size = config.image_size

    

    np.random.seed(1234)

    rotate = False
    img = np.array(get_dataset(image_size, config, args.dataset, rotate))


    