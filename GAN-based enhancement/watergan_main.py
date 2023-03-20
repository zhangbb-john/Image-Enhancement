import os
import numpy as np
import tensorflow as tf

from absl import flags,app
from utils import pp
from ModelGAN import WGAN
## define flags
FLAGS = flags.FLAGS
# optimizer
flags.DEFINE_integer("epoch", 15, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta_1", 0.5, "Momentum term of adam [0.5]")

#train parameter
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
                                                                    # flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 256, "The size of batch images [64]")
flags.DEFINE_integer("train_size", 10000, "The size of train images [np.inf]")

# image size
flags.DEFINE_boolean("is_crop", True, "True for needing resizing , False for otherwise [False]")
flags.DEFINE_integer("input_height", 120, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 160, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("input_water_height", 1024, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_water_width", 1360, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 120, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 160, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_float("max_depth",10, "maximum of depth. [3.0]")
# dataset$pattern
flags.DEFINE_string("water_dataset", "water_images", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("air_dataset","nyu_images","The name of dataset with air images")
flags.DEFINE_string("depth_dataset","nyu_depth","The name of dataset with depth images")
flags.DEFINE_string("test_dataset","nyu_images","The name of dataset with depth images")
flags.DEFINE_string("test_depth","nyu_depth","The name of dataset with depth images")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
# sample parameters
flags.DEFINE_integer("num_samples",2000, "True for visualizing, False for nothing [4000]")
flags.DEFINE_integer("sample_output_height", 480, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("sample_output_width", 640, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
# save parameter
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("results_dir", "results", "Directory name to save the checkpoints [results]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("back_dir", "back", "Directory name to save the image samples [samples]")
flags.DEFINE_integer("save_epoch",25, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("ckpt_path","./save/", "checkpoint")
flags.DEFINE_string("ckpt_path2","./save", "checkpoint")

def main(_):

    if not os.path.exists(FLAGS.ckpt_path):
        os.makedirs(FLAGS.ckpt_path)
    # pp.pprint(flags.FLAGS.__flags)
    print(FLAGS)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.results_dir):
        os.makedirs(FLAGS.results_dir)
    if not os.path.exists(FLAGS.back_dir):
        os.makedirs(FLAGS.back_dir)
    wgan = WGAN(
        # train parameter
        batch_size=FLAGS.batch_size,

        # image size
        is_crop=FLAGS.is_crop,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        input_water_width=FLAGS.input_water_width,
        input_water_height=FLAGS.input_water_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        c_dim=FLAGS.c_dim,
        max_depth = FLAGS.max_depth,
        # dataset$pattern
        water_dataset_name=FLAGS.water_dataset,
        air_dataset_name=FLAGS.air_dataset,
        depth_dataset_name=FLAGS.depth_dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        test_dataset_name=FLAGS.test_dataset,
        test_depth_name=FLAGS.test_depth,
        #sample parameters
        num_samples=FLAGS.num_samples,
        sample_output_height=FLAGS.sample_output_height,
        sample_output_width=FLAGS.sample_output_width,
        #save parameter

        save_epoch=FLAGS.save_epoch,
        checkpoint_dir=FLAGS.checkpoint_dir,
        results_dir = FLAGS.results_dir,
        sample_dir=FLAGS.sample_dir,
        back_dir=FLAGS.back_dir,
        ckpt_path= FLAGS.ckpt_path,
        ckpt_path2= FLAGS.ckpt_path2
        )
    wgan.train(FLAGS)

if __name__ == '__main__':
  app.run(main)
