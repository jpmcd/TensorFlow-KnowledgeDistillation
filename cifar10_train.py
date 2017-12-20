# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#from tensorflow.models.image.cifar10 import cifar10
import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/scratch/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

IMAGE_SIZE = 24


def resize_image_with_crop_or_pad(images, target_height, target_width):
  b, original_height, original_width, _ = images.shape

  if target_width <= 0:
    raise ValueError('target_width must be > 0.')
  if target_height <= 0:
    raise ValueError('target_height must be > 0.')

  offset_crop_width = 0
  if target_width < original_width:
    offset_crop_width = (original_width - target_width) // 2

  offset_crop_height = 0
  if target_height < original_height:
    offset_crop_height = (original_height - target_height) // 2

  top = offset_crop_height
  bot = offset_crop_height + target_height
  lef = offset_crop_width
  rig = offset_crop_width + target_width

  return images[:,top:bot,lef:rig,...]


def per_image_whitening(images):
  stddev = np.std(images, axis=(1,2,3), keepdims=True)
  adj_std = np.maximum(stddev, 1./np.sqrt(np.prod(images.shape[1:])))
  mean = np.mean(images, axis=(1,2,3), keepdims=True)

  return (images - mean)/adj_std


def preprocess():
  target_height = IMAGE_SIZE
  target_width = IMAGE_SIZE

  concat = []

  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-py')
  filenames = [os.path.join(data_dir, 'data_batch_%d' % i)
                 for i in xrange(1, 6)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

    with open(f, 'rb') as fo:
      dict = np.load(fo)

    raw_images = dict['data']

    # Data is 10000x3072
    # Reshape to [batch,depth,height,width] and transpose to [batch,height,width,depth]
    raw_images = raw_images.reshape((10000,3,32,32)).transpose((0,2,3,1))

    # Crop the central [height, width] of the image.
    resized_images = resize_image_with_crop_or_pad(raw_images, target_height, target_width)

    # Cast from uint8 to float32
    float_images = resized_images.astype('float32')

    # Subtract off the mean and divide by the variance of the pixels.
    images = per_image_whitening(float_images)

    # Append to numpy array
    concat.append(images)

  images_set = np.concatenate(concat, axis=0)
  images_path = os.path.join(FLAGS.data_dir, 'img.npz')
  np.savez_compressed(images_path, images_set=images_set)
  

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    with tf.variable_scope('model') as m_scope:
      logits = cifar10.inference(images)

      # Calculate loss.
      loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                            graph_def=sess.graph_def)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f, (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        
        # Update logits for student model
        print("Computing new logits")
        images_path = os.path.join(FLAGS.data_dir, 'img.npz')
        logits_path = os.path.join(FLAGS.train_dir, 'log.npz')
        with np.load(images_path) as data:
          images_set = data['images_set']
        #logits_set = sess.run(logits,feed_dict={images:images_set})
###################################
        batch_size = 500
        NUM_BATCHES = int(len(images_set)/batch_size)
        concat = []
        for i in range(NUM_BATCHES):
          print("\r%d/%d"%(i, NUM_BATCHES), end="")
          sys.stdout.flush()
          logits_set = sess.run(logits,
            feed_dict={images:images_set[i*batch_size:(i+1)*batch_size]})
          concat.append(logits_set)
        logits_set = np.concatenate(concat, axis=0)
        print(logits_set.shape)
        np.savez_compressed(logits_path, logits_set=logits_set)
        print("\nFinished computing new logits")


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  images_path = os.path.join(FLAGS.data_dir, 'img.npz')
  if not os.path.exists(images_path):
    print("Preprocessing image data")
    preprocess()
  train()


if __name__ == '__main__':
  tf.app.run()
