# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

# Reference
# 한국어, 나름 비슷, https://excelsior-cjh.tistory.com/162
# 영어로 된 앞 부분 개념, https://databricks.com/tensorflow/distributed-computing-with-tensorflow
# 영어로 된 앞 부분 개념, 좀 더 자세하고 친절, https://medium.com/@alienrobot/understanding-distributed-tensorflow-2cdbd9881d9b
# 제일 자세한 영어 ppt, https://www.math.purdue.edu/~nwinovic/slides/Getting_Started_with_TensorFlow_II.pdf

# keywords
# parameter servers : worker랑 똑같음. 보통 worker에 필요한 변수 저장하는 cpu임, 가중치 같은 거, 태스크임 이것도
# workers : 계산하는 친구
# forward pass : worker가 ps로부터 변수를 가져오는 것
# backward pass : worker가 계산해서 ps에게 새로운 가중치를 주는 것
# chief : checkpoints를 핸들링하는 태스크, 모델 저장 등 

"""A deep MNIST classifier using convolutional layers.

This example was adapted from
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py.

Each worker reads the full MNIST dataset and asynchronously trains a CNN with dropout and using the Adam optimizer,
updating the model parameters on shared parameter servers.

The current training accuracy is printed out after every 100 steps.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

import json
import logging
import os
import sys

import tensorboard.program as tb_program
import tensorflow as tf

# Environment variable containing port to launch TensorBoard on, set by TonY.
TB_PORT_ENV_VAR = 'TB_PORT'

# Input/output directories
tf.flags.DEFINE_string('data_dir', '/tmp/tensorflow/mnist/input_data',
                       'Directory for storing input data')
tf.flags.DEFINE_string('working_dir', '/tmp/tensorflow/mnist/working_dir',
                       'Directory under which events and output will be '
                       'stored (in separate subdirectories).')

# Training parameters
tf.flags.DEFINE_integer("steps", 1500,
                        "The number of training steps to execute.")
tf.flags.DEFINE_integer("batch_size", 64, "The batch size per step.")

FLAGS = tf.flags.FLAGS


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.

    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.

    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def create_model():
    """Creates our model and returns the target nodes to be run or populated"""
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.int64, [None])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_,
                                                               logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)
    
    # global step tensor초기화, optimizer에서 값을 업데이트할 때마다 하나씩 증가함
    global_step = tf.train.get_or_create_global_step()
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,
                                                           global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    tf.summary.scalar('cross_entropy_loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    return x, y_, keep_prob, global_step, train_step, accuracy, merged


def start_tensorboard(logdir):
    tb = tb_program.TensorBoard()
    port = int(os.getenv(TB_PORT_ENV_VAR, 6006))
    tb.configure(logdir=logdir, port=port)
    tb.launch()
    logging.info("Starting TensorBoard with --logdir=%s" % logdir)


def main(_):
    logging.getLogger().setLevel(logging.INFO)

    # CLUSTER_SPEC이라는 환경변수를 통해 ps_hosts, worker_hosts에 파라미터 서버와 워커 정보를 저장
    cluster_spec_str = os.environ["CLUSTER_SPEC"]
    cluster_spec = json.loads(cluster_spec_str)
    ps_hosts = cluster_spec['ps']
    worker_hosts = cluster_spec['worker']
    
    # 핵심이 되는 부분. 우리가 사용할 파라미터 서버와 워커를 정의하는 부분. 음 얼마나 있는지 내가 알 길이 없네!
    # 저장된 정보를 바탕으로 클러스터 생성, https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    
    # Create and start a server for the local task.
    # job name은 worker거나 server.
    # server는 각 ps와 worker 모두 정의해야 함!
    # 태스크는 예시로 2개의 ps와 5개의 worker로 구성. ps와 worker의 역할을 job이라고 함
    # 각 태스크는 텐서플로 서버를 실행하며, 계산을 위해 자원을 사용하고 병렬처리를 효율적으로 하도록 클러스터 내 다른 태스크와 통신을 위해 server running
    # 워커 노드에 클러스터를 정의, 서버도 정의
    job_name = os.environ["JOB_NAME"]
    task_index = int(os.environ["TASK_INDEX"])
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    # parameter server면 join만 해줌
    # ps에는 모델(연산 그래프) 작성 안 함, 대신 프로세스가 종료되지 않도록 server.join()으로 병렬 계산이 수행되는 동안 ps가 종료되지 않도록 함
    if job_name == "ps":
        server.join()
    # worker면 task index에 따라서 별개의 디바이스에서 태스크 수행할 준비 (모델 생성)
    elif job_name == "worker":
        # Create our model graph. Assigns ops to the local worker by default.
        # replica를 통해 각 태스크에 모델을 복제. 인자로 클러스터 태스크 설정
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):
            features, labels, keep_prob, global_step, train_step, accuracy, \
            merged = create_model()

        if task_index is 0:  # chief worker
          # chief worker은 session을 준비한다. 그동안 다른 워커는 준비될 때까지 기다림
            tf.gfile.MakeDirs(FLAGS.working_dir)
            start_tensorboard(FLAGS.working_dir)

        # The StopAtStepHook handles stopping after running given steps.
        # https://www.tensorflow.org/api_docs/python/tf/estimator/StopAtStepHook
        # 특정 스텝이 지나면 training을 멈추게 하는 일종의 플래그 같은 친구네
        # num_steps과 last_step 둘 중에 하나가 인자로 들어가야 함
        # num_steps를 하면 begin()이라는 메소드가 호출되고 FLAGS.steps가 300 지나야 종료되는 건데,
        # worker마다 begin을 하는 시점이 달라, 어떤 워커는 mnist 갖고 오느라 109(글로벌)스텝에서 시작할 수도 있어.
        # 그러면 이 워커는 409에서 종료가 되는 거고, 0번에서 begin한 애는 내가 원하는 300에서 종료하는 거지.
        # last_step이 내가 생각하는 그거야, 늦게 시작하던 말던 글로벌이 steps에 딱 되면 둘 다 종료되는 거지.
        # 그래서 예를 들어 300이면 0번 워커는 210번, 1번 워커는 90번 돌고 종료가 
        hooks = [tf.train.StopAtStepHook(num_steps=FLAGS.steps)]

        # Filter all connections except that between ps and this worker to
        # avoid hanging issues when one worker finishes. We are using
        # asynchronous training so there is no need for the workers to
        # communicate.
        # 세션은 device_filters가 아닌 디바이스들은 다 무시
        config_proto = tf.ConfigProto(
            device_filters=['/job:ps', '/job:worker/task:%d' % task_index])
        
        # is_chief : 메인 노드인가? 얘가 모든 클러스터를 관리하는 대장이야
        # 세션이란 실제로 계산이 이뤄지면서 그래프가 동작하는 친구, 근데 이 세션은 특이한 게 분산을 지원하는 세션인 거지
        # 초기화, 리커버리, 훅을 조절하는 session과 같은 object
        # 훅 사용, 에러 발생하면 세션 복원, 변수 초기화 편리하게 해줌
        # 체크포인트와 서머리 저장을 자동화시키는 세션 정의, 분산된 장치에서 학습을 용이하게 해주는 세션 
        # working_dir에 저장, 한 번 모니터 세션이 초기화되면 그래프는 동결돼서 못 수정함, 그래서 모델이랑 글로벌 스텝 먼저 정의해야 함
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index == 0),
                                               checkpoint_dir=FLAGS.working_dir,
                                               hooks=hooks,
                                               config=config_proto) as sess:
            # Import data
            logging.info('Extracting and loading input data...')
            mnist = input_data.read_data_sets(FLAGS.data_dir)

            # Train
            logging.info('Starting training')
            i = 0
            # 세션 종료 조건 전까지 계속 되어야 함, hook이 이 상태를 결정함, 어떻게? run_context.request_stop()메소드를 통해 중지
            while not sess.should_stop():
                batch = mnist.train.next_batch(FLAGS.batch_size)
                if i % 100 == 0:
                    step, _, train_accuracy = sess.run(
                        [global_step, train_step, accuracy],
                        feed_dict={features: batch[0], labels: batch[1],
                                   keep_prob: 1.0})
                    logging.info('Step %d, training accuracy: %g' % (
                    step, train_accuracy))
                else:
                    sess.run([global_step, train_step],
                             feed_dict={features: batch[0], labels: batch[1],
                                        keep_prob: 0.5})
                i += 1

        logging.info('Done training!')
        sys.exit()


if __name__ == '__main__':
    tf.app.run()

    
    
'''
@tf_export(v1=["train.StopAtStepHook"])
class StopAtStepHook(session_run_hook.SessionRunHook):
  """Hook that requests stop at a specified step."""

  def __init__(self, num_steps=None, last_step=None):
    """Initializes a `StopAtStepHook`.
    This hook requests stop after either a number of steps have been
    executed or a last step has been reached. Only one of the two options can be
    specified.
    if `num_steps` is specified, it indicates the number of steps to execute
    after `begin()` is called. If instead `last_step` is specified, it
    indicates the last step we want to execute, as passed to the `after_run()`
    call.
    Args:
      num_steps: Number of steps to execute.
      last_step: Step after which to stop.
    Raises:
      ValueError: If one of the arguments is invalid.
    """
    if num_steps is None and last_step is None:
      raise ValueError("One of num_steps or last_step must be specified.")
    if num_steps is not None and last_step is not None:
      raise ValueError("Only one of num_steps or last_step can be specified.")
    self._num_steps = num_steps
    self._last_step = last_step

  def begin(self):
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError("Global step should be created to use StopAtStepHook.")

  def after_create_session(self, session, coord):
    if self._last_step is None:
      global_step = session.run(self._global_step_tensor)
      self._last_step = global_step + self._num_steps

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    global_step = run_values.results + 1
    if global_step >= self._last_step:
      # Check latest global step to ensure that the targeted last step is
      # reached. global_step read tensor is the value of global step
      # before running the operation. We're not sure whether current session.run
      # incremented the global_step or not. Here we're checking it.

      step = run_context.session.run(self._global_step_tensor)
      if step >= self._last_step:
        run_context.request_stop()
'''
