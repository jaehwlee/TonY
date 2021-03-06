from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import input_data

import json, logging, os, sys, getpass
import tensorflow as tf

FileSystemType = os.environ['FILE_SYSTEM_TYPE']
APP_ID = os.environ['APP_ID']
if FileSystemType == 'LUSTRE':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(BASE_DIR, '..'))
    tf.flags.DEFINE_string('working_dir', '/mnt/lustre/'+APP_ID+'/working_dir', 'Directory under which events and output will be stored in lustre')
    tf.flags.DEFINE_string('data_dir', data_dir + '/input_data/mnist', 'Directory for storing input data')
else:
    tf.flags.DEFINE_string('working_dir', 'hdfs:///master.idpl.org:8020/tmp/melon/'+APP_ID+'/working_dir', 'Directory under which events and output will be sotred in HDFS')
    tf.flags.DEFINE_string('data_dir', './input_data/mnist', 'Directory for storing input data')
tf.flags.DEFINE_integer('steps', 10000, 'The number of training steps to execute.')
tf.flags.DEFINE_integer('batch_size', 64, 'The batch size per step.')

FLAGS = tf.flags.FLAGS

def deepnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def create_model():
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.int64, [None])

    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

    global_step = tf.train.get_or_create_global_step()

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    
    accuracy = tf.reduce_mean(correct_prediction)

    tf.summary.scalar('cross_entropy_loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    return x, y_, keep_prob, global_step, train_step, accuracy, merged

def main(_):
    logging.getLogger().setLevel(logging.INFO)

    fraction = os.environ.get("FRACTION", None)
    cluster_spec_str = os.environ['CLUSTER_SPEC']
    cluster_spec = json.loads(cluster_spec_str)
    print("cluster_spec")
    print(cluster_spec)
    ps_hosts = cluster_spec['ps']
    worker_hosts = cluster_spec['worker']
    
    # 사용할 parameter sever와 worker 정의
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

    # job_name은 ps나 worker 둘 중에 하나
    job_name = os.environ["JOB_NAME"]
    task_index = int(os.environ["TASK_INDEX"])
    server_config = None

    if fraction is not None:
        server_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=float(fraction)))
    else:
        server_config = tf.ConfigProto()
        server_config.gpu_options.allow_growth = True
        
    # worker, ps 모두 클러스터 내 다른 태스크와 통신을 위해 각각의 tensorflow server를 실행
    server = tf.train.Server(cluster, config=server_config, job_name=job_name, task_index=task_index)
    
    # ps는 연산 그래프 작성 안 함, 대신 계산이 수행되는 동안 ps가 종료되지 않도록 함 (listen 상태로 계속 있다고 보면 될 듯)
    if job_name == "ps":
        server.join()
    # worker면 task index에 따라서 별개의 replica device에서 task를 수행할 준비(모델 생성)
    elif job_name == "worker":
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d"%task_index, cluster=cluster)):
            features, labels, keep_prob, global_step, train_step, accuracy, merged = create_model()

            logging.info('==============================in main4')
            # task index가 0이면 chief worker라는 뜻, working dir을 만듦
            if task_index is 0:
                if FileSystemType == 'LUSTRE':
                    os.system('mkdir -p /mnt/lustre/'+APP_ID+'/working_dir')
                    os.system('chmod 777 /mnt/lustre/'+APP_ID+'/working_dir')
                else:
                    os.system('hdfs dfs -mkdir /tmp/melon/'+APP_ID)
                    os.system('hdfs dfs -mkdir /tmp/melon/'+APP_ID+'/working_dir')
                    os.system('hdfs dfs -chmod 777 /tmp/melon/'+APP_ID+'/working_dir')

                logging.info('=========getcwd()'+os.getcwd())

            logging.info('==============================in main befor hooks')
            # 특정 스텝이 지나면 training을 멈추게 함
            # parameters
            # num_steps : 프로그램이 실행하고 나서부터 step만큼 지나야 종료, 즉 chief는 딱 맞춰 끝나지만 나머지는 스텝이 글로벌 스텝이 FLAGS.steps보다 넘어가는 상황 발생
            # last step : worker 모두가 합쳐서 step만큼 진행하고 종료
            hooks = [tf.train.StopAtStepHook(num_steps=FLAGS.steps)]
            logging.info('==============================in main3==='+getpass.getuser())

            config_proto = tf.ConfigProto(device_filters=['/job:ps', '/job:worker/task:%d'%task_index])
            logging.info('==============================in main2')

            config_proto.gpu_options.allow_growth = True
               
            # 분산된 장치에서 학습을 용이하게 해주는 모니터 세션, hook을 통해 학습을 종류시키기도, 에러가 발생하면 세션 복원, 변수 초기화 등을 담당
            with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(task_index==0), hooks=hooks, config=config_proto) as sess:
                logging.info('==============================in main1')
                if FileSystemType != 'LUSTRE':
                    logging.info('loading input data...')
                    os.system('mkdir -p ./input_data/mnist')
                    os.system('hdfs dfs -get /tmp/melon/input_data/mnist/* ./input_data/mnist/')
                mnist = input_data.read_data_sets(FLAGS.data_dir)

                logging.info('Starting training')
                # i는 local step
                i = 0
                while not sess.should_stop():
                    # next_batch의 인자로 shuffle=False를 주면 worker들이 같은 데이터를 학습하게 됨. default value는 True
                    batch = mnist.train.next_batch(FLAGS.batch_size)
                    if i % 1000 == 0:
                        # step은 global_step을 의미
                        step, _, train_accuracy = sess.run([global_step, train_step, accuracy], feed_dict={features: batch[0], labels: batch[1], keep_prob: 1.0})
                        logging.info('Step %d, training accuarcy: %g'%(step, train_accuracy))
                    else:
                        sess.run([global_step, train_step], feed_dict={features: batch[0], labels: batch[1], keep_prob: 0.5})

                    i += 1

            logging.info('Done training!')
            sys.exit()

if __name__ == '__main__':
    tf.app.run()

