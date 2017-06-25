"""
Code for TF tutorial MNIST for ML Beginners (with summaries!)
(in case of jupyter notebook related problems with GPU memory)

Adapted from:
https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py

Instructions:
  1. Run the script (fingers crossed!)
  2. In the repo's dir, run:
    > tensorboard --logdir=MNIST_logs/
  3. Open a web browser at http://0.0.0.0:6006/
  4. Enjoy!
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = 'MNIST_data/'
LOG_DIR = 'MNIST_logs/'
LEARNING_RATE = 0.05
MAX_STEPS = 1000

def train():
    """MNIST basic model training."""

    def variable_summaries(var):
        """
        Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        """
        # A name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

    sess = tf.InteractiveSession()

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y_-input')

    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]))
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]))
        variable_summaries(b)

    with tf.name_scope('y'):
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        tf.summary.histogram('y', y)
    tf.summary.histogram('y_', y_)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                                      reduction_indices=[1]))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(
            LEARNING_RATE).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test')
    tf.global_variables_initializer().run()

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data & add training summaries

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            batch_xs, batch_ys = mnist.train.next_batch(100)
        else:
            batch_xs, batch_ys = mnist.test.images, mnist.test.labels
        return {x: batch_xs, y_: batch_ys}

    for i in range(MAX_STEPS):
        if i % 10 == 0:  # Record summaries and test-set accuracy

            summary, acc = sess.run([merged, accuracy],
                                    feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
          if i % 100 == 99:  # Record execution stats
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step],
                                  feed_dict=feed_dict(True),
                                  options=run_options,
                                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            print('Adding run metadata for', i)
          else:  # Record a summary
            summary, _ = sess.run([merged, train_step],
                                  feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()

    print(sess.run(accuracy,
                   feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


def main(_):
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
    train()

if __name__ == '__main__':
    tf.app.run(main=main)


# [Jose]: With GPU, needed to add /Developer/NVIDIA/CUDA-8.0/extras/CUPTI/lib/
#   to DYLD_LIBRARY_PATH (or equivalent solution for Linux) to avoid crash.
#   See https://github.com/tensorflow/tensorflow/issues/8830 for more details.
#
# Output extract (may be different):
#  Accuracy at step 0: 0.098
#  Accuracy at step 10: 0.5811
#  Accuracy at step 20: 0.7032
#  Accuracy at step 30: 0.7869
#  Accuracy at step 40: 0.8046
#  ...
#  Accuracy at step 960: 0.8979
#  Accuracy at step 970: 0.899
#  Accuracy at step 980: 0.8989
#  Accuracy at step 990: 0.8999
#  Adding run metadata for 999
#  0.9
