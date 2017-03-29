import numpy as np
np.random.seed(456)
import  tensorflow as tf
tf.set_random_seed(456)

d = 1
N = 10

w_true = 5
b_true = 2
noise_scale = .1
x_np = np.random.rand(N, d)
noise = np.random.normal(scale=noise_scale, size=(N, d))
y_np = np.reshape(w_true * x_np  + b_true + noise, (-1))


with tf.name_scope("placeholders"):
  x = tf.placeholder(tf.float32, (N, d))
  y = tf.placeholder(tf.float32, (N,))
with tf.name_scope("weights"):
  W = tf.Variable(tf.random_normal((d, 1)))
  b = tf.Variable(tf.random_normal((1,)))
with tf.name_scope("loss"):
  l = tf.reduce_sum((y - (tf.matmul(x, W) + b))**2)
with tf.name_scope("optim"):
  train_op = tf.train.AdamOptimizer(.01).minimize(l)

with tf.name_scope("summaries"):
  tf.summary.scalar("loss", l)
  merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('/tmp/lr-train', tf.get_default_graph())

n_steps = 1000
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(n_steps):
    feed_dict = {x: x_np, y: y_np}
    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
    print("loss: %f" % loss)
    train_writer.add_summary(summary, i)
