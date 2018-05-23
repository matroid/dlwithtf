import numpy as np
np.random.seed(456)
import  tensorflow as tf
tf.set_random_seed(456)
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def pearson_r2_score(y, y_pred):
  """Computes Pearson R^2 (square of Pearson correlation)."""
  return pearsonr(y, y_pred)[0]**2

# Generate synthetic data
d = 1
N = 50
w_true = 5
b_true = 2
noise_scale = .1
x_np = np.random.rand(N, d)
noise = np.random.normal(scale=noise_scale, size=(N, d))
y_np = np.reshape(w_true * x_np  + b_true + noise, (-1))

# Save image of the data distribution
plt.scatter(x_np, y_np)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Raw Linear Regression Data")
plt.savefig("fcnet_regression_data.png")

# Generate tensorflow graph
n_hidden = 15
with tf.name_scope("placeholders"):
  x = tf.placeholder(tf.float32, (N, d))
  y = tf.placeholder(tf.float32, (N,))
with tf.name_scope("layer-1"):
  W = tf.Variable(tf.random_normal((d, n_hidden)))
  b = tf.Variable(tf.random_normal((n_hidden,)))
  x_1 = tf.nn.relu(tf.matmul(x, W) + b)
with tf.name_scope("output"):
  W = tf.Variable(tf.random_normal((n_hidden, 1)))
  b = tf.Variable(tf.random_normal((1,)))
  y_pred = tf.transpose(tf.matmul(x_1, W) + b)
with tf.name_scope("loss"):
  lvec = (y - y_pred)**2
  l = tf.reduce_sum(lvec)
with tf.name_scope("optim"):
  train_op = tf.train.AdamOptimizer(.001).minimize(l)

with tf.name_scope("summaries"):
  tf.summary.scalar("loss", l)
  merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('/tmp/fcnet-regression-train', tf.get_default_graph())

n_steps = 1000
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # Train model
  for i in range(n_steps):
    feed_dict = {x: x_np, y: y_np}
    _, summary, loss, lossvec = sess.run([train_op, merged, l, lvec], feed_dict=feed_dict)
    print("step %d, loss: %f, loss-vec-size: %s" % (i, loss, lossvec.shape))
    train_writer.add_summary(summary, i)

  # Make Predictions
  y_pred_np = sess.run(y_pred, feed_dict={x: x_np})

y_pred_np = np.reshape(y_pred_np, -1)
r2 = pearson_r2_score(y_np, y_pred_np)
print("Pearson R^2: %f" % r2)

# Clear figure
plt.clf()
plt.xlabel("Y-true")
plt.ylabel("Y-pred")
plt.title("Predicted versus true values")
plt.scatter(y_np, y_pred_np)
plt.savefig("fcnet_regression_pred.png")

# Now draw with learned regression line
plt.clf()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Predicted versus true values")
plt.xlim(0, 1)
plt.scatter(x_np, y_np)
plt.scatter(x_np, y_pred_np)
plt.savefig("fcnet_regression_learned.png")
