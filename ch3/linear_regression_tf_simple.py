import  tensorflow as tf

d = 10
N = 100

x = tf.placeholder(tf.float32, (N, d))
y = tf.placeholder(tf.float32, (N,))
W = tf.Variable(tf.random_normal((d, 1)))
b = tf.Variable(tf.random_normal((1,)))
l = tf.reduce_sum((y - (tf.matmul(x, W) + b))**2)

with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)

