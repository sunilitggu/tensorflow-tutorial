
import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, [3, None], name='X' )

W = tf.Variable( tf.random_uniform( [2,3], -1.0, +1.0) , name = 'W' )

z1 = tf.matmul(X, W)
z2 = tf.nn.sigmoid(z1)

with tf.Session() as sess:
	sess.run( tf.initialize_all_variables() )
	a, b = sess.run([z1,z2], {X:np.ones((3,2))} )
	print a
	print b








