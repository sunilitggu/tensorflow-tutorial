import tensorflow as tf
import numpy as np

W1 = tf.convert_to_tensor(np.random.rand(4,3) )
W2 = tf.Variable(tf.ones((3,5), dtype='float64', name = 'weights') )

A = tf.matmul(W1, W2)

with tf.Session() as sess:
	print (sess.run(W1) )

	sess.run( tf.initialize_all_variables() )

	print (sess.run(W2) )

	print (sess.run(A))


