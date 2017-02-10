
import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(4.0)

c = a*b

with tf.Session() as sess:

	print( sess.run(c) )
	print( c.eval() )

