import tensorflow as tf

s = tf.constant(0, name='counter')
ns = tf.add(s, tf.constant(1) )
#update_s = tf.assign(s, ns)
s = ns 

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables() )
	print sess.run(s)
#	sess.run(update_s)
	print sess.run(ns)

 
