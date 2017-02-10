import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float64, [None, None, 8] )
X_lengths = tf.placeholder(tf.int32, [None] )

cell1 = tf.nn.rnn_cell.BasicRNNCell(num_units=4)
outputs, last_states = tf.nn.dynamic_rnn(
    			cell = cell1,
    			dtype = tf.float64,
    			sequence_length = X_lengths,
    			inputs = X
			)

# Create input data
x = np.random.randn(6, 5, 8)		
x_lengths = [4, 3, 4, 5, 5, 4]

with tf.Session() as sess:
	sess.run( tf.initialize_all_variables() )
	out, sta = sess.run([outputs, last_states], {X:x, X_lengths:x_lengths} )
	print 'outputs', out
 	print 'final_states', sta



