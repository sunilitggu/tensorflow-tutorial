import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float64, [None, None, 8] )
X_lengths = tf.placeholder(tf.int32, [None] )

cell1 = tf.nn.rnn_cell.BasicRNNCell(num_units=4)
cell2 = tf.nn.rnn_cell.BasicRNNCell(num_units=4)
outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
    			cell_fw = cell1,
			cell_bw = cell2,
    			dtype = tf.float64,
    			sequence_length = X_lengths,
    			inputs = X
			)

output_fw, output_bw = outputs
output_state_fw, output_state_bw = last_states

# Create input data
x = np.random.randn(6, 5, 8)		
x_lengths = [4, 3, 4, 5, 5, 4]

with tf.Session() as sess:
	sess.run( tf.initialize_all_variables() )
	out_fw, out_bw, sta_fw, sta_bw = sess.run([output_fw, output_bw, output_state_fw, output_state_bw], {X:x, X_lengths:x_lengths} )
	print 'forword output', out_fw 
	print 'farward final_outputs', sta_fw
	print 'backword output', out_bw
	print 'backword final_outputs', sta_bw


