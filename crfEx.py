import tensorflow as tf
import numpy as np

sentMax = 10 		# Max lenght of sentence 
num_classes = 3		# number of classes 
num_features = 15 	# number of features

X = tf.placeholder(tf.float32, [None, sentMax, num_features] )	# (6, 10, 15)
input_y = tf.placeholder(tf.int32, [None, sentMax] )		# (6, 10)
X_lengths = tf.placeholder(tf.int32, [None] )			# (6)
X1 = tf.reshape( X, [-1, num_features] )			# (6*10, 15)

#Fully connected layer operations
W_ff = tf.Variable(tf.random_uniform([num_features, num_classes], -1.0, +1.0), name="W")  # (15,3) 
b_ff = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b") 			 
H1 = tf.nn.xw_plus_b(X1, W_ff, b_ff, name="H1")			# ( 6*10, 3)					 

Z1 = tf.reshape(H1, [-1, sentMax, num_classes] )		# ( 6, 10, 3)

#CRF layer
log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood( Z1, input_y, X_lengths )
loss = tf.reduce_mean(-log_likelihood)  


# Create input data
x = np.asarray( np.random.randn(6, sentMax, num_features), dtype='float32' )	# dataset: #sentence=6, #words = 10, #features = 15 
y = np.random.randint(3, size=[6, sentMax])		# 3 number of lables ( B, I, O )
x_lengths = [8, 5, 7, 9, 10, 4]				# actual length of each sentence

with tf.Session() as sess:
	sess.run( tf.initialize_all_variables() )
	l, tp = sess.run([loss, transition_params], {X:x, X_lengths:x_lengths, input_y:y } )
	print l

	#Decoding	
	us, ps = sess.run([Z1, transition_params], {X:x, X_lengths:x_lengths, input_y:y } )	
	viterbi_sequence,_ = tf.contrib.crf.viterbi_decode(us[1], ps)		# highest scoring sequence.
	print 'true seq', y[1]
	print 'pred seq', viterbi_sequence
	
	


