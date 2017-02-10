import tensorflow as tf
import numpy as np

sentMax = 10 		# Max lenght of sentence 
num_classes = 3		# number of classes 
word_dict_size = 100	# word dictionary length
emb_size = 50		# word embedding size
h_len = 20		# size of rnn layer

w = tf.placeholder(tf.int32, [None, sentMax], name="x")		# word index matrix (N, 10)
sent_lens = tf.placeholder(tf.int64, [None], name="sent_lengths")
input_y = tf.placeholder(tf.int32, [None, sentMax], name="input_y") # label matrix (N, 10) 

# Embedding layer
W_wemb =  tf.Variable(tf.random_uniform([word_dict_size, emb_size], -1.0, +1.0))	# word embeddng matrix (100, 50)
emb0 = tf.nn.embedding_lookup(W_wemb, w)			# (N, 10, 50)

cell1 = tf.nn.rnn_cell.GRUCell(num_units=h_len)
cell2 = tf.nn.rnn_cell.GRUCell(num_units=h_len)  
outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
    			cell_fw = cell1,
			cell_bw = cell2,
    			dtype = tf.float32,
    			sequence_length = sent_lens,
    			inputs = emb0
			)

out_fw, out_bw = outputs 			# [N, 10, 20],  [N, 10, 20]
h1 = tf.concat(2, [out_fw, out_bw])		# (N, 10, 40)
h1_len = 2*h_len				# new h_len : 40

h2 = tf.reshape( h1, [-1, h1_len] )		# ( N*10, 40)

#Fully connected layer operations
W_ff = tf.Variable(tf.random_uniform([h1_len, num_classes], -1.0, +1.0), name="W")  # (40,3) 
b_ff = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b") 			 
H1 = tf.nn.xw_plus_b(h2, W_ff, b_ff, name="H1")			# ( N*10, 3)					 

Z1 = tf.reshape(H1, [-1, sentMax, num_classes] )		# ( N, 10, 3)

#CRF layer
log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood( Z1, input_y, sent_lens )
loss = tf.reduce_mean(-log_likelihood)  

train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


with tf.Session() as sess :

	sess.run( tf.initialize_all_variables() )

	# Create Dummy Dataset
 	D = np.asarray( np.random.randint(0, high=word_dict_size, size = [1000, sentMax] ), dtype='int32')		# 1000 : size of dataset, sentMax : number of words 
	Y = np.random.randint(0, high = num_classes, size= [1000, sentMax] )  									# lable matrix
	lengths =  np.random.randint(0, high = sentMax, size=[1000] ) 

	X_train = D[0:800]; Y_train = Y[0:800] ; lengths_train = lengths[0:800]
	X_test = D[800:] ; Y_test = Y[800:] ; lengths_test = lengths[800:]
	
	# Training
	for k in range(100) :
		_,l = sess.run([train_op, loss], {w:X_train, input_y:Y_train, sent_lens:lengths_train} )
		print 'loss', l

	# Testing
	up_list, pp = sess.run( [Z1, transition_params], {w:X_test, input_y:Y_test, sent_lens:lengths_test} )
	for up_score in up_list:
	 	viterbi_seq,_ = tf.contrib.crf.viterbi_decode(up_score, pp)		# highest scoring sequence.
		print viterbi_seq



