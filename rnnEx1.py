import tensorflow as tf
import numpy as np

sentMax = 10 		# Max lenght of sentence 
num_classes = 3		# number of classes 
word_dict_size = 100	# word dictionary length
emb_size = 50		# word embedding size
h1_len = 70 		# number of filters we apply


w = tf.placeholder(tf.int32, [None, sentMax], name="x")		# word index matrix (N, 10)
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y") # label matrix (N, 3) 
sent_len = tf.placeholder(tf.int64, [None], name='sent_len')		# length of every sentence in batch (N)

W_wemb =  tf.Variable(tf.random_uniform([word_dict_size, emb_size], -1.0, +1.0))	# word embeddng matrix (100, 50)

emb0 = tf.nn.embedding_lookup(W_wemb, w)	# (N, 10, 50)
X =  emb0

cell_f = tf.nn.rnn_cell.LSTMCell( num_units=h1_len )
cell_b = tf.nn.rnn_cell.LSTMCell( num_units=h1_len )
outputs, states = tf.nn.bidirectional_dynamic_rnn(
							cell_fw	=cell_f, 
							cell_bw	=cell_b, 
							inputs	=X,
							sequence_length=sent_len, 
							dtype	= tf.float32 	
						)

output_fw, output_bw = outputs	# N, M, 100
states_fw, states_bw = states	# [N, 100] [N, 100]

h1_fw = states_fw.h		# final hidden states of farword rnn
h1_bw = states_bw.h		# final hidden states of backword rnn

h1 = tf.concat(1, [h1_fw, h1_bw])
h1_size = 2 * h1_len

#Fully connected layer operations
W_ff = tf.Variable(tf.truncated_normal([h1_size, num_classes], stddev=0.1), name="W") # (70, 3)
b_ff = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")			# (3)
H2 = tf.nn.xw_plus_b(h1, W_ff, b_ff, name="H1")			# (N, 3)

losses = tf.nn.softmax_cross_entropy_with_logits(H2, input_y)	#  N
loss = tf.reduce_mean(losses)  		# 1
predictions = tf.argmax(H2, 1, name="predictions")
correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1) )
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

optimizer = tf.train.AdamOptimizer(1e-2)
grads_and_vars = optimizer.compute_gradients(loss)
global_step = tf.Variable(0, name="global_step", trainable=False)
train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)


with tf.Session() as sess :

	sess.run( tf.initialize_all_variables() )

	# Create Dummy Dataset
 	D = np.asarray( np.random.randint(0, high=word_dict_size, size = [1000, sentMax] ), dtype='int32')		# 1000 : size of dataset

	sentlen = np.asarray( np.random.randint(0, high=sentMax, size = [1000] ), dtype='int32')			# 1000 values for each sent length

	Y = np.asarray ( np.zeros((1000, num_classes)), dtype='float32' )
	for i in range(1000) :
		k = np.random.randint(0,3)			# create a number between (0,3]
		Y[i][k] = 1.0
	 
	X_train = D[0:800]; Y_train = Y[0:800]; sentlen_train = sentlen[0:800]
	X_test = D[800:] ; Y_test = Y[800:]; sentlen_test = sentlen[800:]
	 

	# Training
	for k in range(100) :
		_,l, acc = sess.run([train_op, loss, accuracy], {w:X_train, input_y:Y_train, sent_len:sentlen_train} )
		print 'loss and accuracy', l, acc
	# Testing
	acc, pred = sess.run( [accuracy, predictions], {w:X_test, input_y:Y_test, sent_len:sentlen_test} )
	print "Accuracy in test set", acc



