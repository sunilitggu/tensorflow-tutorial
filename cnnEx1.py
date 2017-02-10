import tensorflow as tf
import numpy as np

sentMax = 10 		# Max lenght of sentence 
num_classes = 3		# number of classes 
word_dict_size = 100	# word dictionary length
emb_size = 50		# word embedding size
filter_size = 3		# filter/kernal size (in terms of number of words)
num_filters = 70 	# number of filters we apply

w = tf.placeholder(tf.int32, [None, sentMax], name="x")		# word index matrix (N, 10)
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y") # label matrix (N, 3) 
W_wemb =  tf.Variable(tf.random_uniform([word_dict_size, emb_size], -1.0, +1.0))	# word embeddng matrix (100, 50)

emb0 = tf.nn.embedding_lookup(W_wemb, w)	# (N, 10, 50)
X = tf.expand_dims(emb0, -1) 			# (N, 10, 50, 1)

# Convolution layer
W_conv = tf.Variable(tf.truncated_normal([filter_size, emb_size, 1, num_filters], stddev=0.1), name="W_conv")	# ( 3, 50, 1, 70 )
b_conv = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_conv") 			# ( 70 )
conv = tf.nn.conv2d(X, W_conv, strides=[1, 1, 1, 1], padding="VALID", name="conv") 	# (N, 8, 1, 70)
conv = tf.nn.bias_add(conv, b_conv, name='z1')			# add biase value
h1 = tf.nn.relu(conv, name="relu") 				# apply activaiton function (N, 8, 1, 70)
pooled = tf.nn.max_pool(h1, ksize=[1, sentMax-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")	# ( N, 1, 1, 70)
p1 = tf.squeeze(pooled)								# (N, 70)	

#Fully connected layer operations
W_ff = tf.Variable(tf.truncated_normal([num_filters, num_classes], stddev=0.1), name="W") # (70, 3)
b_ff = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")			# (3)
H2 = tf.nn.xw_plus_b(p1, W_ff, b_ff, name="H1")			# (N, 3)
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
	Y = np.asarray ( np.zeros((1000, num_classes)), dtype='float32' )
	for i in range(1000) :
		k = np.random.randint(0,3)			# create a number between (0,3]
		Y[i][k] = 1.0
	 
	X_train = D[0:800]; Y_train = Y[0:800]
	X_test = D[800:] ; Y_test = Y[800:]


	# Training
	for k in range(100) :
		_,l, acc = sess.run([train_op, loss, accuracy], {w:X_train, input_y:Y_train} )
		print 'loss and accuracy', l, acc
	# Testing
	acc, pred = sess.run( [accuracy, predictions], {w:X_test, input_y:Y_test} )
	print "Accuracy in test set", acc



