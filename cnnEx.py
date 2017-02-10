import tensorflow as tf
import numpy as np

word_dict_size = 100; sentMax=5; emb_size=4; filter_size=3; num_filters=8

w = tf.placeholder(tf.int32, [None, sentMax], name="x")		# word index matrix (N, 5)
W_wemb =  tf.Variable(tf.random_uniform([word_dict_size, emb_size], -1.0, +1.0))	# word embeddng matrix (100, 4)
emb0 = tf.nn.embedding_lookup(W_wemb, w)	# (N, 5, 4)
X = tf.expand_dims(emb0, -1) 			# (N, 5, 4, 1)

# Convolution layer
W_conv = tf.Variable(tf.truncated_normal([filter_size, emb_size, 1, num_filters], stddev=0.1), name="W_conv")	# ( 3, 4, 1, 8 )
b_conv = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_conv") 			# ( 8 )

conv = tf.nn.conv2d(X, W_conv, strides=[1, 1, 1, 1], padding="VALID", name="conv") 	# (N, 3, 1, 8)
conv1 = tf.nn.bias_add(conv, b_conv, name='conv1')		# add biase value
h1 = tf.nn.relu(conv1, name="relu") 				# apply activaiton function (N, 3, 1, 8)
pooled = tf.nn.max_pool(h1, ksize=[1, sentMax-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")	# ( N, 1, 1, 8)
p1 = tf.squeeze(pooled)	 # (N,8)

# Running
with tf.Session() as sess :

	sess.run( tf.initialize_all_variables() )
	D = np.asarray( np.random.randint(0, high=word_dict_size, size = [10, sentMax] ), dtype='int32')		# 1000 : size of dataset
	p = sess.run(p1, {w:D} )	
	print np.shape(p)		# (10, 8)



