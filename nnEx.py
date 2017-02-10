import tensorflow as tf
import numpy as np



i_dim = 10
h1_dim = 5
num_classes = 3

#Symbolic or Place holder for input and output
X = tf.placeholder(tf.float32, [None, i_dim], name='input')		 	# 100 X 10 
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y") 	# 100 X 3 

#Initialize parameters
W1 = tf.get_variable('W_1', shape = [i_dim, h1_dim], initializer = tf.random_normal_initializer() ) 	# 10 X 5
b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]), name="b1")						# 5
W2 = tf.get_variable('W_2', shape = [h1_dim, num_classes], initializer = tf.random_normal_initializer() )	# 5X3  
b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")						# 3



#First layer operations
H1 = tf.nn.xw_plus_b(X, W1, b1, name="H2")	# 100 X 5
Z1 = tf.sigmoid(H1) 	
			# 100 X 5
#Second layer operations
H2 = tf.nn.xw_plus_b(Z1, W2, b2, name="H2")	# 100 X 3

#Loss function
losses = tf.nn.softmax_cross_entropy_with_logits(H2, input_y)	# 100
loss = tf.reduce_mean(losses) + 0.001 * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) ) 		# 1

#Predicitons of the batch
predictions = tf.argmax(H2, 1, name="predictions")

#Accuracy of correct prediction in batch
correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1) )
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

#Optimization
optimizer = tf.train.AdamOptimizer(1e-2)
grads_and_vars = optimizer.compute_gradients(loss)
global_step = tf.Variable(0, name="global_step", trainable=False)
train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)




with tf.Session() as sess :

	sess.run(tf.initialize_all_variables())

	# Create Dataset
 	D = np.asarray( np.random.rand(1000, 10), dtype='float32')
	Y = np.asarray ( np.zeros((1000, 3)), dtype='float32' )
	for i in range(1000) :
		k = np.random.randint(3)
		Y[i][k] = 1.0

	X_train = D[0:800]; Y_train = Y[0:800]
	X_test = D[800:] ; Y_test = Y[800:]


	# Training
	for k in range(100) :
		_,l, acc = sess.run([train_op, loss, accuracy], {X:X_train, input_y:Y_train} )
		print 'loss and accuracy', l, acc
	# Testing
	acc, pred = sess.run( [accuracy, predictions], {X:X_test, input_y:Y_test} )
	print "Accuracy in test set", acc




