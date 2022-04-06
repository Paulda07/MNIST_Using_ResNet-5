#-----------------------------------------------------
# The following is a sample starter code for Project 2,
#  however it does not meet the specifications, and also uses
#  functions that you are not allowed to use.
#
# Refer to project description for specifications on how to
#  implement the project
#-----------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import activations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os

np.set_printoptions(suppress=True)   # suppress scientific notation




mnist = tf.keras.datasets.mnist

(x_train, label_train), (x_test, label_test) = mnist.load_data()
x_train, x_valid, x_test = x_train[:4000], x_train[4000:5000], x_test[:1000]
label_train, label_valid, label_test = label_train[:4000], label_train[4000:5000], label_test[:1000]
x_train, x_test = x_train / 255.0, x_test / 255.0

#-----------
# Save an image to a file
#-----------
for i in range(5):
	direct = "C:/Users/Achsah/Documents/Paul's/CV/New folder/output/"
	title = direct + "train_" + str(i) + ".png"
	cv2.imwrite(title, 255*x_train[i])
	title = direct + "test_"+str(i)+".png"
	cv2.imwrite(title, 255*x_test[i])
	title = direct + "valid_"+str(i)+".png"
	cv2.imwrite(title, 255*x_valid[i])


num_train = x_train.shape[0]
num_test  = x_test.shape[0]

# One-hot encode the training
y_train = np.zeros([num_train, 10])
for i in range(num_train):
	y_train[i, label_train[i]] = 1

# One-hot encode the testing
y_test  = np.zeros([num_test, 10])
for i in range(num_test):
	y_test[i, label_test[i]] = 1

print("x_train.shape", x_train.shape)
print("x_test.shape", x_test.shape)
print("label_train.shape", label_train.shape)
print("label_test.shape", label_test.shape)
print("y_train.shape", y_train.shape)
print("y_test.shape", y_test.shape)
input("press enter to continue")

#--------------------------
# Create the model
#--------------------------

#----------
# Construct the model using layers
#
# NOTE:  For Proj2 you CANNOT use Sequential
#  you must create the layers one at a time as variables
#----------
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),    # All deep layers should use relu
#   tf.keras.layers.Dense(10, activation='softmax')
# ])

# NOTE:  You may use code similar to the following
z = layers.Input(shape=[28,28])
print("z")
print(z)


a = layers.Reshape(target_shape=[784])(z)
print("a")
print(a)

b = tf.keras.layers.RepeatVector(2)(a)    # All deep layers should use relu
print("b")
print(b)

c = layers.Reshape(target_shape=[28, 28, 2])(b)
print("c")
print(c)

d = tf.keras.layers.Conv2D(filters = 2, kernel_size = 3, padding = "SAME")(c)
print("d")
print(d)

e = tf.keras.layers.Conv2D(filters = 2, kernel_size = 3, padding = "SAME")(d)
print("e")
print(e)

f = tf.keras.layers.Concatenate(axis=3)([c, e])
print("f")
print(f)

g = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding = "SAME")(f)
print("g")
print(g)

h = tf.keras.layers.Conv2D(filters = 4, kernel_size = 3, padding = "SAME")(g)
print("h")
print(h)

i = tf.keras.layers.Conv2D(filters = 4, kernel_size = 3, padding = "SAME")(h)
print("i")
print(i)

j = tf.keras.layers.Concatenate(axis=3)([g, i])
print("j")
print(j)

k = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding = "SAME")(j)
print("k")
print(k)

l = tf.keras.layers.Conv2D(filters = 8, kernel_size = 3, padding = "SAME")(k)
print("l")
print(l)

m = tf.keras.layers.Conv2D(filters = 8, kernel_size = 3, padding = "SAME")(l)
print("m")
print(m)

n = tf.keras.layers.Concatenate(axis=3)([k, m])
print("n")
print(n)

o = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding = "valid")(n)
print("o")
print(o)

p = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, padding = "SAME")(o)
print("p")
print(p)

q = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, padding = "SAME")(p)
print("q")
print(q)

r = tf.keras.layers.Concatenate(axis=3)([o, q])
print("r")
print(r)

s = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding = "valid")(r)
print("s")
print(s)

t = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = "SAME")(s)
print("t")
print(t)

u = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = "SAME")(t)
print("u")
print(u)

v = tf.keras.layers.Flatten()(u)
print("v")
print(v)

w = tf.keras.layers.Dense(20, activation='relu')(v)
print("w")
print(w)

x = tf.keras.layers.Dense(10, activation='softmax')(w)
print("x")
print(x)



model = models.Model(z,x)
print("model")
print(model)


#----------
# Create the loss function
#
# NOTE:  For Proj2 you CANNOT use a pre-made keras loss function
#  you must create the loss function as additional layers
#----------
loss_fn = tf.keras.losses.MeanSquaredError()

#----------
# Compile the model
#
# NOTE:  For Proj2 you CANNOT just compile the model,
#  you must create an optimizer variable as listed in
#  keras.optimizers, and give the model's training weights
#  to the optimizer. 
#
# NOTE:  You MAY use code similar to the following
#   adam_op = optimizers.Adam(learning_rate=0.0001)
#
#----------
# model.compile(optimizer='adam',
#               loss=loss_fn,
#               metrics=['accuracy'])

adam_op = optimizers.Adam(learning_rate=0.0001)
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
#----------
# Fit the model
#
# NOTE:  For Proj2 you CANNOT call model.fit
#  you must create a custom training loop with pseudocode
#
# batch_size = 100
# num_batches = num_train / batch_size
#
# for e in range(num_epochs):
#     for b in range(num_batches):
#         # perform gradient update over minibatch using "gradient tape"
#----------
# print("model.fit")
# model.fit(x_train, y_train, epochs=1)

#----------
# For proj2 you are discouraged from calling model.evaluate
#   you are encouraged to load the data and run the model directly
#----------
# metrics = model.evaluate(x_test,  y_test, verbose=2)

# print("metrics")
#print(metrics)
#input("press enter to continue")

#----------------
# Let's play with gradient tape
#----------------
# def logloss_gradient (m):
# 	logloss = 0
# 	for i in range(100):
# 		for j in range(10):
# 			logloss-= 

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# Gradient tape records the gradient for every operation
num_epochs = 50
num_batch = 40
for epoch in range(num_epochs):
	flag = 1
	for batch in range(num_batch):
		x = x_train[batch*100:(batch*100)+100,:]   # take a "minibatch" of 10 images
		y = label_train[batch*100:(batch*100)+100]

		# K.constant converts a numpy array into a tensor 
		#   i.e. loading data onto the GPU
		x_tensor = K.constant(x)
		y_tensor = K.constant(y)
		with tf.GradientTape() as tape:
			predict_tensor = model(x_tensor)    # This runs the data through the model   F(x)
			predict = predict_tensor.numpy()     # Read the data out of tensorflow into numpy
			# print("Here",predict.shape)
			# calculate mse = (y-F(x))^2
			# print(y_tensor, predict_tensor)
			# logloss_gradient (y, predict)
			#mse_tensor = (y_tensor - predict_tensor) * (y_tensor - predict_tensor)
			
			logloss_tensor = - y_tensor * tf.math.log(tf.clip_by_value(predict_tensor, clip_value_min= 0.0001, clip_value_max=1.0))
			if (flag == 1):
				#print(mse_tensor)
				# print(tf.clip_by_value(predict_tensor, clip_value_min= 0.0001, clip_value_max=1.0))
				# print(logloss_tensor)
				flag = 0
			logloss_gradient=tape.gradient(logloss_tensor, model.trainable_weights)
			adam_op.apply_gradients(zip(logloss_gradient, model.trainable_weights))
			train_acc_metric.update_state(y_tensor, predict_tensor)
			# Note: your loss must be of the following form
			#  logloss = - y * log F(x)
	train_acc = train_acc_metric.result()
	print("Training acc over epoch here: %.4f" %(float(train_acc)))

    # Reset training metrics at the end of each epoch
	c=train_acc_metric.reset_states()

# print("predict vector")
# for i in range(10):
# 	print("y",y[i],"F(x)",predict[i])
# input("press enter to continue")

# What was the gradient of the above operation ?
#predict_gradient=tape.gradient(predict_tensor, model.trainable_weights);
#print("predict_gradient")
#print(predict_gradient)
#input("above is predict_gradient press enter to continue")

# What is the gradient with respect to mean squared error
# mse_gradient=tape.gradient(mse_tensor, model.trainable_weights)

# Note: to perform a step of gradient descent you must use
#  code similar to the following
#   adam_op.apply_gradients(zip(logloss_gradient, model.trainable_weights))

print("mse_gradient")
# print(mse_gradient)
input("above is mse_gradient press enter to continue")







