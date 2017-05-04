from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf
import random
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3

# 동일 item이라고 판단하는 margin 값
margin = 1

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''

    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))



def create_pairs(x, digit_indices, pairs_left, pairs_right, labels):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    n = min([len(digit_indices[d]) for d in range(10)]) - 1 # 각 class마다 cardinality가 다른데, 가장 크기가 작은 class의 cardinality를 n으로..
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1] # 같은 class안에서 2개를 짝짓는다, positive pair
            pairs_left[d*i*2] = x[z1]
            pairs_right[d*i*2] = x[z2]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i] # 다른 class의 데이터와 짝짓는다, negative pair
            pairs_left[d*i*2+1] = x[z1]
            pairs_right[d*i*2+1] = x[z2]
            labels[d*i*2] = 1
            labels[d*i*2+1] = 0

def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''

    base_model = InceptionV3(weights='imagenet', include_top=False
                             , input_shape=input_dim
                             , pooling='avg')

    return base_model

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''

    return labels[predictions.ravel() < margin].mean()

# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[0:1000]
y_train = y_train[0:1000]
x_test = x_test[0:1000]
y_test = y_test[0:1000]

print("cifar10 load done")

# x_train = x_train.reshape(60000, 784)
x_train = x_train.repeat(5, axis=1)
x_train = x_train.repeat(5, axis=2)
#x_train = np.expand_dims(x_train, axis=1)
# x_test = x_test.reshape(10000, 784)
x_test = x_test.repeat(5, axis=1)
x_test = x_test.repeat(5, axis=2)
#x_test = np.expand_dims(x_test, axis=1)
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
# input_dim = 784
#y_train = np.ravel(y_train)
#y_test = np.ravel(y_test)

input_dim = (32*5, 32*5, 3)
epochs = 1

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(10)]
n = min([len(digit_indices[d]) for d in range(10)]) - 1
tr_pairs_left = np.empty([20*n, 160 ,160, 3])
tr_pairs_right = np.empty([20*n, 160 ,160, 3])
tr_y = np.empty([20*n])
# tr_pairs, tr_y = create_pairs(x_train, digit_indices)
create_pairs(x_train, digit_indices, tr_pairs_left, tr_pairs_right, tr_y)

digit_indices = [np.where(y_test == i)[0] for i in range(10)]
n = min([len(digit_indices[d]) for d in range(10)]) - 1
te_pairs_left = np.empty([2*n*10, 160 ,160, 3])
te_pairs_right = np.empty([2*n*10, 160 ,160, 3])
te_y = np.empty([20*n])
# te_pairs, te_y = create_pairs(x_test, digit_indices)
create_pairs(x_test, digit_indices, te_pairs_left, te_pairs_right, te_y)

print("pairs are ready")

# network definition
base_network = create_base_network(input_dim)

# input_a = Input(shape=(input_dim,))
# input_b = Input(shape=(input_dim,))
input_a = Input(shape=input_dim)
input_b = Input(shape=input_dim)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
# base network에 각각의 tensor를 넣으면 output tensor들이 나온다
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)


rms = RMSprop()
# configure the learning process
model.compile(loss=contrastive_loss, optimizer=rms)

# train
# model.fit([[tr_pairs[:][0]], [tr_pairs[:][1]]], tr_y,
#           batch_size=128, nb_epoch=epochs, validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

print("fit start")
model.fit([tr_pairs_left[0:128], tr_pairs_right[0:128]], tr_y[0:128],
          batch_size=10, epochs=epochs, validation_data=([te_pairs_left[0:10], te_pairs_right[0:10]], te_y[0:10]), verbose=2)

print("fit ended")

#compute final accuracy on training and test sets
pred = model.predict([tr_pairs_left[0:10], tr_pairs_right[0:10]], verbose=1)
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs_left[0:10], te_pairs_right[0:10]], verbose=1)
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")