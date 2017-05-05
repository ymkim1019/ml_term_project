from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from Cifar10Dataset import Cifar10DataSet

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

    temp = labels[labels[predictions.ravel() < margin]]

    return 0 if temp is None else temp.mean()

def main():
    epochs = 1
    batch_size = 2

    dataset = Cifar10DataSet(training_size = 10, validation_size = 4, test_size = 20)

    # network definition
    base_network = create_base_network(dataset.get_input_dim())

    # input tensors
    input_a = Input(shape=dataset.get_input_dim())
    input_b = Input(shape=dataset.get_input_dim())

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    rms = RMSprop()
    # configure the learning process
    model.compile(loss=contrastive_loss, optimizer=rms)

    print("fit start")
    num_train_steps = dataset.training_size // batch_size
    # num_val_steps = dataset.validation_size // batch_size
    train_fit_generator = dataset.pair_generator(dataset.x_train, dataset.y_train, dataset.training_size, batch_size)
    # val_fit_generator = dataset.pair_generator(dataset.x_validation, dataset.y_validation, dataset.validation_size
    #                                            , batch_size)
    # model.fit_generator(train_fit_generator, steps_per_epoch=num_train_steps, epochs=epochs
    #                     , validation_data=val_fit_generator, validation_steps=num_val_steps, verbose=2)
    model.fit_generator(train_fit_generator, steps_per_epoch=num_train_steps, epochs=epochs, verbose=2)
    print("fit end")

    #compute final accuracy on training and test sets
    num_steps = dataset.training_size // batch_size
    pred_pair_gen = dataset.pair_generator(dataset.x_train, dataset.y_train, dataset.training_size, batch_size)
    i = 0
    pred = []
    labels = []
    for x, y in pred_pair_gen:
        if i == num_steps:
            break
        pred.append(model.predict(x, verbose=1))
        labels.append(y)
        i += 1
    pred = np.vstack(pred)
    labels = np.hstack(labels)
    tr_acc = compute_accuracy(pred, labels)

    num_steps = dataset.test_size // batch_size
    pred_pair_gen = dataset.pair_generator(dataset.x_test, dataset.y_test, dataset.test_size, batch_size)
    i = 0
    pred = []
    labels = []
    for x, y in pred_pair_gen:
        if i == num_steps:
            break
        pred.append(model.predict(x, verbose=1))
        labels.append(y)
        i += 1
    pred = np.vstack(pred)
    labels = np.hstack(labels)
    te_acc = compute_accuracy(pred, labels)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

if __name__ == '__main__':
    main()