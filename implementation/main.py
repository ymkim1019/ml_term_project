# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
#tf.python.control_flow_ops = tf
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint
from vector_similarity import TS_SS
from Street2ShopDataset import Street2ShopDataset
from NBatchLogger import NBatchLogger
import os
import math

margin = 40
epochs = 100
batch_size = 32
validation_size = 0.2
tf.logging.set_verbosity(tf.logging.ERROR)

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

def robust_contrastive_loss(y_true, y_pred):
    return K.mean(y_true * K.square(K.minimum(y_pred, margin)) +
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

    return 0 if temp is None or len(temp) == 0 else temp.mean()

def main():
    # category_list = ['outerwear', 'pants', 'bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings',
    #                  'skirts', 'tops']
    category_list = ['bags']
    retrieval_meta_fname_list = [
        os.path.abspath("../dataset/meta/meta/json/retrieval_" + category + "_cleaned.json") for category in
        category_list]
    pair_meta_fname_list = [os.path.abspath("../dataset/meta/meta/json/train_pairs_" + category + "_cleaned.json")
                            for category in category_list]
    img_dir_list = [os.path.abspath("../dataset/images/" + category) for category in category_list]
    dataset = Street2ShopDataset(retrieval_meta_fname_list, pair_meta_fname_list, img_dir_list, batch_size
                                 , validation_size)

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

    #distance = Lambda(TS_SS)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    rms = RMSprop()
    # configure the learning process
    #model.compile(loss=contrastive_loss, optimizer=rms)
    model.compile(loss=robust_contrastive_loss, optimizer=rms)

    print("fit start")
    num_train_steps = math.ceil(dataset.get_num_of_test_samples() / batch_size)
    num_val_steps = math.ceil(dataset.get_num_of_validation_samples() / batch_size)

    # keras callbacks
    out_batch = NBatchLogger()
    stop_callbacks = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    filepath = "weights-improvement.hdf5"
    # filepath = "weights-improvement-{epoch:03d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit_generator(dataset.test_pair_generator()
                        , steps_per_epoch=num_train_steps
                        , epochs=epochs, validation_data=dataset.validation_pair_generator()
                        , validation_steps=num_val_steps
                        , verbose=2, callbacks=[out_batch, stop_callbacks, checkpoint])

    print("fit end")

    #compute final accuracy on training and test sets
    # num_steps = dataset.training_size // batch_size
    # pred_pair_gen = dataset.pair_generator(dataset.x_train, dataset.y_train, dataset.training_size, batch_size)
    # i = 0
    # pred = []
    # labels = []
    # for x, y in pred_pair_gen:
    #     if i == num_steps:
    #         break
    #     pred.append(model.predict(x, verbose=1))
    #     labels.append(y)
    #     i += 1
    # pred = np.vstack(pred)
    # labels = np.hstack(labels)
    # tr_acc = compute_accuracy(pred, labels)
    #
    # num_steps = dataset.test_size // batch_size
    # pred_pair_gen = dataset.pair_generator(dataset.x_test, dataset.y_test, dataset.test_size, batch_size)
    # i = 0
    # pred = []
    # labels = []
    # for x, y in pred_pair_gen:
    #     if i == num_steps:
    #         break
    #     pred.append(model.predict(x, verbose=1))
    #     labels.append(y)
    #     i += 1
    # pred = np.vstack(pred)
    # labels = np.hstack(labels)
    # te_acc = compute_accuracy(pred, labels)
    #
    # print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    # print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    # serialize model to JSON
    print("saving full model...")
    model.save("model.h5")
    print("Saved..")

if __name__ == '__main__':
    main()
