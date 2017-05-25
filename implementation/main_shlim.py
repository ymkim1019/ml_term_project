# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
#tf.python.control_flow_ops = tf
from keras.models import Model
from keras.layers import Input, Lambda, Dense, concatenate
from keras.optimizers import RMSprop, SGD
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint
from vector_similarity import TS_SS
from Street2ShopDataset import Street2ShopDataset
from NBatchLogger import NBatchLogger
import os
import math
import matplotlib.pyplot as plt

margin = 40
epochs = 12000
batch_size = 32
validation_size = 0.2
tf.logging.set_verbosity(tf.logging.ERROR)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def cos_distance(vects):
    x, y = vects
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

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

#def cross_entropy(y_true, y_pred):
#    print(y_true)
#    a,b = y_true
#    pa = math.exp(a)/(math.exp(a)+math.exp(b))
#    pb = math.exp(b) / (math.exp(a) + math.exp(b))
#
#    return - (y_true*math.log(pa) + (1-y_true)*math.log(pb))

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
    merged_vector = concatenate([processed_a,processed_b],axis=-1)
    metriclayer1 = Dense(2048,activation='relu')(merged_vector)
    metriclayer2 = Dense(2048,activation='relu')(metriclayer1)
    prediction = Dense(1,activation='sigmoid')(metriclayer2)


    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    # distance = Lambda(cos_distance,
    #                   output_shape=cos_dist_output_shape)([processed_a, processed_b])

    #distance = Lambda(TS_SS)([processed_a, processed_b])

    model = Model([input_a, input_b], prediction)
    #model = Model([input_a, input_b], distance)

    rms = RMSprop()
    sgd = SGD(lr=5e-5, decay=10e-4, momentum=0.9)
    # configure the learning process
    #model.compile(loss=contrastive_loss, optimizer=rms)
    model.compile(loss='binary_crossentropy', optimizer=rms)
    #model.compile(loss=cross_entropy, optimizer=rms)

    print("fit start")
    num_train_steps = math.ceil(dataset.get_num_of_train_samples() / batch_size)
    num_val_steps = math.ceil(dataset.get_num_of_validation_samples() / batch_size)

    # keras callbacks
    out_batch = NBatchLogger()
    stop_callbacks = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min')
    #filepath = "weights-improvement.hdf5"
    #filepath = "weights-improvement-{epoch:03d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    history = model.fit_generator(dataset.train_pair_generator()
                        , steps_per_epoch=num_train_steps
                        , epochs=epochs, validation_data=dataset.validation_pair_generator()
                        , validation_steps=num_val_steps
                        , verbose=2, callbacks=[out_batch, stop_callbacks])#, checkpoint])

    print("fit end")

    # serialize model to JSON
    print("saving full model...")
    model.save("model.h5")
    print("Saved..")

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('history.png')

if __name__ == '__main__':
    main()
