# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import RMSprop, SGD
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from vector_similarity import TS_SS
from Street2ShopDataset import Street2ShopDataset
from NBatchLogger import NBatchLogger
import os
import math
import pickle
import argparse

epochs = 12000
batch_size = 32
validation_size = 0.2
tf.logging.set_verbosity(tf.logging.ERROR)
margin = 0.0

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def cos_distance(vects):
    x, y = vects
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return 1-K.mean(x * y, axis=-1)


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
                  (1 - y_true) * K.maximum(K.square(margin) - K.square(y_pred), 0))

def contrastive_loss_cos(y_true, y_pred):
    losses = K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.maximum(K.square(margin) - K.square(y_pred), 0))
    return losses


def robust_contrastive_loss(y_true, y_pred):
    return K.mean(y_true * K.minimum(K.square(y_pred), K.square(margin)) +
                  (1 - y_true) * K.maximum(K.square(margin) - K.square(y_pred), 0))


def robust_contrastive_loss_cos(y_true, y_pred):
    losses = K.mean(y_true * K.minimum(y_pred, margin) +
                  (1 - y_true) * K.maximum(margin - y_pred, 0))
    return losses

def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''

    base_model = InceptionV3(weights='imagenet', include_top=False
                             , input_shape=input_dim
                             , pooling='avg')

    return base_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-distance", "--distance", default='euclidian')
    parser.add_argument("-loss", "--loss", default='contrastive')
    parser.add_argument("-m", "--m", default=40.0, type=float)
    parser.add_argument("-opt", "--opt", default='rms')
    parser.add_argument("-outputfile", "--outputfile", default='weights-improvement.hdf5')
    parser.add_argument("-categories", "--categories", nargs='+', type=int)

    args = parser.parse_args()

    print(args)

    margin = args.m

    category_list = ['outerwear', 'pants', 'bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings',
                      'skirts', 'tops']
    input_category_list = [category_list[i] for i in args.categories]

    retrieval_meta_fname_list = [
        os.path.abspath("../dataset/meta/meta/json/retrieval_" + category + "_cleaned.json") for category in
        input_category_list]
    pair_meta_fname_list = [os.path.abspath("../dataset/meta/meta/json/train_pairs_" + category + "_cleaned.json")
                            for category in input_category_list]
    img_dir_list = [os.path.abspath("../dataset/images/" + category) for category in input_category_list]
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

    if args.distance == 'euclidian':
        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    elif args.distance == 'cos':
        distance = Lambda(cos_distance,
                          output_shape=cos_dist_output_shape)([processed_a, processed_b])
    elif args.diatance == 'tsss':
        distance = Lambda(TS_SS)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    if args.opt == 'rms':
        opt = RMSprop(lr=0.0001)
    elif args.opt == 'sgd':
        opt = SGD(lr=5e-5, decay=10e-4, momentum=0.9)

    if args.loss == 'contrastive':
        loss_func = contrastive_loss
    elif args.loss == 'r_contrastive':
        loss_func = robust_contrastive_loss

    # configure the learning process
    model.compile(loss=loss_func, optimizer=opt)

    print('distance func =', distance)
    print('optimizer =', opt)
    print('loss_func =', loss_func)
    print('margin =', margin)

    print("fit start")
    num_train_steps = math.ceil(dataset.get_num_of_train_samples() / batch_size)
    num_val_steps = math.ceil(dataset.get_num_of_validation_samples() / batch_size)

    # keras callbacks
    out_batch = NBatchLogger()
    stop_callbacks = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min',min_delta=margin*0.01)
    checkpoint = ModelCheckpoint(args.outputfile, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    history = model.fit_generator(dataset.train_pair_generator()
                        , steps_per_epoch=num_train_steps
                        , epochs=epochs, validation_data=dataset.validation_pair_generator()
                        , validation_steps=num_val_steps
                        , verbose=2, callbacks=[out_batch, stop_callbacks, checkpoint])

    print("fit end")

    # serialize model to JSON
    print("saving full model...")
    model.save("model.h5")
    print("Saved..")
    # pickle.dump(history, open("history_" + args.outputfile + ".p", "wb"))

if __name__ == '__main__':
    main()
