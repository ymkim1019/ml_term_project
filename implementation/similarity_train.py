# -*- coding: utf-8 -*-
# train street photo classifier
# 11 categories

from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Lambda, concatenate
from keras.optimizers import RMSprop, SGD, Adam
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Street2ShopDataset import Street2ShopDataset
from NBatchLogger import NBatchLogger
import os
import math
import argparse
from vector_similarity import TS_SS
from keras import backend as K

epochs = 12000
validation_size = 0.2
tf.logging.set_verbosity(tf.logging.ERROR)
margin = 100

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def cos_distance(vects):
    x, y = vects
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return K.variable(1)-K.mean(x * y, axis=-1)


def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, distance):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''

    return K.mean(y_true * K.square(distance) +
                  (1 - y_true) * K.square(K.maximum(margin - distance, 0)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-base_model_file", "--base_model_file")
    parser.add_argument("-model_weight", "--model_weight")
    parser.add_argument("-category", "--category")
    parser.add_argument("-distance", "--distance", default='euclidian')
    parser.add_argument("-opt", "--opt", default='rms')
    parser.add_argument("-mode", "--mode", default='distance')
    parser.add_argument("-skip_training", "--skip_training", default=False, type=bool)
    parser.add_argument("-batch_size", "--batch_size", default=32, type=int)
    parser.add_argument("-data_augmentation", "--data_augmentation", default=False, type=bool)
    parser.add_argument("-data_augmentation_ratio", "--data_augmentation_ratio", default=1, type=int)
    parser.add_argument("-negative_pair_ratio", "--negative_pair_ratio", default=1, type=int)
    parser.add_argument("-augmented_pair", "--augmented_pair", default=False, type=bool)

    args = parser.parse_args()

    print(args)
    batch_size = args.batch_size

    # dataset
    retrieval_meta_fname = os.path.abspath("../dataset/meta/meta/json/retrieval_" + args.category + "_cleaned.json")
    pair_meta_fname = os.path.abspath("../dataset/meta/meta/json/train_pairs_" + args.category + "_cleaned.json")
    img_dir = os.path.abspath("../dataset/images/" + args.category)
    dataset = Street2ShopDataset([retrieval_meta_fname], [pair_meta_fname], [img_dir], batch_size, validation_size,
                                 data_augmentation=args.data_augmentation
                                 , data_augmentation_ratio=args.data_augmentation_ratio
                                 , negative_pair_ratio=args.negative_pair_ratio
                                 , augmented_pair=args.augmented_pair)

    # load the base model (state of the art CNN + 11 category classifier)
    base_model = load_model(args.base_model_file)
    print('The base model has been loaded..')
    base_model.summary()
    # remove softmax layer
    base_model.outputs = [base_model.layers[-3].output]

    # build a siamese network
    input_a = Input(shape=dataset.get_input_dim())
    input_b = Input(shape=dataset.get_input_dim())
    processed_a = base_model(input_a)
    processed_b = base_model(input_b)

    if args.mode == 'distance' or args.mode == 'contrasive_loss':
        if args.distance == 'euclidian':
            distance = Lambda(euclidean_distance,
                              output_shape=eucl_dist_output_shape)([processed_a, processed_b])
        elif args.distance == 'cos':
            distance = Lambda(cos_distance,
                              output_shape=cos_dist_output_shape)([processed_a, processed_b])
        elif args.distance == 'tsss':
            distance = Lambda(TS_SS)([processed_a, processed_b])

        if args.mode == 'distance':
            prediction = Dense(1, activation='sigmoid')(distance)
        else:
            prediction = distance

    elif args.mode == 'metric_learning':
        # freeze all convolutional InceptionV3 layers
        # for layer in base_model.layers:
        #     layer.trainable = False
        merged = concatenate([processed_a, processed_b], axis=-1)
        x = Dense(1024, activation='relu')(merged)
        x = Dense(1024, activation='relu')(x)
        prediction = Dense(1, activation='sigmoid')(x)

    model = Model([input_a, input_b], prediction)

    if args.opt == 'rms':
        opt = RMSprop(lr=0.0001)
    elif args.opt == 'sgd':
        opt = SGD(lr=5e-5, decay=10e-4, momentum=0.9)
    elif args.opt == 'adam':
        opt = Adam()

    # configure the learning process
    print("compiling model..")
    if args.mode == 'contrasive_loss':
        model.compile(loss=contrastive_loss, optimizer=opt)
    else:
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

    if args.skip_training is False:
        print("fit start")
        num_train_steps = dataset.get_num_of_train_steps(batch_size)
        num_val_steps = dataset.get_num_of_val_steps(batch_size)

        # keras callbacks
        out_batch = NBatchLogger()
        if args.mode == 'metric_learning':
            stop_callbacks = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
            checkpoint = ModelCheckpoint(args.model_weight, monitor='val_acc', verbose=1, save_best_only=True
                                            , save_weights_only=True, mode='max')
        else:
            stop_callbacks = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
            checkpoint = ModelCheckpoint(args.model_weight, monitor='val_loss', verbose=1, save_best_only=True
                                            , save_weights_only=True, mode='min')

        history = model.fit_generator(dataset.train_pair_generator()
                                      , steps_per_epoch=num_train_steps
                                      , epochs=epochs, validation_data=dataset.validation_pair_generator()
                                      , validation_steps=num_val_steps
                                      , verbose=2, callbacks=[out_batch, stop_callbacks, checkpoint])

        print("fit end")

    if args.skip_training is True:
        model.load_weights(args.model_weight)
        print('Weights have been loaded..')

    # dataset
    retrieval_meta_fname = os.path.abspath("../dataset/meta/meta/json/retrieval_" + args.category + "_cleaned.json")
    pair_meta_fname = os.path.abspath("../dataset/meta/meta/json/test_pairs_" + args.category + "_cleaned.json")
    img_dir = os.path.abspath("../dataset/images/" + args.category)
    dataset = Street2ShopDataset([retrieval_meta_fname], [pair_meta_fname], [img_dir], batch_size)

    num_test_steps = math.ceil(dataset.get_num_of_train_samples() / batch_size)
    print("Evaluating..")
    print(model.evaluate_generator(dataset.train_pair_generator(), num_test_steps))

if __name__ == '__main__':
    main()
