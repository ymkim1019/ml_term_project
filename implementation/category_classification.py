# -*- coding: utf-8 -*-
# train street photo classifier
# 11 categories

from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Lambda, Dense, GlobalAveragePooling2D
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

def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''

    base_model = InceptionV3(weights='imagenet', include_top=False
                             , input_shape=input_dim)

    return base_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-weightfile", "--weightfile", default='model_category_weights.hdf5')
    parser.add_argument("-modelfile", "--modelfile", default='model_category.h5')
    parser.add_argument("-mode", "--mode", default='train')

    args = parser.parse_args()

    print(args)

    category_list = ['outerwear', 'pants', 'bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings',
                      'skirts', 'tops']

    retrieval_meta_fname_list = [
        os.path.abspath("../dataset/meta/meta/json/retrieval_" + category + "_cleaned.json") for category in
        category_list]
    pair_meta_fname_list = [os.path.abspath("../dataset/meta/meta/json/train_pairs_" + category + "_cleaned.json")
                            for category in category_list]
    img_dir_list = [os.path.abspath("../dataset/images/" + category) for category in category_list]

    if args.mode == 'train':
        dataset = Street2ShopDataset(retrieval_meta_fname_list, pair_meta_fname_list, img_dir_list, batch_size
                                     , validation_size)

        # network definition
        base_network = create_base_network(dataset.get_input_dim())

        # input tensors
        input_a = Input(shape=dataset.get_input_dim())

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        x = base_network(input_a)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(11, activation='softmax')(x)

        model = Model(input_a, predictions)

        # first: train only the top layers
        # freeze all convolutional InceptionV3 layeres
        for layer in base_network.layers:
            layer.trainable = False

        # configure the learning process
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # for i, layer in enumerate(base_network.layers):
        #     print(i, layer.name)

        print("1st fit start")
        num_train_steps = math.ceil(dataset.get_num_of_train_samples() / batch_size)
        num_val_steps = math.ceil(dataset.get_num_of_validation_samples() / batch_size)

        # keras callbacks
        out_batch = NBatchLogger()
        history = model.fit_generator(dataset.train_category_generator()
                            , steps_per_epoch=num_train_steps
                            , epochs=10, validation_data=dataset.val_category_generator()
                            , validation_steps=num_val_steps
                            , verbose=2, callbacks=[out_batch])

        print("1st fit end")

        for layer in base_network.layers:
            layer.trainable = True

        print("2nd fit start")
        stop_callbacks = EarlyStopping(monitor='acc', patience=10, verbose=1, mode='min', min_delta=0.0001)
        checkpoint = ModelCheckpoint(args.weightfile, monitor='acc', verbose=1, save_best_only=True, mode='min')
        num_train_steps = math.ceil(dataset.get_num_of_train_samples() / batch_size)
        num_val_steps = math.ceil(dataset.get_num_of_validation_samples() / batch_size)

        # keras callbacks
        out_batch = NBatchLogger()
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 172 layers and unfreeze the rest:
        for layer in model.layers[:172]:
            layer.trainable = False
        for layer in model.layers[172:]:
            layer.trainable = True
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0001, momentum=0.9), metrics=['accuracy'])
        history = model.fit_generator(dataset.train_category_generator()
                                      , steps_per_epoch=num_train_steps
                                      , epochs=1000, validation_data=dataset.val_category_generator()
                                      , validation_steps=num_val_steps
                                      , verbose=2, callbacks=[out_batch, stop_callbacks, checkpoint])

        print("2nd fit end")

        # serialize model to JSON
        print("saving full model...")
        # model.save(args.modelfile)
        print("Saved..")
    else:
        model = load_model(args.modelfile, custom_objects={'contrastive_loss': contrastive_loss})
        print('model loaded..')
        model.load_weights(args.weightfile)
        print('weights loaded..')
        # test
        print("Evaluating...")
        pair_meta_fname_list = [os.path.abspath("../dataset/meta/meta/json/test_pairs_" + category + "_cleaned.json")
                                for category in category_list]
        test_dataset = Street2ShopDataset(retrieval_meta_fname_list, pair_meta_fname_list, img_dir_list, batch_size)
        steps = math.ceil(test_dataset.get_num_of_train_samples() / batch_size)
        scores = model.evaluate_generator(test_dataset.train_category_generator(), steps=steps)
        print(model.metrics_names)
        print(scores)

if __name__ == '__main__':
    main()
