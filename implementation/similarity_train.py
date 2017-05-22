# -*- coding: utf-8 -*-
# train street photo classifier
# 11 categories

from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.optimizers import RMSprop, SGD
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Street2ShopDataset import Street2ShopDataset
from NBatchLogger import NBatchLogger
import os
import math
import argparse

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
    parser.add_argument("-out_weight_file", "--out_weight_file", default='model_similarity_weights.hdf5')
    parser.add_argument("-out_model_file", "--out_model_file", default='model_similarity.h5')
    parser.add_argument("-classifier_weight_file", "--classifier_weight_file", default='model_category_weights.hdf5')
    parser.add_argument("-classifier_model_file", "--classifier_model_file", default='model_category.h5')
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

    classifier = load_model(args.modelfile, custom_objects={'contrastive_loss': contrastive_loss})
    print('classifier model loaded..')
    classifier.load_weights(args.weightfile)
    print('classifier weights loaded..')

if __name__ == '__main__':
    main()
