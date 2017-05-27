# -*- coding: utf-8 -*-

from keras.models import load_model, Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Lambda, concatenate
from keras.optimizers import RMSprop, SGD, Adam
from Street2ShopDataset import Street2ShopDataset
import os
import math
from keras import backend as K
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-base_model_file", "--base_model_file")
    parser.add_argument("-model_weight", "--model_weight")
    parser.add_argument("-category", "--category")
    parser.add_argument("-opt", "--opt", default='rms')
    parser.add_argument("-batch_size", "--batch_size", default=32, type=int)

    args = parser.parse_args()

    print(args)
    batch_size = args.batch_size

    retrieval_meta_fname_list = [
        os.path.abspath("../dataset/meta/meta/json/retrieval_" + args.category + "_cleaned.json")]
    pair_meta_fname_list = [os.path.abspath("../dataset/meta/meta/json/test_pairs_" + args.category + "_cleaned.json")]
    img_dir_list = [os.path.abspath("../dataset/images/" + args.category)]
    dataset = Street2ShopDataset(retrieval_meta_fname_list, pair_meta_fname_list, img_dir_list, batch_size)

    # load the base model (state of the art CNN + 11 category classifier)
    base_model = load_model(args.base_model_file)
    print('The base model has been loaded..')
    # remove softmax layer
    base_model.outputs = [base_model.layers[-3].output]

    # build a siamese network
    input_a = Input(shape=dataset.get_input_dim())
    input_b = Input(shape=dataset.get_input_dim())
    processed_a = base_model(input_a)
    processed_b = base_model(input_b)

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
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

    model.load_weights(args.model_weight)
    print('model weights loaded..')
    model.summary()

    num_of_products = dataset.get_num_of_product_in_category(0)
    num_of_test_photos = dataset.get_num_of_pairs_in_category(0)
    street_photo_indexes = dataset.get_test_photo_indexes_in_category(0)
    product_photo_indexes = dataset.get_product_indexes_in_category(0)
    product_indexes = dataset.get_test_product_indexes_in_category(0)
    result = np.zeros(num_of_test_photos)

    print(str.format('category : {}, # of products = {}, # of test street photos = {}', args.category, num_of_products
                     , num_of_test_photos))

    for i, street_photo in enumerate(street_photo_indexes):
        gen = dataset.test_pair_generator(street_photo, 0)
        pred = model.predict_generator(gen, steps=math.ceil(len(dataset.test_product_indexes_in_category) / batch_size), verbose=1)
        sorted_indexes = [k[0] for k in sorted(enumerate(pred), key=lambda x: x[1])][-20:]
        top_k_product = [product_photo_indexes[idx] for idx in sorted_indexes]
        result[i] = 1 if product_photo_indexes[i] in top_k_product else 0
        print(str.format('{}/{} : street = {}, product_id = {}, product_photo_id = {}, found = {}, list = {}'
                         , i+1, num_of_test_photos, street_photo, product_indexes[i], product_photo_indexes[i]
                         , result[i], top_k_product))

    print(str.format('top 20 retrieval accuracy = {}', sum(result) / num_of_test_photos))

if __name__ == '__main__':
    main()