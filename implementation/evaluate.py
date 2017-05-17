# -*- coding: utf-8 -*-

from keras.models import load_model, Model
from keras.layers import Input
from Street2ShopDataset import Street2ShopDataset
import os
import math
from keras import backend as K
from PIL import Image
import numpy as np

batch_size = 2

def product_x_generator(retrieval_meta_fname, img_dir, batch_size, out_photo_indexes)
    import json

def main():
    # category_list = ['outerwear', 'pants', 'bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings',
    #                  'skirts', 'tops']
    category_list = ['bags']
    retrieval_meta_fname_list = [
        os.path.abspath("../dataset/meta/meta/json/retrieval_" + category + "_cleaned.json") for category in
        category_list]
    pair_meta_fname_list = [os.path.abspath("../dataset/meta/meta/json/test_pairs_" + category + "_cleaned.json")
                            for category in category_list]
    img_dir_list = [os.path.abspath("../dataset/images/" + category) for category in category_list]
    dataset = Street2ShopDataset(retrieval_meta_fname_list, pair_meta_fname_list, img_dir_list, batch_size)

    model = load_model('model.h5')
    model.summary()
    new_model = Model(model.input, model.layers[3].get_input_at(0))

    // 각 카테고리의 product 이미지에 대해 vector를 계산한다
    img_vec_dicts = list()
    for i, category in enumerate(category_list):
        product_feature_vectors, _ = new_model.predict_generator(dataset.product_x_generator(i)
                                           , steps=math.ceil(dataset.get_num_of_product_in_category(i) / batch_size))

        test_feature_vectors, _ = new_model.predict_generator(dataset.test_x_generator(i)
                                           , steps=math.ceil(dataset.get_num_of_pairs_in_category(i) / batch_size))

if __name__ == '__main__':
    main()