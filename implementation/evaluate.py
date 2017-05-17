# -*- coding: utf-8 -*-

from keras.models import load_model, Model
from keras.layers import Input
from Street2ShopDataset import Street2ShopDataset
import os
import math
from keras import backend as K
from PIL import Image
import numpy as np

batch_size = 32
margin = 40

def robust_contrastive_loss(y_true, y_pred):
    return K.mean(y_true * K.square(K.minimum(y_pred, margin)) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def main():
    # category_list = ['outerwear', 'pants', 'bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings',
    #                  'skirts', 'tops']
    category_list = ['eyewear']
    retrieval_meta_fname_list = [
        os.path.abspath("../dataset/meta/meta/json/retrieval_" + category + "_cleaned.json") for category in
        category_list]
    pair_meta_fname_list = [os.path.abspath("../dataset/meta/meta/json/test_pairs_" + category + "_cleaned.json")
                            for category in category_list]
    img_dir_list = [os.path.abspath("../dataset/images/" + category) for category in category_list]
    dataset = Street2ShopDataset(retrieval_meta_fname_list, pair_meta_fname_list, img_dir_list, batch_size)

    model = load_model('model.h5', custom_objects={'robust_contrastive_loss': robust_contrastive_loss})
    model.summary()
    new_model = Model(model.input, model.layers[3].get_input_at(0))

    for i, category in enumerate(category_list):
        print(str.format('--------- {}', category_list[i]))
        # get feature vectors of product images in given category
        print('retrieving feature vectors of product images..# of products =', dataset.get_num_of_product_in_category(i))
        product_feature_vectors, _ = new_model.predict_generator(dataset.product_x_generator(i)
                                           , steps=math.ceil(dataset.get_num_of_product_in_category(i) / batch_size)
                                            , verbose=1)

        print(len(product_feature_vectors), product_feature_vectors[0].shape)

        # get feature vectors of the test test
        print('retrieving feature vectors of test images..# of test images =', dataset.get_num_of_pairs_in_category(i))
        test_feature_vectors, _ = new_model.predict_generator(dataset.test_x_generator(i)
                                           , steps=math.ceil(dataset.get_num_of_pairs_in_category(i) / batch_size)
                                            , verbose=1)

        print(len(test_feature_vectors), test_feature_vectors[0].shape)



if __name__ == '__main__':
    main()