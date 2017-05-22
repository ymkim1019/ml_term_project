# -*- coding: utf-8 -*-

from keras.models import load_model, Model
from keras.layers import Input
from Street2ShopDataset import Street2ShopDataset
import os
import math
from keras import backend as K
from PIL import Image
import numpy as np
from scipy import spatial

batch_size = 32
margin = 40.0

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''

    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.maximum(K.square(margin) - K.square(y_pred), 0))

def robust_contrastive_loss(y_true, y_pred):
    return K.mean(y_true * K.square(K.minimum(y_pred, margin)) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def euclidean_distance(vects):
    x, y = vects
    return math.sqrt(max(np.sum((x - y)**2), 1e-07))

def cos_distance(vects):
    x, y = vects
    return spatial.distance.cosine(x, y)

def main():
    # category_list = ['outerwear', 'pants', 'bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings',
    #                  'skirts', 'tops']
    category_list = ['belts']
    retrieval_meta_fname_list = [
        os.path.abspath("../dataset/meta/meta/json/retrieval_" + category + "_cleaned.json") for category in
        category_list]
    pair_meta_fname_list = [os.path.abspath("../dataset/meta/meta/json/train_pairs_" + category + "_cleaned.json")
                            for category in category_list]
    img_dir_list = [os.path.abspath("../dataset/images/" + category) for category in category_list]
    dataset = Street2ShopDataset(retrieval_meta_fname_list, pair_meta_fname_list, img_dir_list, batch_size)

    model = load_model('model.h5', custom_objects={'contrastive_loss': contrastive_loss})
    print('model loaded..')
    model.load_weights('weights-belts.hdf5')
    print('weights loaded..')
    model.summary()
    new_model = Model(model.input, model.layers[3].get_input_at(0))

    for i, category in enumerate(category_list):
        num_of_products = dataset.get_num_of_product_in_category(i)
        num_of_test_photos = dataset.get_num_of_pairs_in_category(i)

        print(str.format('--------- {}', category_list[i]))
        # get feature vectors of product images in given category
        print('retrieving feature vectors of product images..# of products =', num_of_products)
        product_feature_vectors, _ = new_model.predict_generator(dataset.product_x_generator(i)
                                           , steps=math.ceil(num_of_products / batch_size)
                                            , verbose=1)

        # print(len(product_feature_vectors), product_feature_vectors[0].shape)

        # get feature vectors of the test test
        print('retrieving feature vectors of test images..# of test images =', num_of_test_photos)
        test_feature_vectors, _ = new_model.predict_generator(dataset.test_x_generator(i)
                                           , steps=math.ceil(num_of_test_photos / batch_size)
                                            , verbose=1)

        # print(len(test_feature_vectors), test_feature_vectors[0].shape)
        product_photo_indexes = dataset.get_product_indexes_in_category(i)
        test_photo_indexes = dataset.get_test_photo_indexes_in_category(i)
        test_y = dataset.get_test_product_indexes_in_category(i)
        result = np.zeros(num_of_test_photos)
        for j in range(num_of_test_photos):
            distances = [euclidean_distance((test_feature_vectors[j], product_feature)) for product_feature in product_feature_vectors]
            sorted_indexes = [k[0] for k in sorted(enumerate(distances), key=lambda x: x[1])][0:20]
            top20_items = [product_photo_indexes[idx] for idx in sorted_indexes]
            print([distances[k] for k in sorted_indexes])
            print(str.format('test item = {}, y = {}, top 20 = {}', test_photo_indexes[j], test_y[j], top20_items))
            result[j] = 1 if test_y[j] in top20_items else 0
        print(str.format('top 20 retrieval accuracy = {}', sum(result) / num_of_test_photos))


if __name__ == '__main__':
    main()