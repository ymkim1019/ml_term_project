# -*- coding: utf-8 -*-

from keras.models import load_model, Model
from keras.layers import Input
from Street2ShopDataset import Street2ShopDataset
from keras.layers import Input, Dense, concatenate
import os
import math
from keras import backend as K
from keras import layers
from PIL import Image
import numpy as np
import heapq

batch_size = 32

def euclidean_distance(vects):
    x, y = vects
    return math.sqrt(max(np.sum((x - y)**2), 1e-07))

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
    dataset = Street2ShopDataset(retrieval_meta_fname_list, pair_meta_fname_list, img_dir_list, batch_size)

    #model = load_model('model.h5', custom_objects={'robust_contrastive_loss': robust_contrastive_loss})
    model = load_model('model.h5')
    print('model loaded..')
    #model.load_weights('weights-improvement-allcategories.hdf5')
    #print('weights loaded..')
    model.summary()
    print(model.layers[3].get_input_at(0))
    new_model = Model(model.input, model.layers[3].get_input_at(0))

    input_a = Input(shape=(2048,))
    input_b = Input(shape=(2048,))
    merged_vector = concatenate([input_a, input_b], axis=-1)
    metriclayer1 = Dense(2048, activation='relu')(merged_vector)
    metriclayer2 = Dense(2048, activation='relu')(metriclayer1)
    prediction = Dense(1, activation='sigmoid')(metriclayer2)
    metric_model = Model([input_a, input_b], prediction)

    metric_model.summary()

    metric_model.layers[3].set_weights(model.layers[4].get_weights())
    metric_model.layers[4].set_weights(model.layers[5].get_weights())
    metric_model.layers[5].set_weights(model.layers[6].get_weights())




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
        #np.save("product_feature_Vectors.npy",product_feature_vectors)
        #np.savetxt("product_feature_Vectors.txt", product_feature_vectors)

        # get feature vectors of the test test
        print('retrieving feature vectors of test images..# of test images =', num_of_test_photos)
        test_feature_vectors, _ = new_model.predict_generator(dataset.test_x_generator(i)
                                           , steps=math.ceil(num_of_test_photos / batch_size)
                                            , verbose=1)
        #np.save("test_feature_Vectors.npy", test_feature_vectors)
        #np.savetxt("test_feature_Vectors.txt", test_feature_vectors)
        # print(len(test_feature_vectors), test_feature_vectors[0].shape)

        product_photo_indexes = dataset.get_product_indexes_in_category(i)
        test_photo_indexes = dataset.get_test_photo_indexes_in_category(i)
        test_y = dataset.get_test_product_indexes_in_category(i)
        result = np.zeros(num_of_test_photos)
        product_feature_vectors = np.array(product_feature_vectors)

        for j in range(num_of_test_photos):
            test_feature = test_feature_vectors[j]
            test_feature_metrix = np.array(test_feature)
            test_feature_metrix = np.reshape(test_feature_metrix,(1,2048))
            test_feature_repeat = np.repeat(test_feature_metrix,product_feature_vectors.shape[0],axis=0)

            #pair_metrix = np.concatenate((test_feature_repeat,product_feature_vectors,axis=1)
            similarity = metric_model.predict([test_feature_repeat,product_feature_vectors],batch_size=1024)
            sorted_indexes = [k[0] for k in sorted(enumerate(similarity), key=lambda x: x[1])][-20:product_feature_vectors.shape[0]]
            top20_items = [product_photo_indexes[idx] for idx in sorted_indexes]
            #print([distances[k] for k in sorted_indexes])
            #print(str.format('test item = {}, y = {}, top 20 = {}', test_photo_indexes[j], test_y[j], top20_items))
            result[j] = 1 if test_y[j] in top20_items else 0
        print(str.format('top 20 retrieval accuracy = {}', sum(result) / num_of_test_photos))


if __name__ == '__main__':
    main()