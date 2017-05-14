# -*- coding: utf-8 -*-

import json
import numpy as np
import math
import os
from PIL import Image


class Street2ShopDataset:
    def __init__(self, retrieval_meta_fname_list, pair_meta_fname_list, img_dir_list, batch_size=32):
        self.retrieval_meta_fname_list = retrieval_meta_fname_list
        self.pair_meta_fname_list = pair_meta_fname_list
        self.img_dir_list = img_dir_list
        self.product_photo_dict = dict()
        self.x = []
        self.y = []
        self.batch_size = batch_size

        self.load_pair_meta()

    def get_input_dim(self):
        return 299, 299, 3

    def load_pair_meta(self):
        num_of_categories = len(self.retrieval_meta_fname_list)
        for category_idx in range(num_of_categories):
            product_photo_dict = dict()
            with open(self.retrieval_meta_fname_list[category_idx]) as f:
                js = json.loads(f.read())
                for each in js:
                    product_photo_dict[each['product']] = each['photo']

            with open(self.pair_meta_fname_list[category_idx]) as f:
                js = json.loads(f.read())
                n = len(js)
                for i in range(n):
                    # positive pair
                    positive_pair = [js[i]['photo'], product_photo_dict[js[i]['product']], category_idx]
                    # negative pair
                    temp = np.random.randint(n)
                    while i == temp:
                        temp = np.random.randint(n)
                    negative_pair = [js[i]['photo'], product_photo_dict[js[temp]['product']], category_idx]
                    self.x.append(positive_pair)
                    self.x.append(negative_pair)
                    self.y.append(1)
                    self.y.append(0)

        print('load_pair_meta done..')

    def getImgPath(self, photo_id, img_dir):
        for root, dirs, files in os.walk(img_dir):
            temp = [fname for fname in files if "%09d"%int(photo_id) in fname]

            if len(temp) == 0:
                print('getImgPath fail!! :', photo_id)
                return None
            else:
                return os.path.join(root, temp[0])

    def get_num_of_samples(self):
        return len(self.x)


    def pair_generator(self):
        while True:
            n = len(self.x)
            indexes = np.random.randint(n, size=n) # shuffle
            left = []
            right = []
            labels = []
            for i in range(n):
                path = self.getImgPath(self.x[indexes[i]][0], self.img_dir_list[self.x[indexes[i]][2]])
                im_left = Image.open(path)
                im_left.load()
                path = self.getImgPath(self.x[indexes[i]][1], self.img_dir_list[self.x[indexes[i]][2]])
                im_right = Image.open(path)
                im_right.load()
                left.append(np.asarray(im_left, dtype="int32"))
                right.append(np.asarray(im_right, dtype="int32"))
                labels.append(self.y[indexes[i]])

                if (i+1) % self.batch_size == 0 or i == n-1:
                    left = np.array(left)
                    right = np.array(right)
                    labels = np.array(labels)
                    yield [left, right], labels
                    left = []
                    right = []
                    labels = []

if __name__ == '__main__':
    category_list = ['outerwear', 'pants', 'bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings',
                     'skirts', 'tops']
    retrieval_meta_fname_list = [os.path.abspath("..\\dataset\\meta\\meta\\json\\retrieval_" + category + "_cleaned.json") for category in category_list]
    pair_meta_fname_list = [os.path.abspath("..\\dataset\\meta\\meta\\json\\train_pairs_" + category + "_cleaned.json") for category in category_list]
    img_dir_list = [os.path.abspath("..\\dataset\\images\\" + category) for category in category_list]

    dataset = Street2ShopDataset(retrieval_meta_fname_list, pair_meta_fname_list, img_dir_list)
    print(next(dataset.pair_generator()))
