# -*- coding: utf-8 -*-

import json
import numpy as np
import math
import os
from PIL import Image


class Street2ShopDataset:
    def __init__(self, retrieval_meta_fname_list, pair_meta_fname_list, img_dir_list, batch_size=32, validation_size=None):
        self.retrieval_meta_fname_list = retrieval_meta_fname_list
        self.pair_meta_fname_list = pair_meta_fname_list
        self.img_dir_list = img_dir_list
        self.product_photo_dict = dict()
        self.x = []
        self.x_val = []
        self.y = []
        self.y_val = []
        self.batch_size = batch_size
        self.validation_size = validation_size # should be less than 1.0
        self.hx = 299
        self.hy = 299

        self.load_pair_meta()

    def get_input_dim(self):
        return self.hx, self.hy, 3

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
                val_interval = None if self.validation_size is None else np.ceil(1.0/self.validation_size)
                for i in range(n):
                    # positive pair
                    positive_pair = [js[i]['photo'], product_photo_dict[js[i]['product']], category_idx]
                    # negative pair
                    temp = np.random.randint(n)
                    while i == temp:
                        temp = np.random.randint(n)
                    negative_pair = [js[i]['photo'], product_photo_dict[js[temp]['product']], category_idx]
                    if val_interval is not None and i % val_interval == 0:
                        self.x_val.append(positive_pair)
                        self.x_val.append(negative_pair)
                        self.y_val.append(1)
                        self.y_val.append(0)
                    else:
                        self.x.append(positive_pair)
                        self.x.append(negative_pair)
                        self.y.append(1)
                        self.y.append(0)

        print('load_pair_meta done..')

    def get_num_of_test_samples(self):
        return len(self.x)

    def get_num_of_validation_samples(self):
        return len(self.x_val)

    def test_pair_generator(self):
        while True:
            n = len(self.x)
            if n == 0:
                break
            indexes = np.random.randint(n, size=n) # shuffle
            left = []
            right = []
            labels = []
            for i in range(n):
                path = os.path.join(self.img_dir_list[self.x[indexes[i]][2]],
                                    "%09d" % int(self.x[indexes[i]][0]) + '.jpeg')
                im_left = Image.open(path)
                if im_left.format == 'GIF' or im_left.format == 'PNG':
                    im_left = im_left.convert('RGB')
                elif im_left.format == 'PNG':
                    print(im_left.mode, path)
                    im_left.load()
                    background = Image.new("RGB", im_left.size, (255, 255, 255))
                    background.paste(im_left, mask=im_left.split()[-1])  # 3 is the alpha channel
                    im_left = background

                path = os.path.join(self.img_dir_list[self.x[indexes[i]][2]],
                                    "%09d" % int(self.x[indexes[i]][1]) + '.jpeg')
                im_right = Image.open(path)
                if im_right.format == 'GIF' or im_right.format == 'PNG':
                    im_right = im_right.convert('RGB')
                elif im_right.format == 'PNG':
                    # print(im_right.mode, path)
                    im_right.load()
                    background = Image.new("RGB", im_right.size, (255, 255, 255))
                    background.paste(im_right, mask=im_right.split()[-1])  # 3 is the alpha channel
                    im_right = background
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

    def validation_pair_generator(self):
        while True:
            n = len(self.x_val)
            if n == 0:
                break
            indexes = np.random.randint(n, size=n) # shuffle
            left = []
            right = []
            labels = []
            for i in range(n):
                path = os.path.join(self.img_dir_list[self.x_val[indexes[i]][2]],
                                    "%09d" % int(self.x_val[indexes[i]][0]) + '.jpeg')
                im_left = Image.open(path)
                if im_left.format == 'GIF' or im_left.format == 'PNG':
                    im_left = im_left.convert('RGB')
                elif im_left.format == 'PNG':
                    # print(im_left.mode, path)
                    im_left.load()
                    background = Image.new("RGB", im_left.size, (255, 255, 255))
                    background.paste(im_left, mask=im_left.split()[-1])  # 3 is the alpha channel
                    im_left = background

                path = os.path.join(self.img_dir_list[self.x_val[indexes[i]][2]],
                                    "%09d" % int(self.x_val[indexes[i]][1]) + '.jpeg')
                im_right = Image.open(path)
                if im_right.format == 'GIF' or im_right.format == 'PNG':
                    im_right = im_right.convert('RGB')
                elif im_right.format == 'PNG':
                    # print(im_right.mode, path)
                    im_right.load()
                    background = Image.new("RGB", im_right.size, (255, 255, 255))
                    background.paste(im_right, mask=im_right.split()[-1])  # 3 is the alpha channel
                    im_right = background
                left.append(np.asarray(im_left, dtype="int32"))
                right.append(np.asarray(im_right, dtype="int32"))
                labels.append(self.y_val[indexes[i]])

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
    retrieval_meta_fname_list = [os.path.abspath("../dataset/meta/meta/json/retrieval_" + category + "_cleaned.json") for category in category_list]
    pair_meta_fname_list = [os.path.abspath("../dataset/meta/meta/json/train_pairs_" + category + "_cleaned.json") for category in category_list]
    img_dir_list = [os.path.abspath("../dataset/images/" + category) for category in category_list]

    dataset = Street2ShopDataset(retrieval_meta_fname_list, pair_meta_fname_list, img_dir_list)
    for i in range(math.ceil(dataset.get_num_of_samples()/32)):
        print(i, '/', math.ceil(dataset.get_num_of_samples()/32))
        next(dataset.pair_generator())
