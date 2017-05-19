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

        self.num_of_product_in_category = list()
        self.num_of_pairs_in_category = list()
        self.product_indexes_in_category = list()
        self.test_photo_indexes_in_category = list()
        self.test_product_indexes_in_category = list()

        self.load_pair_meta()

    def load_image(self, path):
        try:
            im = Image.open(path)
            if im.format == 'GIF' or im.format == 'PNG':
                im = im.convert('RGB')
            elif im.format == 'PNG':
                im.load()
                background = Image.new("RGB", im.size, (255, 255, 255))
                background.paste(im, mask=im.split()[-1])  # 3 is the alpha channel
                im = background

            arr = np.asarray(im, dtype="int32")
            return arr
        except Exception as e:
            print(e)
            print(path)
            return None

    def get_input_dim(self):
        return self.hx, self.hy, 3

    def load_pair_meta(self):
        num_of_categories = len(self.retrieval_meta_fname_list)
        for category_idx in range(num_of_categories):
            product_photo_dict = dict()
            with open(self.retrieval_meta_fname_list[category_idx]) as f:
                js = json.loads(f.read())
                self.num_of_product_in_category.append(len(js))
                product_indexes = list()
                for each in js:
                    product_indexes.append(each['photo']) # photo id of product
                    product_photo_dict[each['product']] = each['photo']
                self.product_indexes_in_category.append(product_indexes)

            with open(self.pair_meta_fname_list[category_idx]) as f:
                js = json.loads(f.read())
                self.test_photo_indexes_in_category.append([each['photo'] for each in js])
                self.test_product_indexes_in_category.append([product_photo_dict[each['product']] for each in js])
                n = len(js)
                self.num_of_pairs_in_category.append(n)
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

    def get_test_product_indexes_in_category(self, category_idx):
        return self.test_product_indexes_in_category[category_idx]

    def get_test_photo_indexes_in_category(self, category_idx):
        return self.test_photo_indexes_in_category[category_idx]

    def get_num_of_pairs_in_category(self, category_index):
        return self.num_of_pairs_in_category[category_index]

    def get_product_indexes_in_category(self, category_index):
        return self.product_indexes_in_category[category_index]

    def get_num_of_train_samples(self):
        return len(self.x)

    def get_num_of_validation_samples(self):
        return len(self.x_val)

    def get_num_of_product_in_category(self, category_idx):
        return self.num_of_product_in_category[category_idx]

    def product_x_generator(self, category_idx):
        with open(self.retrieval_meta_fname_list[category_idx]) as f:
            js = json.loads(f.read())
            photo_indexes = [each['photo'] for each in js]

        while True:
            n = len(photo_indexes)
            left = list()
            right = list()
            cnt = 0;
            for i in range(n):
                path = os.path.join(self.img_dir_list[category_idx], "%09d" % int(photo_indexes[i]) + '.jpeg')
                im = self.load_image(path)
                if im not None:
                    left.append(im)
                    right.append(im)  # dummy
                    cnt += 1

                if cnt % self.batch_size == 0 or i == n - 1:
                    left = np.array(left)
                    right = np.array(right)
                    yield [left, right]
                    left = []
                    right = []

    def test_x_generator(self, category_idx):
        photo_indexes = self.test_photo_indexes_in_category[category_idx]

        while True:
            n = len(photo_indexes)
            if n == 0:
                return
            left = []
            right = []
            for i in range(n):
                path = os.path.join(self.img_dir_list[category_idx],
                                    "%09d" % int(photo_indexes[i]) + '.jpeg')
                im_left = self.load_image()
                im_left = Image.open(path)
                if im_left.format == 'GIF' or im_left.format == 'PNG':
                    im_left = im_left.convert('RGB')
                elif im_left.format == 'PNG':
                    # print(im_left.mode, path)
                    im_left.load()
                    background = Image.new("RGB", im_left.size, (255, 255, 255))
                    background.paste(im_left, mask=im_left.split()[-1])  # 3 is the alpha channel
                    im_left = background

                left.append(np.asarray(im_left, dtype="int32"))
                right.append(np.asarray(im_left, dtype="int32")) # dummy

                if (i+1) % self.batch_size == 0 or i == n-1:
                    left = np.array(left)
                    right = np.array(right)
                    yield [left, right]
                    left = []
                    right = []

    def train_pair_generator(self):
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
                    # print(im_left.mode, path)
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
