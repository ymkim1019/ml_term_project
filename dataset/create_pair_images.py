import json
from os import listdir
from os.path import isfile, join, splitext
from PIL import Image
import numpy as np
import os


def main(retrieval_meta_fname, pair_meta_fname_list, img_dir):
    product_photo_dict = dict()

    with open(retrieval_meta_fname) as f:
        js = json.loads(f.read())
        for each in js:
            product_photo_dict[each['product']] = each['photo']

    for pair_meta_fname in pair_meta_fname_list:
        with open(pair_meta_fname) as f:
            js = json.loads(f.read())
            new_js = []
            n = len(js)
            for i, each in enumerate(js):
                if i % 500 == 0:
                    print(img_dir, ':', i+1, '/', n)
                street_path = join(img_dir, "%09d" % int(each['photo']) + '.jpeg')
                shop_path = join(img_dir, "%09d" % int(product_photo_dict[each['product']]) + '.jpeg')

                im = Image.open(street_path)
                if im.format == 'GIF' or im.format == 'PNG':
                    im = im.convert('RGB')
                elif im.format == 'PNG':
                    im.load()
                    background = Image.new("RGB", im.size, (255, 255, 255))
                    background.paste(im, mask=im.split()[-1])  # 3 is the alpha channel
                    im = background

                im2 = Image.open(shop_path)
                if im2.format == 'GIF' or im.format == 'PNG':
                    im2 = im2.convert('RGB')
                elif im2.format == 'PNG':
                    im2.load()
                    background = Image.new("RGB", im.size, (255, 255, 255))
                    background.paste(im2, mask=im2.split()[-1])  # 3 is the alpha channel
                    im2 = background

                new_img = Image.new('RGB', (598, 299), "white")
                new_img.paste(im, (0, 0))
                new_img.paste(im2, (299, 0))
                p = os.path.join(img_dir+'/test_pair', str.format('{}_{}_product_id_{}.jpeg', each['photo']
                                                             , product_photo_dict[each['product']], each['product']))
                new_img.save(p)

if __name__ == '__main__':
    #category_list = ['pants', 'outerwear', 'bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings', 'skirts', 'tops']
    category_list = ['dresses']
    for category in category_list:
        retrieval_meta_fname = "meta/meta/json/retrieval_" + category + "_cleaned.json"
        pair_meta_fname_list = list()
        #pair_meta_fname_list.append("meta/meta/json/train_pairs_" + category + "_cleaned.json")
        pair_meta_fname_list.append("meta/meta/json/test_pairs_" + category + "_cleaned.json")
        img_dir = "images/" + category
        main(retrieval_meta_fname, pair_meta_fname_list, img_dir)
        print(category + ' cleaned..')
