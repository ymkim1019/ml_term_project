import json
import numpy as np
from tomorrow import threads
import imghdr
import requests
import os
import logging
from PIL import Image
import io

photo_url_dict = dict()
product_photo_dict = dict()

def split(x):
    first = x.find(',')
    return (x[:first], x[first+1:])

def main(url_fname, retrieval_meta_fname, pair_meta_fname_list, save_dir):
    @threads(20)
    def download(item_id, url, cnt, n, images_dir, source_fname, bbox=None):
        if bbox is not None:
            width, top, height, left = bbox

        try:
            r = requests.get(url)
            if r.status_code == 200:
                image_type = imghdr.what(None, r.content)
                if image_type is not None:
                    p = os.path.join(images_dir, "%09d"%int(item_id) + '.' + image_type)
                    # print(p)
                    with open(p, 'wb') as f:
                        f.write(r.content)
                        f.close()
                    # resize
                    im = Image.open(p)
                    if bbox is None:
                        # product image
                        im.thumbnail((256, 256))
                    else:
                        # user image
                        im = im.crop((left, top, left+width, top+height))
                        im.thumbnail((256, 256))

                    background = Image.new('RGB', (299, 299), "white")
                    offset = (int((299 - im.width) / 2), int((299 - im.height) / 2))
                    background.paste(im, offset)
                    im.close()
                    background.save(p)
                # else:
                #     logging.error('%s\t%s\tunknown_type' % (item_id, url))
            else:
                logging.error('%s\t%s\tstatus:%d' % (item_id, url, r.status_code))
        except Exception as e:
            print(e)
            print("Unexpected error:", r.status_code)

        if cnt % 200 == 0:
            print(str.format("{} - {}/{}", source_fname, cnt+1, n))

    with open(url_fname) as f:
        itr = enumerate(f)

        for i, line in itr:
            [item_id, url] = split(line.strip())
            photo_url_dict[int(item_id)] = url

        f.close()

    print('url list loaded..')

    with open(retrieval_meta_fname) as f:
        js = json.loads(f.read())
        n = len(js)
        for cnt, each in enumerate(js):
            photo_id = int(each['photo'])
            photo_url = photo_url_dict[photo_id]
            download(photo_id, photo_url, cnt, n, save_dir, retrieval_meta_fname)

    print('product retrieval info loaded..')

    for meta_fname in pair_meta_fname_list:
        with open(meta_fname, 'r') as f:
            js = json.loads(f.read())
            n = len(js)
            for cnt, each in enumerate(js):
                # # product photo
                # product_photo_id = product_photo_dict[int(js_obj["product"])]
                # product_photo_url = photo_url_dict[product_photo_id]
                # download(product_photo_id, product_photo_url, cnt, n, save_dir)

                # user photo
                user_photo_id = int(each["photo"])
                user_photo_url = photo_url_dict[user_photo_id]
                width = int(each["bbox"]["width"])
                top = int(each["bbox"]["top"])
                height = int(each["bbox"]["height"])
                left = int(each["bbox"]["left"])
                download(user_photo_id, user_photo_url, cnt, n, save_dir, meta_fname, bbox=(width, top, height, left))

if __name__ == '__main__':
    category_list = ['outerwear', 'pants', 'bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings', 'skirts', 'tops']
    for category in category_list:
        url_fname = "photos\\photos.txt"
        retrieval_meta_fname = "meta\\meta\\json\\retrieval_" + category + ".json"
        pair_meta_fname_list = list()
        pair_meta_fname_list.append("meta\\meta\\json\\test_pairs_" + category + ".json")
        pair_meta_fname_list.append("meta\\meta\\json\\train_pairs_" + category + ".json")
        save_dir = "images\\" + category
        if os.path.isdir(save_dir) is False:
            os.mkdir(save_dir)

        main(url_fname, retrieval_meta_fname, pair_meta_fname_list, save_dir)
