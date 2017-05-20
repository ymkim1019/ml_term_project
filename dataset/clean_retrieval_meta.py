import json
from os.path import join
from PIL import Image
import numpy as np


def load_image(path):
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
        print('img load error.. :', path)
        return None


def main(retrieval_meta_fname, img_dir):
    with open(retrieval_meta_fname) as f:
        js = json.loads(f.read())
        new_js = []
        n = len(js)
        for i, each in enumerate(js):
            if i % 500 == 0:
                print(i+1, '/', n);
            if load_image(join(img_dir, "%09d" % int(each['photo']) + '.jpeg')) is not None:
                new_js.append(each)

    with open(retrieval_meta_fname, 'w') as f:
        json.dump(new_js, f)


if __name__ == '__main__':
    category_list = ['pants', 'outerwear', 'bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings', 'skirts', 'tops']
    #category_list = ['belts']
    for category in category_list:
        retrieval_meta_fname = "meta/meta/json/retrieval_" + category + "_cleaned.json"
        img_dir = "images/" + category
        main(retrieval_meta_fname, img_dir)
        print(category + ' cleaned..')
