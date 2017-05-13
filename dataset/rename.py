import json
import os

def main(img_dir):
    for root, dirs, files in os.walk(img_dir):
        for f in files:
            filename, extension = os.path.splitext(f)
            os.rename(os.path.join(root, f), os.path.join(root, str("%09d" % int(filename))+extension))

if __name__ == '__main__':
    #category_list = ['pants', 'outerwear', 'bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings', 'skirts', 'tops']
    category_list = ['outerwear']
    for category in category_list:
        img_dir = "images\\" + category
        main(img_dir)
        print('files in '+category+' have been renamed..')
