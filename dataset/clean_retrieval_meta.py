import json
from os import listdir, getcwd
from os.path import isfile, join, splitext

def main(retrieval_meta_fname, img_dir):
    items = set()
    onlyfiles = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    for p in onlyfiles:
        filename, _ = splitext(p)
        items.add(int(filename))

    with open(retrieval_meta_fname) as f:
        js = json.loads(f.read())
        new_js = [obj for obj in js if int(obj['photo']) in items]

    filename, extension = splitext(retrieval_meta_fname)
    p = filename + '_cleaned' + extension
    with open(p, 'w') as f:
        json.dump(new_js, f)

if __name__ == '__main__':
    category_list = ['pants', 'outwear', 'bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings', 'skirts', 'tops']
    for category in category_list:
        retrieval_meta_fname = "meta\\meta\\json\\retrieval_" + category + ".json"
        img_dir = "images\\" + category
        main(retrieval_meta_fname, img_dir)
        print(category + ' cleaned..')
