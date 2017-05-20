import json
from os import listdir, getcwd
from os.path import isfile, join, splitext

def main(retrieval_meta_fname, pair_meta_fname_list, img_dir):
    items = set()
    onlyfiles = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    for p in onlyfiles:
        filename, _ = splitext(p)
        items.add(int(filename))

    product_photo_dict = dict()
    with open(retrieval_meta_fname) as f:
        js = json.loads(f.read())
        for each in js:
            product_photo_dict[int(each['product'])] = int(each['photo'])

    for pair_meta_fname in pair_meta_fname_list:
        with open(pair_meta_fname) as f:
            js = json.loads(f.read())
            new_js = [obj for obj in js if int(obj['photo']) in items and product_photo_dict[int(obj['product'])] in items]

        filename, extension = splitext(pair_meta_fname)
        p = filename + '_cleaned' + extension
        with open(p, 'w') as f:
            json.dump(new_js, f)

if __name__ == '__main__':
    category_list = ['pants', 'bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings', 'skirts', 'tops']
    for category in category_list:
        retrieval_meta_fname = "meta\\meta\\json\\retrieval_" + category + ".json"
        pair_meta_fname_list = list()
        pair_meta_fname_list.append("meta\\meta\\json\\train_pairs_" + category + ".json")
        pair_meta_fname_list.append("meta\\meta\\json\\test_pairs_" + category + ".json")
        img_dir = "images\\" + category
        main(retrieval_meta_fname, pair_meta_fname_list, img_dir)
        print(category + ' cleaned..')
