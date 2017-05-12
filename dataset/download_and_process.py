    import json
    from tomorrow import threads
    import imghdr
    import requests
    import os
    from itertools import islice
    import sys

    def split(x):
        first = x.find(',')
        return (x[:first], x[first+1:])

    def main(url_fname, meta_fname, save_dir):
        @threads(10)
        def download(item_id, url, i, n, images_dir=''):
            try:
                r = requests.get(url)

                if r.status_code == 200:

                    image_type = imghdr.what(None, r.content)

                    if image_type is not None:
                        p = os.path.join(images_dir, str(item_id) + '.' + image_type)
                        #print(p)
                        with open(p, 'wb') as f:
                            f.write(r.content)
                            f.close()
                    else:
                        logging.error('%s\t%s\tunknown_type' % (item_id, url))
                else:
                    logging.error('%s\t%s\tstatus:%d' % (item_id, url, r.status_code))

            except:
                print("Unexpected error:", r.status_code)

            if i % 200 == 0:
                print(str.format("{}/{}", i, n))

        js = None
        url_dict = dict()
        download_set = set()

        with open(url_fname) as f:
            itr = enumerate(f)

            for i, line in itr:
                [item_id, url] = split(line.strip())
                url_dict[int(item_id)] = url

            f.close()

        with open(meta_fname, 'r') as f:
            js = json.loads(f.read())
            for each in js:
                download_set.add(int(each['photo']))
                download_set.add(int(each['product']))

            f.close()

        n = len(download_set)
        for i, photo_id in enumerate(download_set):
            download(photo_id, url_dict[photo_id], i, n, images_dir=save_dir)


    if __name__ == '__main__':
        # main(sys.args[1])
        url_fname = u"C:\\Users\\user\\PycharmProjects\\ml_term_project-master\\dataset\\photos\\photos.txt"
        meta_fname = u"C:\\Users\\user\\PycharmProjects\\ml_term_project-master\\dataset\\meta\\meta\\json\\test_pairs_outerwear.json"
        save_dir = u"C:\\Users\\user\\PycharmProjects\\ml_term_project-master\\dataset\\images\\outerwear"
        main(url_fname, meta_fname, save_dir)
