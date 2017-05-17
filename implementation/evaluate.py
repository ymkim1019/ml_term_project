from keras.models import load_model, Model
from keras.layers import Input
from Street2ShopDataset import Street2ShopDataset
import os
import math
from keras import backend as K

batch_size = 32

def main():
    # category_list = ['outerwear', 'pants', 'bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings',
    #                  'skirts', 'tops']
    category_list = ['bags']
    retrieval_meta_fname_list = [
        os.path.abspath("../dataset/meta/meta/json/retrieval_" + category + "_cleaned.json") for category in
        category_list]
    pair_meta_fname_list = [os.path.abspath("../dataset/meta/meta/json/test_pairs_" + category + "_cleaned.json")
                            for category in category_list]
    img_dir_list = [os.path.abspath("../dataset/images/" + category) for category in category_list]
    dataset = Street2ShopDataset(retrieval_meta_fname_list, pair_meta_fname_list, img_dir_list, batch_size)

    model = load_model('model.h5')
    model.summary()
    # model.layers.pop()
    # model.outputs = [model.layers[-2].get_output_at(0)]
    get_2rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[2].output_at(0)])
    layer_output = get_2rd_layer_output([next(dataset.test_x_generator())])[0]

    print('predict start')
    new_model = Model(model.input, model.layers[2].get_output_at(0))
    y = new_model.predict(next(dataset.test_x_generator()))
    print('predict end')

    # model_input = Input(shape=dataset.get_input_dim())
    # model_output = model.layers[2](model_input)
    # # new_model = Model(model.input, model.layers[2].get_output_at(0))
    # new_model = Model(model_input, model_output)
    # print('predict start')
    # y = new_model.predict_generator(dataset.test_x_generator()
    #                                 , steps=math.ceil(dataset.get_num_of_train_samples() / batch_size))
    # print('predict end')

    print(y.type)
    print(y.shape)


if __name__ == '__main__':
    main()