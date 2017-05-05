import numpy as np
from keras.datasets import cifar10

class Cifar10DataSet:
    def __init__(self, training_size = 640, validation_size = 64, test_size = 192):
        self.num_of_items_per_class = 6000
        self.training_size = training_size
        self.validation_size = validation_size
        self.test_size = test_size

        assert self.training_size % 2 == 0
        assert self.validation_size % 2 == 0
        assert self.test_size % 2 == 0
        assert self.training_size + self.validation_size <= self.num_of_items_per_class // 2
        assert self.test_size <= self.num_of_items_per_class // 2

        self.x_train = []
        self.y_train = []
        self.x_validation = []
        self.y_validation = []
        self.x_test = []
        self.y_test = []
        self.oversampling_ratio = 9
        self.num_of_classes = 10
        self.hx = 32
        self.hy = 32
        self.input_dim = (self.hx * self.oversampling_ratio, self.hy * self.oversampling_ratio, 3)
        self.load_data()

    # return the dim of the input data
    def get_input_dim(self):
        return self.input_dim

    def load_data(self):
        '''
        Cifar10 image loading
        '''
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        digit_indices = [np.where(y_train == i)[0] for i in range(10)]

        # training set
        self.x_train = x_train[digit_indices[0][0:self.training_size]]
        self.y_train = y_train[digit_indices[0][0:self.training_size]]
        for c in range(1, self.num_of_classes):
            self.x_train = np.vstack((self.x_train, x_train[digit_indices[c][0:self.training_size//2]]))
            self.y_train = np.vstack((self.y_train, y_train[digit_indices[c][0:self.training_size//2]]))
        self.y_train = self.y_train.reshape((self.y_train.shape[0]))

        # validation set
        self.x_validation = x_train[digit_indices[0][self.training_size:(self.training_size + self.validation_size)]]
        self.y_validation = y_train[digit_indices[0][self.training_size:(self.training_size + self.validation_size)]]
        for c in range(1, self.num_of_classes):
            self.x_validation = np.vstack((self.x_validation, x_train[digit_indices[c][self.training_size//2:self.training_size//2 + self.validation_size//2]]))
            self.y_validation = np.vstack((self.y_validation, y_train[digit_indices[c][self.training_size//2:self.training_size//2 + self.validation_size//2]]))
        self.y_validation = self.y_validation.reshape((self.y_validation.shape[0]))

        # test set
        digit_indices = [np.where(y_test == i)[0] for i in range(10)]
        self.x_test = x_test[digit_indices[0][0:self.test_size]]
        self.y_test = y_test[digit_indices[0][0:self.test_size]]
        for c in range(1, self.num_of_classes):
            self.x_test = np.vstack((self.x_test, x_test[digit_indices[c][0:self.test_size//2]]))
            self.y_test = np.vstack((self.y_test, y_test[digit_indices[c][0:self.test_size//2]]))
        self.y_test = self.y_test.reshape((self.y_test.shape[0]))

    def pair_generator(self, x, y, dataset_size, batch_size=32):
        while True:
            digit_indices = []
            for c in range(self.num_of_classes):
                indices = np.where(y == c)[0]
                digit_indices.append(np.random.permutation(indices))

            left = []
            right = []
            labels = []
            for i in range(dataset_size//2):
                # positive pair
                left.append(x[digit_indices[0][i]])
                right.append(x[digit_indices[0][i + dataset_size // 2]])
                # negative pair
                left.append(x[digit_indices[0][i]])
                right.append(x[digit_indices[np.random.randint(1, self.num_of_classes)][i]])
                # label
                labels.append(1)
                labels.append(0)
                if (i+1)*2 % batch_size == 0:
                    left = np.array(left)
                    right = np.array(right)
                    labels = np.array(labels)
                    # oversampling
                    left = left.repeat(self.oversampling_ratio, axis=1)
                    left = left.repeat(self.oversampling_ratio, axis=2)
                    right = right.repeat(self.oversampling_ratio, axis=1)
                    right = right.repeat(self.oversampling_ratio, axis=2)
                    yield [left, right], labels
                    left = []
                    right = []
                    labels = []

def main():
    dataset = Cifar10DataSet()

    for x, y in dataset.pair_generator(dataset.x_train, dataset.y_train, dataset.training_size, 32):
        print(x[0].shape, x[1].shape, y.shape)

    print("done")

if __name__ == '__main__':
    main()