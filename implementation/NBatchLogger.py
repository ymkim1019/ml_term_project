import datetime
from keras.callbacks import Callback

class NBatchLogger(Callback):
    def printProgressBar(self, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()

    def on_train_begin(self, logs={}):
        self.cnt = 0
        print("train begin at ", datetime.datetime.now())

    def on_epoch_end(self, epoch, logs=None):
        self.cnt = 0
        print("epoch ends at ", datetime.datetime.now())

    def on_batch_end(self,batch,logs={}):
        # print(self.params)
        self.cnt += 1
        # print('on_batch_end:', self.cnt, self.params['steps'], self.params['metrics'][0])
        self.printProgressBar(self.cnt, self.params['steps'], prefix='batch:', suffix='Complete', length=50)
        # self.batch_cnt += 1
