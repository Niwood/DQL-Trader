from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
import datetime
import pandas as pd


class ModifiedTensorBoard(TensorBoard):
    ''' Own Tensorboard class '''

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)



def get_dummy_data():
    import random

    in_array = np.linspace(-np.pi, np.pi*5, 1000) 
    price = np.sin(in_array)+1.021
    high = [i*random.uniform(i, 1.01*i) for i in price]
    low = [i*random.uniform(i, 0.99*i) for i in price]
    base = datetime.datetime.today()
    date_list = [(base - datetime.timedelta(days=x)) for x in range(len(price))]
    date_list.reverse()
    df = pd.DataFrame(data={'close':price, 'open':price, 'high':high, 'low':low}, index=date_list)

    return df


def safe_div(x, y):
    return 0 if y == 0 else x / y