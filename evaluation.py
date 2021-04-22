from tensorflow import keras
from data_loader import DataCluster
from environment import StockTradingEnv
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow.keras.backend as K
from keras.models import Sequential
from keras.layers import LSTM, Input
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, SGD
from keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from keras.layers.merge import concatenate
from keras.layers import Flatten, BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Model

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class WeightedCategoricalCrossentropy(CategoricalCrossentropy):

    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        cost_matrix: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        cost_matrix = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        model.compile(loss=WeightedCategoricalCrossentropy(cost_matrix), ...)
    """

    def __init__(self, name='WeightedCategoricalCrossentropy', **kwargs):
        # assert cost_mat.ndim == 2
        # assert cost_mat.shape[0] == cost_mat.shape[1]

        self.cost_mat = np.ones((3,3))
        minor_cost = 1.5
        self.cost_mat[0, 1] = minor_cost
        self.cost_mat[0, 2] = minor_cost
        self.cost_mat[1, 0] = minor_cost
        self.cost_mat[2, 0] = minor_cost

        super().__init__(name=name, **kwargs)
        self.cost_mat = K.cast_to_floatx(self.cost_mat)
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        assert sample_weight is None, "should only be derived from the cost matrix"

        return super().__call__(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=get_sample_weights(y_true, y_pred, self.cost_mat),
        )


class ModelAssessment:


    def __init__(
        self,
        collection=None,
        model_shape=tuple(),
        num_time_steps=None,
        sim_range=300
        ):
        
        # Parameters
        self.st_shape = model_shape[0]
        self.lt_shape = model_shape[1]
        self.num_time_steps = num_time_steps
        self.sim_range = sim_range
        self.astats = pd.DataFrame()
        self.model = None

        # Env
        self.env = StockTradingEnv(
            collection,
            look_back_window=num_time_steps,
            static_initial_step=0)


    def load_model(self, model_name=None):
        # Load model
        self.model_name = model_name
        self.model = keras.models.load_model(
            self.model_name
            )



    def _baseline_model(self):
        ''' Returns a baseline model with randomly permuteed weights  '''
        model = keras.models.load_model(self.model_name)
        weights = model.get_weights()
        weights = [np.random.permutation(weight.flat).reshape(weight.shape) for weight in weights]
        model.set_weights(weights)
        return model



    def simulate(self):

        # Reset env
        obs, _ = self.env.reset()


        # Parameters to save
        self.ticker = self.env.ticker
        self.actions = list()
        self.rewards = list()
        self.prices = list()
        self.sim = self.env.dp.org
        self.sim['trigger'] = 0


        for _ in tqdm(range(self.sim_range), desc=f'Model assessment on {self.ticker}'):
            try:
                _obs_st = obs['st'].reshape((1,) + self.st_shape)
                _obs_lt = obs['lt'].reshape((1,) + self.lt_shape)
            except Exception as e:
                print('-'*10)
                print(e)
                print('COULD NOT RESHAPE')
                print('LEN: ', obs['st'].shape, obs['lt'].shape)
                quit()

            # Make prediction
            action = self.model.predict([_obs_st, _obs_lt])

            if np.isnan(np.sum(action)):
                print('Action contains nan [in evaluation]: ',action), quit()

            # Step env
            obs, reward, done = self.env.step(np.argmax(action))

            # Save parameters
            action = np.argmax(action)
            self.actions.append(action)
            self.rewards.append(reward)
            self.prices.append(self.env.current_price)
            
            if action in (1, 2):
                self.sim.trigger.loc[self.env.current_date]

            # Break if done
            if done: break



    def render(self):
        
        if self.astats.empty:
            print(f'Hold triggers: {self.actions.count(0)} ({round( self.actions.count(0)/len(self.actions) ,3)})')
            print(f'Buy triggers: {self.actions.count(1)} ({round( self.actions.count(1)/len(self.actions) ,3)})')
            print(f'Sell triggers: {self.actions.count(2)} ({round( self.actions.count(2)/len(self.actions) ,3)})')
            self.env.render()
        else:
            self.env.render(stats=self.astats)
            self.astats.loc['holdTrigger'] = round( self.actions.count(0)/len(self.actions) ,3)
            self.astats.loc['buyTrigger'] = round( self.actions.count(1)/len(self.actions) ,3)
            self.astats.loc['sellTrigger'] = round( self.actions.count(2)/len(self.actions) ,3)
            self.astats.loc['lastReward'] = round(self.rewards[-1])

        

    def _save_data(self):
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    num_steps = 300
    wavelet_scales = 100

    dc = DataCluster(
        dataset='realmix',
        remove_features=['close', 'high', 'low', 'open', 'volume'],
        wavelet_scales=wavelet_scales,
        num_time_steps=num_steps
        )
    collection = dc.collection
    (st_shape, lt_shape) = dc.get_model_shape()
    
    model_name = 'models/1618873083/1618873135_EPS1of800.model'
    ma = ModelAssessment(
        collection=collection,
        model_shape=(st_shape, lt_shape),
        num_time_steps=num_steps
        )
    ma.load_model(model_name=model_name)
    ma.sim_range = 300
    ma.simulate()

    print(ma.sim)
    print(ma.actions)
    print(ma.prices)

    df = pd.DataFrame(data={'price':ma.prices ,'action':ma.actions})
    df.plot(subplots=True)
    plt.show()

    print('=== EOL ===')