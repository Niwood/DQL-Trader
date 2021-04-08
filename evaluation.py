from tensorflow import keras
from data_loader import DataLoader
from environment import StockTradingEnv
import numpy as np
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
        dataframe=None,
        num_st_features=None,
        num_lt_features=None,
        num_time_steps=None,
        sim_range=300
        ):
        
        # Parameters
        self.num_st_features = num_st_features
        self.num_lt_features = num_lt_features
        self.num_time_steps = num_time_steps
        self.sim_range = sim_range
        self.astats = False
        self.model = None

        # Dataset
        self.df = dataframe

        # Env
        self.env = StockTradingEnv(
            self.df,
            look_back_window=num_time_steps,
            static_initial_step=0)
        self.date_index = self.env.date_index



    def load_model(self, model_name=None):
        # Load model
        self.model_name = model_name
        loss_function = WeightedCategoricalCrossentropy()
        self.model = keras.models.load_model(
            self.model_name,
            compile=True,
            custom_objects={loss_function.name: loss_function}
            )



    def _baseline_model(self):
        ''' Returns a basleine model with randomly permuteed weights  '''
        model = keras.models.load_model(self.model_name)
        weights = model.get_weights()
        weights = [np.random.permutation(weight.flat).reshape(weight.shape) for weight in weights]
        model.set_weights(weights)
        return model



    def simulate(self):

        # Load baseline models
        # bl1 = self._baseline_model()
        # bl2 = _baseline_model()
        # bl3 = _baseline_model()

        # Reset env
        obs, _ = self.env.reset()

        # Random model
        # self.model = self._create_model()

        # Parameters to save
        self.actions = list()
        self.sell_buy_actions = list()
        self.sell_buy_date = list()
        self.rewards = list()
        

        for _ in tqdm(range(self.sim_range)):
            _obs_st = obs['st'].reshape((1, self.num_time_steps, self.num_st_features))
            _obs_lt = obs['lt'].reshape((1, self.num_time_steps, self.num_lt_features))
            action = self.model.predict([_obs_st, _obs_lt])

            if np.isnan(np.sum(action)):
                print('Action contains nan [in evaluation]: ',action), quit()

            # Step env
            obs, reward, done = self.env.step(np.argmax(action))

            # Save parameters
            action = np.argmax(action)
            self.actions.append(action)
            self.rewards.append(reward)

            if action in (1, 2):
                self.sell_buy_actions.append(action)
                self.sell_buy_date.append(self.env.current_date)

            # Break if done
            if done: break



    def render(self):

        if self.astats is False:
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
    dl = DataLoader(dataframe='google', remove_features=['close', 'high', 'low', 'open', 'volume'])
    df = dl.df
    
    model_name = 'models/1616954622/1616954841_EPS20of500.model'
    ma = ModelAssessment(
        model_name=model_name,
        dataframe=df,
        num_st_features=dl.num_st_features,
        num_lt_features=dl.num_lt_features,
        num_time_steps=180
        )
    ma.sim_range = 100
    ma.simulate()

    print(ma.actions.count(0)/len(ma.actions))
    print(ma.actions.count(1)/len(ma.actions))
    print(ma.actions.count(2)/len(ma.actions))

    print('=== EOL ===')