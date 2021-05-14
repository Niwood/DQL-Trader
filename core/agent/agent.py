from keras.models import Sequential
from keras.layers import LSTM, GRU, Input
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, SGD
from keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy, MeanAbsoluteError
from keras.layers.merge import concatenate
from keras.layers import Flatten, BatchNormalization, Concatenate
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D, MaxPooling3D
from keras.models import Model
from sklearn.metrics import confusion_matrix

from tensorflow.keras.metrics import AUC
import tensorflow.keras.backend as K
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
from core.tools import ModifiedTensorBoard
import random
from statsmodels.nonparametric.smoothers_lowess import lowess as low
from tqdm import tqdm
from statistics import mean
import pickle
import math
from pathlib import Path

from core.tools import safe_div, tic, toc

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)



class Agent:

    def __init__(
        self,
        model_shape=tuple(),
        num_time_steps=None
        ):

        # Constants
        self.DISCOUNT = 0.99
        self.REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
        self.MIN_REPLAY_MEMORY_SIZE = 0.3 * self.REPLAY_MEMORY_SIZE  # Minimum number of steps in a memory to start training
        self.MINIBATCH_SIZE = 16  # How many steps (samples) to use for training
        self.UPDATE_TARGET_EVERY = 10  # Terminal states (end of episodes)

        # Main model - gets trained every step
        self.st_shape = model_shape[0]
        self.lt_shape = model_shape[1]
        self.num_time_steps = num_time_steps
        self.model = self._create_model()

        # Save initial model weights
        self.initial_model_weights = np.array(self.model.get_weights()).ravel()
        
        # Target model this is what we .predict against every step
        self.target_model = self._create_model()
        self.target_model.set_weights(self.model.get_weights())

        # Parameters
        self.replay_memory_allocation = 0
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.replay_priority = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/log-{1}")
        self.target_update_counter = 0
        self.elapsed = 0
        self.conf_mat = np.array([])

        # MAE to calculate the error for prioritized replay priority
        self.mae = MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        self.action_errors = {0:list() ,1:list(), 2:list()}

        # Assertions
        # assert model_shape==tuple() , 'Missing model shape'
        # assert num_time_steps==None , 'Missing num time steps'


    def _create_model(self):
        
        ''' SHORT TERM HEAD '''
        st_head = Input(shape=self.st_shape)
        # st = Dropout(0.3)(st_head)

        st = GRU(8, return_sequences=True)(st_head)
        st = Dropout(0.2)(st)

        st = GRU(8, return_sequences=False)(st)
        st = Dropout(0.2)(st)

        # st = GRU(4, return_sequences=False)(st)
        # st = Dropout(0.2)(st)

        st = Dense(32)(st)

        ''' LONG TERM HEAD '''
        lt_head = Input(shape=self.lt_shape)

        lt = Conv2D(filters=4, kernel_size=2, padding='valid', activation='relu')(lt_head)
        lt = MaxPooling2D(pool_size=4, strides=(1,1), padding='valid')(lt)
        lt = BatchNormalization()(lt)

        # lt = Conv2D(filters=4, kernel_size=2, padding='valid', activation='relu')(lt)
        # lt = MaxPooling2D(pool_size=2, padding='valid')(lt)
        # lt = BatchNormalization()(lt)
        
        # lt = Conv2D(filters=4, kernel_size=4, padding='valid', activation='relu')(lt)
        # lt = MaxPooling2D(pool_size=2, padding='valid')(lt)
        # lt = BatchNormalization()(lt)

        lt = Flatten(name='ltflatten')(lt)
        lt = Dense(128)(lt)
        lt = Dropout(0.2)(lt)

        # lt = Dense(256)(lt)
        # lt = Dropout(0.2)(lt)

        lt = Dense(64)(lt)
        lt = Dropout(0.2)(lt)

        ''' MERGED TAIL '''
        tail = Concatenate()([st, lt])
        
        tail = Dense(32)(tail)
        tail = Dropout(0.2)(tail)

        # tail = Dense(8)(tail)
        # tail = Dropout(0.2)(tail)

        ''' MULTI OUTPUTS '''
        action_prediction = Dense(3, activation='softmax')(tail)
        
        # Compile model
        model = Model(inputs=[st_head, lt_head], outputs=action_prediction)

        opt = Adam(learning_rate=1e-10)

        # Cost of missclassification
        self.cost_matrix = np.ones((3,3))
        minor_cost = 1.5
        self.cost_matrix[0, 1] = minor_cost
        self.cost_matrix[0, 2] = minor_cost
        self.cost_matrix[1, 0] = minor_cost
        self.cost_matrix[2, 0] = minor_cost

        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['Precision', 'Recall', AUC(curve='PR')]
        )
        
        return model


    def compare_initial_weights(self):
        model_weights = np.array(self.model.get_weights()).ravel()
        b = list()
        for _we1, _we2 in zip(self.initial_model_weights, model_weights):
            a = np.square(np.subtract(_we1,_we2)).mean()
            if a>0: b.append(a) #zero if no bias

        return mean(b)


    def load_network(self, name):
        # Load model
        path = Path.cwd() / 'pre_trained_models' / str(name)
        self.model = tf.keras.models.load_model(path)


    def pre_train(self, collection, cached_data=False, epochs=500, sample_size=500, train_ratio=0.8, lr_preTrain=1e-3):
        from core import StockTradingEnv

        # Save default lr and sub the new one for pre-training
        _lr = K.eval(self.model.optimizer.lr)
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     lr_preTrain,
        #     decay_steps=100000,
        #     decay_rate=0.99)
        self.model.optimizer.lr = lr_preTrain

        # Init env for pre-train
        _env = StockTradingEnv(collection, look_back_window=self.num_time_steps, generate_est_targets=True)

        if not cached_data:
            # Sample from env for balanced a target set
            batch_loader_train = {'lt':list(), 'st':list(), 'target':list()}
            batch_loader_test = {'lt':list(), 'st':list(), 'target':list()}
            sample_size = int(sample_size/3)
            train_size = int(sample_size * train_ratio)
            

            for requested_target in [1,2,0]: #3 for each action
                for k in tqdm(range(sample_size), desc=f'Generating pre-training samples with target {requested_target}'):
                    _env.requested_target = requested_target #Specify the requested action for in which the env will find a dataset for

                    try:
                        state, target = _env.reset()
                    except:
                        continue
                    
                    # _env.df_target.plot(subplots=True)
                    # plt.show()
                    # quit()
                    
                    if (state['st'].shape, state['lt'].shape) != (self.st_shape, self.lt_shape):
                        print((state['st'].shape[0], state['lt'].shape[1]),'|',(self.num_time_steps, self.num_time_steps))
                        continue #This happens when there is not enough data left of the dataframe at the sampled action

                    
                    if k < train_size:
                        loader = batch_loader_train
                    else:
                        loader = batch_loader_test

                    loader['st'].append(state['st'])
                    loader['lt'].append(state['lt'])
                    loader['target'].append(target)

            # Save batches
            with open('data/pre_train_data/batch_loader_test.pkl', 'wb') as handle:
                pickle.dump(batch_loader_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('data/pre_train_data/batch_loader_train.pkl', 'wb') as handle:
                pickle.dump(batch_loader_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif cached_data:
            print('Loading cached data for pre-training..')
            with open('data/pre_train_data/batch_loader_test.pkl', 'rb') as handle:
                batch_loader_test = pickle.load(handle)
            with open('data/pre_train_data/batch_loader_train.pkl', 'rb') as handle:
                batch_loader_train = pickle.load(handle)

        a = [np.argmax(i) for i in batch_loader_train['target']]
        print('TRAIN HOLD',a.count(0))
        print('TRAIN BUY',a.count(1))
        print('TRAIN SELL',a.count(2))
        a = [np.argmax(i) for i in batch_loader_test['target']]
        print('TEST HOLD',a.count(0))
        print('TEST BUY',a.count(1))
        print('TEST SELL',a.count(2))


        ### Train
        st_train = np.array(batch_loader_train['st'])
        lt_train = np.array(batch_loader_train['lt'])
        y_train = np.array(batch_loader_train['target'])

        self.model.fit(
            [st_train, lt_train], y_train,
            batch_size=16,
            shuffle=True,
            epochs=epochs
            )
        
        # Evaluation
        st_test = np.array(batch_loader_test['st'])
        lt_test = np.array(batch_loader_test['lt'])
        y_test = np.argmax(np.array(batch_loader_test['target']), axis=1)
        y_hat = np.argmax(self.model.predict([st_test, lt_test]), axis=1)

        # Confusion matrix
        self.conf_mat = confusion_matrix(y_test,y_hat)
        print(self.conf_mat)

        # Switch back to default lr
        self.model.optimizer.lr = _lr

        # Update target model weights
        self.target_model.set_weights(self.model.get_weights())

        # Clean memory
        del batch_loader_test, batch_loader_train, _env, st_test, lt_test, y_test, y_hat, st_train, lt_train, y_train


    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)
        self.replay_priority.append(max(self.replay_priority, default=1))
        self.replay_memory_allocation = len(self.replay_memory) / self.REPLAY_MEMORY_SIZE

    def get_replay_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.replay_priority) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        # Importance scales down the update step during training also called
        # importance sampling weight that is dependent on the prob they were sampled with
        # The weight will “slow down” the learning of certain experience samples with respect to others
        importance = 1/len(self.replay_memory) * 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def set_priorities(self, indices, errors, offset=0.1):
        for i,e in zip(indices, errors):
            self.replay_priority[i] = abs(e) + offset

    def sample_prioritized_replay_memory(self, priority_scale=1.0):
        # UNIFORM (OLD) SAMPLING:
        # minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
        
        # Get the probabilities of the replay buffer
        sample_probabilities = self.get_replay_probabilities(priority_scale)

        # Sample replay buffer indicies based on sample probs
        # To increase performance: SumTree data structure
        sample_indices = random.choices(range(len(self.replay_memory)), k=self.MINIBATCH_SIZE, weights=sample_probabilities)

        # Get the corresponding experiences based on the sampled indicies
        minibatch = np.array(self.replay_memory)[sample_indices]

        # Get importance weights
        importance = self.get_importance(sample_probabilities[sample_indices])

        # return map(list, zip(*minibatch)), importance, sample_indices
        return minibatch, importance, sample_indices



    def predict(self, states, model, minibatch=False):
        # Predict a mini batch OR a single state

        if minibatch:
            out = model.predict([np.array(states['st']), np.array(states['lt'])])
        elif not minibatch:
            st = np.array(states['st']).reshape((1, self.st_shape[0], self.st_shape[1]))
            lt = np.array(states['lt']).reshape((1, self.lt_shape[0], self.lt_shape[1], self.lt_shape[2]))
            out = model.predict([st, lt])

        try:
            self.elapsed = time.time() - self.t0
        except:
            self.elapsed = 0

        return out


    def train(self, terminal_state, step):
        ''' Trains main network every step during episode '''

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        # Timer
        self.t0 = time.time()

        # Get a minibatch of random samples from memory replay
        minibatch, importance, sample_indices = self.sample_prioritized_replay_memory()

        # Get current states from minibatch, then query NN model for Q values
        # current state will have the format (MINIBATCH_SIZE, timesteps, features)
        current_states = {
            'st':[transition[0]['st'] for transition in minibatch],
            'lt':[transition[0]['lt'] for transition in minibatch]}
        current_qs_list = self.predict(current_states, self.model, minibatch=True)
        
        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        # new_current_states = np.array([transition[3] for transition in minibatch])
        new_current_states = {
            'st':[transition[3]['st'] for transition in minibatch],
            'lt':[transition[3]['lt'] for transition in minibatch]}
        future_qs_list = self.predict(new_current_states, self.target_model, minibatch=True)

        # Calculate the errors
        errors = self.mae(future_qs_list, current_qs_list).numpy()

        # Enumerate the batches
        if step==1: #reset action errors only at first step
            self.action_errors = {0:list() ,1:list(), 2:list()}
        _X_st = list()
        _X_lt = list()
        _y = list()
        for index, (current_state, action, reward, _, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # Save the errors for each action
            self.action_errors[action].append(errors[index])

            # Append the training data
            _X_st.append(current_state['st'])
            _X_lt.append(current_state['lt'])
            _y.append(current_qs)


        # Train the model
        self.model.train_on_batch(
            [np.array(_X_st), np.array(_X_lt)],
            np.array(_y),
            sample_weight=importance)
        

        # Update the priorities
        self.set_priorities(sample_indices, errors)


        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0



    def get_qs(self, state):
        ''' Queries main network for Q values given 
        current observation space (environment state) '''
        return self.predict(state, self.model, minibatch=False)[0]






class WeightedCategoricalCrossentropy(CategoricalCrossentropy):

    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        cost_matrix: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        cost_matrix = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        model.compile(loss=WeightedCategoricalCrossentropy(cost_matrix), ...)
    """

    def __init__(self, cost_mat=None, name='WeightedCategoricalCrossentropy', **kwargs):
        assert cost_mat.ndim == 2
        assert cost_mat.shape[0] == cost_mat.shape[1]
        
        super().__init__(name=name, **kwargs)
        self.cost_mat = K.cast_to_floatx(cost_mat)
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        assert sample_weight is None, "should only be derived from the cost matrix"

        return super().__call__(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=get_sample_weights(y_true, y_pred, self.cost_mat),
        )



def get_sample_weights(y_true, y_pred, cost_m):
    num_classes = len(cost_m)
    y_true = tf.dtypes.cast(y_true, tf.float32)

    y_pred.shape.assert_has_rank(2)
    y_pred.shape[1:].assert_is_compatible_with(num_classes)
    y_pred.shape.assert_is_compatible_with(y_true.shape)

    y_pred = K.one_hot(K.argmax(y_pred), num_classes)


    y_true_nk1 = K.expand_dims(y_true, 2)
    y_pred_n1k = K.expand_dims(y_pred, 1)
    cost_m_1kk = K.expand_dims(cost_m, 0)

    sample_weights_nkk = cost_m_1kk * y_true_nk1 * y_pred_n1k
    sample_weights_n = K.sum(sample_weights_nkk, axis=[1, 2])

    return sample_weights_n




if __name__ == '__main__':
    from environment import StockTradingEnv
    from data_loader import DataCluster
    from evaluation import ModelAssessment
    from backtesting.test import GOOG
    from sklearn.preprocessing import MinMaxScaler
    import pandas_ta as ta

    wavelet_scales = 100
    num_steps = 300
    num_stocks = 10
    dc = DataCluster(
        dataset='realmix',
        remove_features=['close', 'high', 'low', 'open', 'volume'],
        num_stocks=num_stocks,
        wavelet_scales=wavelet_scales,
        num_time_steps=num_steps
        )
    collection = dc.collection
    (st_shape, lt_shape) = dc.get_model_shape()

    env = StockTradingEnv(collection, look_back_window=num_steps)
    
    agent = Agent(
        model_shape=(st_shape, lt_shape),
        num_time_steps=num_steps)

    # ma = ModelAssessment(
    #     collection=collection,
    #     model_shape=(st_shape, lt_shape),
    #     num_time_steps=num_steps
    #     )
    # ma.model = agent.model

    # for i in range(10):
    #     print('RENDER:', i)
    #     ma.simulate()
    #     ma.render()
    #     print('-'*10)
    # quit()

    current_step = env.reset()
    state, reward, done = env.step(0)
    agent.update_replay_memory((state, 0, reward, state, done))
    agent.train(done, 1)
    
    a = agent.get_qs(state)
    print(a)
    print(np.argmax(a))

    print('=== EOL ===')

    
