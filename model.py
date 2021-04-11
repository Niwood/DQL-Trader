# Neural network is the the Q-function, run it like this:
# model.predict(state.reshape(1,64), batch_size=1)

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
from tensorflow.keras.metrics import AUC
import tensorflow.keras.backend as K


import numpy as np
import time
from collections import deque
from tools import ModifiedTensorBoard
import random
from statsmodels.nonparametric.smoothers_lowess import lowess as low
from tqdm import tqdm

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)



class Agent:

    def __init__(
        self,
        num_st_features=None,
        num_lt_features=None,
        num_time_steps=None
        ):

        # Constants
        self.DISCOUNT = 0.99
        self.REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
        self.MIN_REPLAY_MEMORY_SIZE = 0.1 * self.REPLAY_MEMORY_SIZE  # Minimum number of steps in a memory to start training
        self.MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
        self.UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

        # Main model - gets trained every step
        self.num_st_features = num_st_features
        self.num_lt_features = num_lt_features
        self.num_time_steps = num_time_steps
        self.model = self._create_model()

        # Save initial model weights
        self.initial_model_weights = np.array([i.numpy().sum() for i in self.model.trainable_weights])

        # Target model this is what we .predict against every step
        self.target_model = self._create_model()
        self.target_model.set_weights(self.model.get_weights())

        # Parameters
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/log-{1}")
        self.target_update_counter = 0
        self.elapsed = 0


    def _create_model(self):

        ''' SHORT TERM HEAD '''
        st_head = Input(shape=(self.num_time_steps, self.num_st_features))
        # st = Dropout(0.3)(st_head)

        st = Conv1D(filters=12, kernel_size=2, padding='same', activation='relu')(st_head)
        st = MaxPooling1D(pool_size=6, padding='same')(st)
        # st = BatchNormalization()(st)

        st = Conv1D(filters=12, kernel_size=2, padding='same', activation='relu')(st)
        st = MaxPooling1D(pool_size=6, padding='same')(st)


        ''' LONG TERM HEAD '''
        lt_head = Input(shape=(self.num_time_steps, self.num_lt_features))
        # lt = Dropout(0.3)(lt_head)

        lt = Conv1D(filters=12, kernel_size=2, padding='same', activation='relu')(lt_head)
        lt = MaxPooling1D(pool_size=6, padding='same')(lt)
        # lt = BatchNormalization()(lt)

        lt = Conv1D(filters=12, kernel_size=2, padding='same', activation='relu')(lt)
        lt = MaxPooling1D(pool_size=6, padding='same')(lt)


        ''' MERGED TAIL '''
        tail = concatenate([st, lt])
        tail = Conv1D(filters=12, kernel_size=2, padding='same', activation='relu')(tail)
        tail = MaxPooling1D(pool_size=6, padding='same')(tail)

        tail = Flatten()(tail)

        tail = Dense(16)(tail)
        tail = Dropout(0.2)(tail)

        # tail = Dense(8)(tail)
        # tail = Dropout(0.2)(tail)

        ''' MULTI OUTPUTS '''
        action_prediction = Dense(3, activation='softmax')(tail)

        # Compile model
        model = Model(inputs=[st_head, lt_head], outputs=action_prediction)
        # initial_learning_rate = 0.1
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate,
        #     decay_steps=100000,
        #     decay_rate=0.96)
        # opt = SGD(learning_rate=0.1, decay=0.01, momentum=0.99)

        # opt = RMSprop(learning_rate=1e-3, momentum=0.95, epsilon=0.01, decay=0.96)
        opt = Adam(learning_rate=1e-7)

        # Cost of missclassification
        self.cost_matrix = np.ones((3,3))
        minor_cost = 1.5
        self.cost_matrix[0, 1] = minor_cost
        self.cost_matrix[0, 2] = minor_cost
        self.cost_matrix[1, 0] = minor_cost
        self.cost_matrix[2, 0] = minor_cost

        model.compile(
            loss=WeightedCategoricalCrossentropy(cost_mat=self.cost_matrix),
            optimizer=opt,
            metrics=['Precision', 'Recall', AUC(curve='PR')]
        )
        
        return model


    def compare_initial_weights(self):
        return self.initial_model_weights - np.array([i.numpy().sum() for i in self.model.trainable_weights])


    def pre_train(self, collection, epochs=500, num_batches=500, lr_preTrain=1e-5):
        from environment import StockTradingEnv

        # Save default lr and sub the new one for pre-training
        _lr = K.eval(self.model.optimizer.lr)
        self.model.optimizer.lr = lr_preTrain

        # Init env for pre-train
        _env = StockTradingEnv(collection, look_back_window=self.num_time_steps, generate_est_targets=True)

        # Sample from env for balanced a target set
        batch_loader = {'lt':list(), 'st':list(), 'target':list()}
        _num_sub_batches = int(num_batches/3)

        for requested_target in range(3): #3 for each action
            for _ in tqdm(range(_num_sub_batches), desc=f'Generating pre-training batches for action {requested_target}'):
                _env.requested_target = requested_target #Specify the requested action for in which the env will find a dataset for
                state, target = _env.reset()
                if (state['st'].shape[0], state['lt'].shape[0]) != (self.num_time_steps, self.num_time_steps):
                    continue #This happens when there is not enough data left of the dataframe at the sampled action
                batch_loader['st'].append(state['st'])
                batch_loader['lt'].append(state['lt'])
                batch_loader['target'].append(target)


        a = [np.argmax(i) for i in batch_loader['target']]
        print('HOLD',a.count(0))
        print('BUY',a.count(1))
        print('SELL',a.count(2))

        ### Train
        st = np.array(batch_loader['st'])
        lt = np.array(batch_loader['lt'])
        yhat = np.array(batch_loader['target'])
        self.model.fit(
            [st, lt], yhat,
            batch_size=16,
            shuffle=True,
            epochs=epochs
            )
        
        # Switch back to default lr
        self.model.optimizer.lr = _lr

        # Update target model weisghts
        self.target_model.set_weights(self.model.get_weights())


    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)


    def predict(self, states, model, minibatch=False):
        # Predict a mini batch OR a single state

        if minibatch:
            out = model.predict([np.array(states['st']), np.array(states['lt'])])
        elif not minibatch:
            st = np.array(states['st']).reshape((1, self.num_time_steps, self.num_st_features))
            lt = np.array(states['lt']).reshape((1, self.num_time_steps, self.num_lt_features))
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

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

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
        

        # Enumerate the batches
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

            # And append to our training data
            _X_st.append(current_state['st'])
            _X_lt.append(current_state['lt'])
            _y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit([np.array(_X_st), np.array(_X_lt)], np.array(_y), batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0



    def get_qs(self, state):
        ''' Queries main network for Q values given current observation space (environment state) '''
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

    dc = DataCluster(dataset='google', remove_features=['close', 'high', 'low', 'open', 'volume'])
    collection = dc.collection

    num_steps = 90
    env = StockTradingEnv(collection, look_back_window=num_steps)
    agent = Agent(
        num_st_features=dc.num_st_features,
        num_lt_features=dc.num_lt_features,
        num_time_steps=num_steps)
    
    agent.pre_train(collection, epochs=500, num_batches=3_000, lr_preTrain=1e-4)
    print(f' compare_initial_weights: {agent.compare_initial_weights()}')


    ma = ModelAssessment(
        collection=collection,
        num_st_features=dc.num_st_features,
        num_lt_features=dc.num_lt_features,
        num_time_steps=num_steps
        )
    ma.model = agent.model

    for i in range(10):
        print('RENDER:', i)
        ma.simulate()
        ma.render()
        print('-'*10)
    quit()

    current_step = env.reset()
    state, reward, done = env.step(0)
    agent.update_replay_memory((state, 0, reward, state, done))
    agent.train(done, 1)
    
    a = agent.get_qs(state)
    print(a)
    print(np.argmax(a))

    print('=== EOL ===')

    
