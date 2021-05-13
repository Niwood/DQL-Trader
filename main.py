import sys

from core import Agent, DataCluster, ModelAssessment, StockTradingEnv2
from core.tools import safe_div, tic, toc

from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from statistics import mean 
from collections import deque
from datetime import datetime
from pathlib import Path
import json
import logging
import random
import keras.backend as K


# Pre-trained network to load
PT_NETWORK = 1620378939

# Environment settings
EPISODES = int(5_000)
# MAX_STEPS = 90 #Max steps taken by the env until the episode ends
num_stocks = 100
WAVELET_SCALES = 100 #keep - number of frequecys used the wavelet transform

# Exploration settings
epsilon = 1
epsilon_plateau = 0.1
MIN_EPSILON = 1e-6

#  Stats settings
AGGREGATE_STATS_EVERY = 1  #keep - episodes
EPOCH_SIZE = 100


class Trader:

    def __init__(self):

        self.num_time_steps = 300 #keep - number of sequences that will be fed into the model

        # Data cluster
        self.dataset = 'realmix'
        self.data_cluster = DataCluster(
            dataset=self.dataset,
            remove_features=['close', 'high', 'low', 'open', 'volume'],
            num_stocks=num_stocks,
            wavelet_scales=WAVELET_SCALES,
            num_time_steps=self.num_time_steps
            )
        self.collection = self.data_cluster.collection
        (st_shape, lt_shape) = self.data_cluster.get_model_shape()
        
        # Agent
        self.agent = Agent(
            model_shape=(st_shape, lt_shape),
            num_time_steps=self.num_time_steps
            )
        self.agent.load_network(PT_NETWORK)

        # Environment
        self.env = StockTradingEnv2()

        # Epsilon
        self.epsilon_steps = \
            [1.0] * int(EPISODES * epsilon_plateau) + \
            list(np.linspace(1,MIN_EPSILON,EPISODES - int(EPISODES * epsilon_plateau)*2+1)) + \
            [MIN_EPSILON] * int(EPISODES * epsilon_plateau)

        # Statistics
        self.epsilon = epsilon
        stats = [
            'avgReward',
            'buyAndHold',
            'netWorthChng',
            'buyTrigger',
            'sellTrigger',
            'holdTrigger',
            'epsilon',
            'totTrainTime',
            'amountBalance',
            'amountAsset',
            'holdReward',
            'sellReward',
            'buyReward',
            'avgAmount'
            ]
        self.estats = pd.DataFrame(
            np.nan,
            index=np.arange(1, EPISODES+1),
            columns=stats)
        self.estats.index.name = 'Episode'

        astats_index = np.append( np.array([1]) , np.arange(0, EPISODES+1, EPOCH_SIZE)[1::] )
        self.astats = pd.DataFrame(
            np.nan,
            index=astats_index,
            columns=[
                'lastReward',
                'buyAndHold',
                'netWorthChng',
                'buyTrigger',
                'sellTrigger',
                'holdTrigger',
                'avgAmount',
                'networkName'
                ]
            )
        self.astats.index.name = 'Episode'
        
        # Create model folder and model ID
        self.model_id = int(time.time())
        self.agent.model._name = str(self.model_id)
        self.folder = Path.cwd() / 'models' / str(self.model_id)
        self.folder.mkdir(exist_ok=False)

        # Save architechture
        self.agent.model._name = str(self.model_id)
        with open(self.folder / 'arch.txt','w') as fh:
            self.agent.model.summary(print_fn=lambda x: fh.write(x + '\n'))

        # Save metadata
        metadata = {
            'model_id': self.model_id,
            'note': ' '.join([str(x) for x in sys.argv[1::]]),
            'date': str(datetime.now()),
            'episodes': EPISODES,
            'epoch_size': EPOCH_SIZE,
            'time_steps': self.num_time_steps,
            'data': {
                'dataset':self.dataset,
                'length': len(self.collection),
                },
            'optimizer': {
                'algorithm': K.eval(self.agent.model.optimizer._name),
                'learning_rate': float(K.eval(self.agent.model.optimizer.lr))
                },
            'agent': {
                'discount': self.agent.DISCOUNT,
                'replay_memory_size': self.agent.REPLAY_MEMORY_SIZE,
                'min_replay_memory_size': self.agent.MIN_REPLAY_MEMORY_SIZE,
                'minibatch_size': self.agent.MINIBATCH_SIZE,
                'update_target_network_every': self.agent.UPDATE_TARGET_EVERY
                },
            'pre_training': {
                'conf_mat': str(self.agent.conf_mat)
                }
            }
        with open(self.folder / 'metadata.json', 'w') as outfile:
            json.dump(metadata, outfile)
            

        # Run
        self.run()
    


    def run(self):

        # Iterate over episodes
        last_iteration_time = datetime.fromtimestamp(0)
        episode_iter = tqdm(range(1, EPISODES + 1), ascii=True, unit='episode')

        for episode in episode_iter:

            # Slice estats for this episode for simplicity
            self._estats = self.estats.loc[episode]

            # Timer to track train time
            self.train_time = 0

            # Save actions
            self.actions = list()
            
            # Update tensorboard step every episode
            self.agent.tensorboard.step = episode


            # Reset episode reward and step number
            self.episode_reward = list()
            self.reward_action = [0, 0, 0]
            step = 1

            # Reset environment and get initial state
            current_state, _ = self.env.reset()

            # Reset flag and start iterating until episode ends
            done = False
            
            
            while not done:

                # This part stays mostly the same, the change is to query a model for Q values
                if random.random() > self.epsilon:
                    # Get action from Q table
                    action = np.argmax(self.agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, self.env.ACTION_SPACE_SIZE)

                # STEP ENV
                new_state, reward, done = self.env.step(action)
                

                # Save reward and action
                self.reward_action[action] += reward
                self.actions.append(action)

                # Transform new continous state to new discrete state and count reward
                self.episode_reward.append(reward)

                # Every step we update replay memory and train main network
                self.agent.update_replay_memory((current_state, action, reward, new_state, done))
                self.agent.train(done, step)
                self.train_time += self.agent.elapsed
                
                current_state = new_state
                step += 1


            # Save model
            if not episode % EPOCH_SIZE or episode == 1:
                self._save_model(episode)
            else:
                # Set default values to evaluation stats
                self._estats.loc['TotTrainTime'] = round(self.train_time,1)

            # Render
            if not episode % AGGREGATE_STATS_EVERY:
                self._render(episode)

            # Decay epsilon
            self.epsilon = self.epsilon_steps[episode]


    def _render(self, episode):
        ''' Renders stats for a certain episodes '''
        print('='*20)

        # Env to render
        self.env.render(stats=self._estats)

        # Episode aggregated stats
        self._estats.loc['epsilon'] = round(self.epsilon,2)
        self._estats.loc['avgReward'] = round(mean(self.episode_reward),3)
        self._estats.loc['totTrainTime'] = round(self.train_time,1)
        self._estats.loc['holdTrigger'] = round( self.actions.count(0)/len(self.actions) ,3)
        self._estats.loc['buyTrigger'] = round( self.actions.count(1)/len(self.actions) ,3)
        self._estats.loc['sellTrigger'] = round( self.actions.count(2)/len(self.actions) ,3)
        self._estats.loc['holdReward'] = round( safe_div(self.reward_action[0], self.actions.count(0)) ,3)
        self._estats.loc['buyReward'] = round( safe_div(self.reward_action[1], self.actions.count(1)) ,3)
        self._estats.loc['sellReward'] = round( safe_div(self.reward_action[2], self.actions.count(2)) ,3)

        # Print episode stats
        self.estats.loc[episode] = self._estats
        print(self.estats.loc[episode-10:episode])

        # Pickle and save episode stats
        self.estats.loc[1:episode].to_pickle(self.folder / 'estats.pkl')



    def _save_model(self, episode):
        ''' For each epoch save model and make a sample inference with epsilon=0 '''
        epoch_id = int(time.time())
        self.last_model_name = f'{epoch_id}_EPS{episode}of{EPISODES}.model'
        self.agent.model.save(self.folder / self.last_model_name)
        try:
            self._model_assessment(episode)
        except:
            pass



    def _model_assessment(self, episode):
        ''' Make a model predict on a sample with epsilon=0 '''

        # Load data
        dc = DataCluster(
            dataset=self.dataset,
            remove_features=['close', 'high', 'low', 'open', 'volume'],
            num_stocks=1,
            verbose=False,
            wavelet_scales=WAVELET_SCALES,
            num_time_steps=self.num_time_steps
            )
        collection = dc.collection
        (st_shape, lt_shape) = dc.get_model_shape()
        
        # Model assessment
        ma = ModelAssessment(
            collection=collection,
            model_shape=(st_shape, lt_shape),
            num_time_steps = self.num_time_steps
            )
        ma.astats = self.astats.loc[episode]
        ma.load_model(model_name=self.folder / self.last_model_name)
        ma.simulate()
        ma.render()
        self.astats.loc[episode] = ma.astats

        # Save stats for the last model name
        self.astats.loc[episode, 'networkName'] = self.last_model_name

        # Print assessment stats
        print(self.astats.loc[episode-10:episode])

        # Pickle and save assessment stats and simulation run
        self.astats.loc[1:episode].to_pickle(self.folder / 'astats.pkl')
        ma.sim.to_pickle(self.folder / f'sim_{ma.ticker}_EPS{episode}of{EPISODES}.pkl')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    Trader()
    print('=== EOL ===')
