import sys

from core import Agent, DataCluster, ModelAssessment, StockTradingEnv
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
import gc
import pickle

import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from backtesting.test import GOOG
import pandas_ta as ta


NUM_STOCKS = 0
WAVELET_SCALES = 100 #keep - number of frequecys used the wavelet transform
MAX_STEPS = 90 #Max steps taken by the env until the episode ends



class Stager:

    def __init__(self):

        self.num_time_steps = 300 #keep - number of sequences that will be fed into the model

        # Data cluster
        self.dataset = 'realmix'
        self.data_cluster = DataCluster(
            dataset=self.dataset,
            remove_features=['close', 'high', 'low', 'open', 'volume'],
            num_stocks=NUM_STOCKS,
            wavelet_scales=WAVELET_SCALES,
            num_time_steps=self.num_time_steps
            )
        self.collection = self.data_cluster.collection
        (st_shape, lt_shape) = self.data_cluster.get_model_shape()
        
        # Env
        self.env = StockTradingEnv(
            self.collection,
            look_back_window=self.num_time_steps,
            max_steps=MAX_STEPS,
            static_initial_step=0
            )

        # Run
        self.run()


    def run(self):

        batches = 25
        iterations_per_batch = 45
        
        for batch in range(batches):

            batch_dict = {'df':list(), 'obs':list()}

            for iteration in tqdm(range(iterations_per_batch), desc=f'Batch {batch}'):

                # List to save all obs for several iterations
                iteration_list = list()
                break_flag = False

                try:
                    obs,_ = self.env.reset()
                except:
                    print('CONTINUED ON RESET')
                    continue

                # Append obs from reset
                iteration_list.append(obs)

                for step in range(MAX_STEPS):
                    
                    # Step with arbitrary action
                    try:
                        obs,_,_ = self.env.step(0)
                    except:
                        print('CONTINUED DURING ITERATION')
                        break_flag = True
                        break

                    # Append obs from step
                    iteration_list.append(obs)
                
                # Append to batch_dict
                batch_dict['df'].append(self.env.df_reward)
                batch_dict['obs'].append(iteration_list)

            if break_flag: continue


            # Pickle batch
            with open(Path.cwd() / 'data' / 'staged' / f'staged_batch_{batch}.pkl', 'wb') as handle:
                pickle.dump(batch_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # print(f'Batch {batch} finished with {len(iteration_list)} iterations')

            # Free memory
            del iteration_list, batch_dict
            gc.collect()



if __name__ == '__main__':
    Stager()
    print('=== EOL ===')