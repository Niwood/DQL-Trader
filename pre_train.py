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

import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from backtesting.test import GOOG
import pandas_ta as ta


# Environment settings
NUM_STOCKS = 1
WAVELET_SCALES = 100 #keep - number of frequecys used the wavelet transform

SAMPLE_SIZE = 10
PT_EPOCHS = 300

class Trainer:

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
        
        # Agent
        self.agent = Agent(
            model_shape=(st_shape, lt_shape),
            num_time_steps=self.num_time_steps
            )
        self.agent.pre_train(
            self.collection,
            cached_data=False,
            epochs=PT_EPOCHS,
            sample_size=SAMPLE_SIZE,
            lr_preTrain=1e-3
            )

        # Save the model
        self.agent.model.save(
            Path.cwd() / 'pre_trained_models' / str(int(time.time()))
            )
        


if __name__ == '__main__':
    Trainer()
    print('=== EOL ===')
