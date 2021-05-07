import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess as low
import pywt
import glob
import pickle

from backtesting.test import GOOG
from math import copysign
from core.tools import safe_div
pd.options.mode.chained_assignment = None


INITIAL_ACCOUNT_BALANCE = 100_000


class StockTradingEnv2(gym.Env):
    """ A stock trading environment for OpenAI gym
        StockTradingEnv2 uses pre-processed staged data
        This is only used for the reinforcement learning stage
        Generate the staged data via stage_data.py
    """

    metadata = {'render.modes': ['human']}
    ACTION_SPACE_SIZE = 3 # Buy, Sell or Hold


    def __init__(self):
        super(StockTradingEnv2, self).__init__()

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

        # Parameters
        self.load_batch = True #To load the first batch



    def _next_observation(self):

        # Iterate over staged obs
        try:
            obs = self.staged_obs[self.iteration_step]
        except:
            print('ENDED IN NEXT OBS')
            quit()
        target = None

        return obs, target



    def _take_action(self, action):
        # Set the current price to a random price within the time step
        # current_price = random.uniform(self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "close"])
        action_type = action

        try:
            self.current_price = self.df_reward.close[self.current_step]
        except:
            print('ISSUE IN _take_action')
            print(self.df_reward.close)
            print(self.current_step)
            quit()
        # print(self.current_step, self.start_step)

        comission = 0.02 # The comission is applied to both buy and sell
        amount = 0.3

        if action_type == 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / self.current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * self.current_price * (1 + comission)

            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought
            self.shares_bought_total += additional_cost
            self.buy_triggers += 1

        elif action_type == 2:
            # Sell amount % of shares held
            self.shares_sold = int(self.shares_held * amount)
            self.balance += self.shares_sold * self.current_price * (1 - comission)
            self.shares_held -= self.shares_sold
            self.total_shares_sold += self.shares_sold
            self.total_sales_value += self.shares_sold * self.current_price
            self.sell_triggers += 1

        # Save amount
        self.amounts.append(amount)

        # Update the net worth
        self.net_worth = self.balance + self.shares_held * self.current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0



    def step(self, action):

        # Execute one time step within the environment
        self._take_action(action)
        self.current_date = self.df_reward.date[self.current_step]


        # Check if there are no more steps in the data or we have met the maximum amount of steps
        if self.current_step == self.start_step + self.max_steps:
            delay_modifier = 1
            done = True
        else:
            delay_modifier = (self.current_step - self.start_step) / self.max_steps
            done = False

        # Calculate the reward
        '''
        Gain reward:
            + The asset goes up and we have invested
            + The asset goes down and we have not invested
        Loose reward:
            - The asset goes up and we have not invested
            - The asset goes down and we have invested
        '''
        
        _teomax_net_worth = self.df_reward.teomax.loc[self.current_step] * INITIAL_ACCOUNT_BALANCE

        reward = self.net_worth / _teomax_net_worth

        if action == 2: # sell
            # reward += -1 * copysign(current_lowess_grad2_scaled, current_lowess_grad2) # Down/Up shift 
            # reward -= abs(current_lowess_grad_scaled) # Less reward if not triggered on max/min points
            reward *= 1 - self.sell_triggers / self.max_steps # More triggers causes less reward
            # pass
            
        elif action == 1: # buy
            # reward += copysign(current_lowess_grad2_scaled, current_lowess_grad2) # Down/Up shift
            # reward -= abs(current_lowess_grad_scaled) # Less reward if not triggered on max/min points
            reward *= 1 - self.buy_triggers / self.max_steps  # More triggers causes less reward
            # pass

        elif action == 0: # hold
            # reward += ( 1 - INITIAL_ACCOUNT_BALANCE / self.net_worth) * 50 # 14 day trailing  
            # reward -= (current_lowess_grad_scaled - 1) * (1 - current_holding) * 50 # Less reward when asset decreses in value
            # reward = 0
            pass

        reward *= delay_modifier

        # Update next observation
        obs, _ = self._next_observation()

        # Done if net worth is negative
        if not done:
            done = self.net_worth <= 0

        # Iterate step
        self.iteration_step += 1
        self.current_step += 1

        return obs, reward, done



    def _load_staged_batch(self):
        '''
        Staged obs has the format:

        np.array(
            np.array(
                DataFrame('st': ..., 'lt':...),
                DataFrame('st': ..., 'lt':...),
                DataFrame('st': ..., 'lt':...), ...
            ),
            np.array(
                DataFrame('st': ..., 'lt':...),
                DataFrame('st': ..., 'lt':...),
                DataFrame('st': ..., 'lt':...), ...
            ), ...
        )
        '''

        # Sample a staged batch
        random_staged_batch_path = random.choice(
            glob.glob('data/staged/staged_batch_*.pkl')
            )

        # Load the batch
        with open(random_staged_batch_path, 'rb') as handle:
            staged = pickle.load(handle)

        # Seperate obs and df_reward
        self.df_reward_batch = staged['df']
        self.staged_obs_batch = staged['obs']

        # Parameters
        self.len_this_batch = len(self.df_reward_batch)
        self.batch_numbering_list = list(range(self.len_this_batch))
        random.shuffle(self.batch_numbering_list)
        self.max_steps = len(self.staged_obs_batch[0]) - 1



    def _next_in_batch(self):

        # Load batch if not loaded        
        if self.load_batch:
            self._load_staged_batch()
            self.load_batch = False
            self.iteration = 0 #Used to iterate obs in 

        # If on the last iteration in batch, load a new batch next time
        if self.iteration == self.len_this_batch-1:
            self.load_batch = True

        # Extract df_reward and obs next iteration
        random_iteration = self.batch_numbering_list[self.iteration]
        self.df_reward = self.df_reward_batch[random_iteration]
        self.staged_obs = self.staged_obs_batch[random_iteration]

        # Step iteration
        self.iteration += 1



    def reset(self):

        # Reset the state of the environment to an initial state
        self.ticker = None
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.shares_bought_total = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.buy_triggers = 0
        self.sell_triggers = 0
        self.amounts = list()

        # Set initial values
        self._next_in_batch()
        self.current_step = self.df_reward.index[0]
        self.start_step = self.current_step
        self.iteration_step = 0 #Counts steps from 0
        self.initial_price = self.df_reward.close[self.start_step]

        return self._next_observation()



    def render(self, mode='human', close=False, stats=False):
        ''' Render the environment to the screen '''


        _stats = {
            'ticker': self.ticker,
            'amountBalance': round(self.balance),
            'amountAsset': round(self.total_sales_value),
            'netWorth': self.net_worth,
            'netWorthChng': round( self.net_worth / INITIAL_ACCOUNT_BALANCE , 3),
            'profit': round(self.net_worth - INITIAL_ACCOUNT_BALANCE),
            'buyAndHold': round(  self.df_reward.close[self.current_step - 1] / self.initial_price , 3)
            }

        if stats is False:
            for statName, stat in _stats.items():
                print(f'{statName}: {stat}')
            return

        for col in stats.index:
            try: stats.loc[col] = _stats[col]
            except: pass




if __name__ == '__main__':
    pass