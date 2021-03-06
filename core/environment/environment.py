import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess as low
import pywt

from math import copysign
from core.tools import safe_div
pd.options.mode.chained_assignment = None


INITIAL_ACCOUNT_BALANCE = 100_000


class StockTradingEnv(gym.Env):
    """ A stock trading environment for OpenAI gym
        StockTradingEnv uses the data cluster for pre-processing
        This is only used to stage data and in the supervised learning stage
    """

    metadata = {'render.modes': ['human']}
    ACTION_SPACE_SIZE = 3 # Buy, Sell or Hold


    def __init__(
        self,
        collection,
        look_back_window,
        max_steps=300,
        static_initial_step=0,
        generate_est_targets=False):

        super(StockTradingEnv, self).__init__()

        # Constants
        self.LOOK_BACK_WINDOW = look_back_window
        self.max_steps = max_steps
        self.static_initial_step = static_initial_step
        self.generate_est_targets = generate_est_targets
        self.requested_target = 0

        # Set number range as df index and save date index
        self.collection = collection

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)



    def _next_observation(self):
        # Define the span for the observation
        span = (
            self.current_step - (self.LOOK_BACK_WINDOW-1), 
            self.current_step
        )
        
        # Re-slice the frame for requested target in pre-training
        if self.generate_est_targets:
            # Request the df slice from dp
            _df = self.dp.get_slice(span)

            # Re-calculate the slice based on the requested target
            span, target = self._specific_slice(_df, requested_target=self.requested_target)

        else:
            # Set target to none since we don't pre-train
            target = None

        # Request datapac to process the data in span
        df_st, df_lt = self.dp.data_process(span)

        return {'st':df_st, 'lt':df_lt}, target



    def _specific_slice(self, _df, requested_target=0):
        '''
        Slice df on a specific target that is calculated from
        a Locally Weighted Scatterplot Smoothing on the close column to determine local max/min
        Outputs a span and target
        '''

        # Lowess and lowess grad
        _df['lowess'] = low(_df.close, _df.index, frac=0.05)[:, 1]
        _df['lowess_grad'] = low(np.gradient(_df.lowess), _df.index, frac=0.01)[:, 1]
        _df_lowess_2grad = low(np.gradient(_df.lowess_grad), _df.index, frac=0.01)[:, 1]
        _df_lowess_grad = _df['lowess_grad'].copy()
        self.df_target = _df.copy()

        # Detect sign change
        _df.lowess_grad = np.sign(_df.lowess_grad)
        _df.lowess_grad = (_df.lowess_grad.shift(periods=1) - _df.lowess_grad)/2
        _df.lowess_grad.fillna(0, inplace=True)
        
        # Set detected peaks/valleys to 0 if they are outside a quantile
        qnt = 0.05
        _df.lowess_grad = _df.lowess_grad * ((_df_lowess_2grad > np.quantile(_df_lowess_2grad, 1-qnt)) | (_df_lowess_2grad < np.quantile(_df_lowess_2grad, qnt)))
    
        # Build target stack: [ [1,0,0], [1,0,0], [0,1,0], ... ]
        target = np.dstack((
            ((_df.lowess_grad == 0) * 1).to_numpy(),
            ((_df.lowess_grad < 0) * 1).to_numpy(),
            ((_df.lowess_grad > 0) * 1).to_numpy()
            ))[0]

        # Get index of requested target
        _target = np.array([np.argmax(i) for i in target])
        self.df_target['_df_lowess_2grad'] = _df_lowess_2grad
        self.df_target['target'] = _target
        self.df_target = self.df_target[['close', 'target', 'lowess', 'lowess_grad']]

        try:
            # selected_target_idx = np.random.choice(np.where(_target == requested_target)[0]) #Take a random choice
            a = self.df_target[(self.df_target.target == requested_target) & (self.df_target.index>self.LOOK_BACK_WINDOW+49)]
            selected_target_idx = np.random.choice(a.index)
        except:
            _old_requested_target = requested_target
            requested_target = 1 if requested_target==2 else 2
            try:
                a = self.df_target[(self.df_target.target == requested_target) & (self.df_target.index>self.LOOK_BACK_WINDOW+49)]
                selected_target_idx = np.random.choice(a.index)
                # print(f'--- Could not find target: {_old_requested_target} - switched to {requested_target} instead')
            except:
                requested_target = 0
                a = self.df_target[(self.df_target.target == requested_target) & (self.df_target.index>self.LOOK_BACK_WINDOW+49)]
                selected_target_idx = np.random.choice(a.index)
                # print(f'--- Could not find target: 1 or 2 - switched to {requested_target} instead')

        # Request datapac to process the data in span
        span = (
            selected_target_idx - (self.LOOK_BACK_WINDOW-1), selected_target_idx
        )

        # Prep output vector for target 
        target_out = np.array([0,0,0])
        target_out[requested_target] = 1

        return span, target_out



    def _take_action(self, action):
        # Set the current price to a random price within the time step
        # current_price = random.uniform(self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "close"])
        action_type = action

        self.current_price = self.df.loc[self.current_step, "close"]
        comission = 0.01 # The comission is applied to both buy and sell
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
        self.current_date = self.date_index[self.current_step]

        # Take next step
        self.current_step += 1
        done = False
        
        # Check if there are no more steps in the data or we have met the maximum amount of steps
        if self.current_step == self.start_step+self.max_steps or self.current_step == len(self.df.loc[:, 'open'].values)-1:
            delay_modifier = 1
            done = True
        else:
            delay_modifier = (self.current_step - self.start_step) / self.max_steps


        # Calculate the reward
        '''
        Gain reward:
            + The asset goes up and we have invested
            + The asset goes down and we have not invested
        Loose reward:
            - The asset goes up and we have not invested
            - The asset goes down and we have invested
        '''

        # _teomax_net_worth = self.df_reward.teomax.loc[self.current_step] * INITIAL_ACCOUNT_BALANCE
        # reward = self.net_worth / _teomax_net_worth

        if action == 2: # sell
            # reward += -1 * copysign(current_lowess_grad2_scaled, current_lowess_grad2) # Down/Up shift 
            # reward -= abs(current_lowess_grad_scaled) # Less reward if not triggered on max/min points
            # reward *= 1 - self.sell_triggers / self.max_steps # More triggers causes less reward
            pass
            
        elif action == 1: # buy
            # reward += copysign(current_lowess_grad2_scaled, current_lowess_grad2) # Down/Up shift
            # reward -= abs(current_lowess_grad_scaled) # Less reward if not triggered on max/min points
            # reward *= 1 - self.buy_triggers / self.max_steps  # More triggers causes less reward
            pass

        elif action == 0: # hold
            # reward += ( 1 - INITIAL_ACCOUNT_BALANCE / self.net_worth) * 50 # 14 day trailing  
            # reward -= (current_lowess_grad_scaled - 1) * (1 - current_holding) * 50 # Less reward when asset decreses in value
            # reward = 0
            pass

        reward = self.net_worth
        reward *= delay_modifier


        # Update next observation
        obs, _ = self._next_observation()

        # Done if net worth is negative
        if not done:
            done = self.net_worth <= 0

        return obs, reward, done


    def _gen_initial_step(self):
        
        # To generate initial step
        if self.static_initial_step == 0:
            self.current_step = random.randint(
                self.LOOK_BACK_WINDOW + 50, len(self.df.loc[:, 'open'].values)-self.max_steps
                ) #add 50 due to that datapack needs to calc the features and remove nan rows
        else:
            self.current_step = self.static_initial_step + self.LOOK_BACK_WINDOW + 2



    def reset(self):

        # Sample a data pack from the cluster and setup df
        self.dp = np.random.choice(self.collection, replace=True)
        self.df = self.dp.df
        self.date_index = self.dp.date_index

        # Reset the state of the environment to an initial state
        self.ticker = self.dp.ticker
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
        self._gen_initial_step()
        self.start_step = self.current_step + 1
        self.initial_price = self.df.loc[self.start_step, "close"]
        self.initial_date = self.date_index[self.start_step]

        # Copy df to use for reward algorithm
        self.df_reward = self.df.loc[self.start_step:self.start_step+self.max_steps].copy()
        self.df_reward['date'] = self.date_index[self.start_step-1:self.start_step+self.max_steps]
        self.df_reward['shifted'] = self.df_reward.close.shift()
        self.df_reward['div'] = self.df_reward.close / self.df_reward.shifted
        self.df_reward['pos_shift'] = self.df_reward['div'] > 1
        self.df_reward['pos_return'] = self.df_reward['div'][self.df_reward['pos_shift']]
        self.df_reward['teomax'] = self.df_reward['pos_return'].fillna(1).cumprod()
        

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
            'buyAndHold': round(  self.df.close[self.current_step] / self.initial_price , 3),
            'fromToDays': (self.date_index[self.current_step] - self.initial_date).days
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