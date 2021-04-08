from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting.test import GOOG
import matplotlib.pyplot as plt
import random

class DataLoader:

    def __init__(self, dataframe=None, remove_features=False):

        # Parameters
        self.remove_features = remove_features

        # Load data
        if dataframe == 'google':
            self.df = GOOG
            self.df_name = 'google'
        elif dataframe == 'sine':
            self.df = GOOG
            self.df_name = 'sine'
            sin_array = np.linspace(0, 20*np.pi, len(self.df))+(random.randint(2, 18)/10)
            price = np.sin(sin_array)+5
            self.df.Close = price
            self.df.Open = price * np.random.rand()
            self.df.High = price * 1.02
            self.df.Low = price * 0.996
            self.df.Volume *= self.df.Close

        # Run init methods
        self.pre_process()
        self.remove()
        self.count_features()

        # Add original values to df
        self.df = self.df.join(self.df_ORG)


    def pre_process(self):
        ''' Pre process data for technical indicators '''
        # Forward fill for missing dates
        self.df = self.df.asfreq(freq='1d', method='ffill')

        '''
        SHORT TERM
        '''

        # MACD
        self.df['MACD'] = self.df.ta.macd(fast=20, slow=40).MACDh_20_40_9
        
        # RSI signal
        rsi_high = 80
        rsi_low = 20
        rsi = self.df.ta.rsi()
        self.df['RSI_high'] = rsi * (rsi>rsi_high)
        self.df['RSI_low'] = rsi * (rsi<rsi_low)

        # TRIX signal
        self.df['TRIX'] = self.df.ta.trix(length=28).TRIXs_28_9

        # Bollinger band
        length = 30
        bband = self.df.ta.bbands(length=length)
        bband['hlc'] = self.df.ta.hlc3()

        # Bollinger band upper signal
        bbu_signal = bband['hlc'] - bband['BBM_'+str(length)+'_2.0']
        bband['BBU_signal'] = (abs(bbu_signal) * (bbu_signal > 0) / bband['BBU_'+str(length)+'_2.0'])

        # Bollinger band lower signal
        bbl_signal = bband['BBM_'+str(length)+'_2.0'] - bband['hlc']
        bband['BBL_signal'] = (abs(bbl_signal) * (bbl_signal > 0)  / bband['BBL_'+str(length)+'_2.0'])

        self.df['BBU_signal'] = bband.BBU_signal
        self.df['BBL_signal'] = bband.BBL_signal


        '''
        LONG TERM
        long term features requires to have "LT_" in the beginning of the name
        '''
        # SMA
        self.df['LT_SMA40'] = self.df.ta.sma(length=40)
        self.df['LT_SMA35'] = self.df.ta.sma(length=35)
        self.df['LT_SMA30'] = self.df.ta.sma(length=30)
        self.df['LT_SMA25'] = self.df.ta.sma(length=25)
        self.df['LT_SMA20'] = self.df.ta.sma(length=20)
        self.df['LT_SMA15'] = self.df.ta.sma(length=15)
        self.df['LT_SMA10'] = self.df.ta.sma(length=10)
        self.df['LT_SMA5'] = self.df.ta.sma(length=5)


        '''
        DROP NA AND SCALE
        '''
        # Drop NA rows
        self.df.dropna(inplace=True)

        # Copy df for original
        self.df_ORG = self.df[['close', 'high', 'low', 'open', 'volume']].copy()

        # Scale all values
        self.scaler = MinMaxScaler()
        self.df[self.df.columns] = self.scaler.fit_transform(self.df[self.df.columns])



    def count_features(self):
        ''' Return number opf features for long/short term df '''

        lt_feats = 0
        for feat in self.df.columns:
            if feat.split('_')[0] == 'LT':
                lt_feats += 1

        self.num_lt_features = lt_feats
        self.num_st_features = len(self.df.columns) - lt_feats

        

    def remove(self):
        ''' Remove certain columns '''
        if self.remove_features:
            self.df.drop(self.remove_features, axis=1, inplace=True)
        
        for col in ['close', 'high', 'low', 'open', 'volume']:
            try:
                self.df.rename(columns={col: "_"+col}, inplace=True)
            except:
                pass


if __name__ == '__main__':
    
    dl = DataLoader(dataframe='sine', remove_features=['high', 'low', 'open', 'volume'])
    df = dl.df
    # df.index = list(range(len(df)))
    # df = df.loc[100:400]

    print(df)

    # df.plot(subplots=True)
    # plt.show()