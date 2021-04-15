from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting.test import GOOG
import matplotlib.pyplot as plt
import random
from pathlib import Path
from tqdm import tqdm
import pywt





class DataCluster:
    '''
    Contains a collection of data packs
    '''

    def __init__(self, dataset=None, remove_features=False, num_stocks=1, wavelet_scales=100, num_time_steps=0 , verbose=True):

        self.collection = list()

        if dataset == 'google':
            self.collection.append(
                DataPack(dataframe=GOOG, remove_features=remove_features, wavelet_scales=wavelet_scales, num_time_steps=num_time_steps)
                )

        elif dataset == 'sine':
            df = GOOG
            sin_array = np.linspace(0, 200*np.pi, len(df))
            price = (np.sin(sin_array)*3 + 10) + np.random.normal(0,0.3,len(df))
            df.Close = price
            df.Open = price + np.random.normal(0,0.3,len(df))
            df.High = price * 1.02
            df.Low = price * 0.996
            df.Volume *= df.Close
            self.collection.append(
                DataPack(dataframe=df, remove_features=remove_features, wavelet_scales=wavelet_scales, num_time_steps=num_time_steps)
                )

        elif dataset == 'random':
            df = GOOG
            price = np.ones(len(df))
            for _ in range(5):
                sin_array = np.linspace(0, random.randint(50, 300)*np.pi, len(df))
                price += (np.sin(sin_array)*3) + np.random.normal(0,0.3,len(df))
            df.Close = price
            df.Open = price + np.random.normal(0,0.3,len(df))
            df.High = price * 1.02
            df.Low = price * 0.996
            df.Volume *= df.Close
            self.collection.append(
                DataPack(dataframe=df, remove_features=remove_features, wavelet_scales=wavelet_scales, num_time_steps=num_time_steps)
                )

        elif dataset == 'realmix':

            # Specify the dir
            data_folder = Path.cwd() / 'data'
            all_files = [x.stem for x in data_folder.glob('*/')]

            # Sample
            iterator = range(len(all_files)) if num_stocks==0 else range(num_stocks)
            random.shuffle(all_files)
            files_range = list(range(len(all_files)))
            for i in tqdm(
                iterator, desc=f'Generating data cluster'
                ) if verbose else iterator:

                skip = False
                while True:
                    try:
                        idx = i if num_stocks==0 else random.choice(files_range) #Sample an index
                        files_range.remove(idx) #Remove that index from the list
                    except:
                        skip = True
                        break
                    _file = all_files[idx] #Get the file

                    # Resample when the file is empty or unreadable
                    try:
                        df = pd.read_csv(f'data/{_file}.txt', delimiter = ",")
                    except: continue 

                    # Resample for small dataframes
                    if len(df) < 500:
                        continue 

                    # All ok -> break
                    break
                
                if skip: continue

                df.set_index('Date', inplace=True)
                df.drop(['OpenInt'], axis=1, inplace=True)
                df.index = pd.to_datetime(df.index)
                    
                self.collection.append(
                    DataPack(dataframe=df, ticker=_file, remove_features=remove_features, num_time_steps=num_time_steps, wavelet_scales=wavelet_scales)
                    )

        # Number of features
        self.num_lt_features = self.collection[0].num_lt_features
        # self.num_lt_features = 100 #Due to wavelet scales, see environment._make_wavelet
        self.num_st_features = self.collection[0].num_st_features



class DataPack:
    '''
    Contains the data for one time serie
    Performes a process of the data incl scaling and feature extraction
    '''

    def __init__(self, dataframe=None, ticker=None, remove_features=False, num_time_steps=0, wavelet_scales=0):

        # Parameters
        self.remove_features = remove_features
        self.ticker = ticker
        self.wavelet_scales = wavelet_scales
        self.num_time_steps = num_time_steps

        # Save original data
        # self.org = dataframe.Close.copy()

        # Load data
        self.df = dataframe

        # Run init methods
        self.pre_process()
        self.remove()
        self.count_features()

        # Add original values to df
        self.df = self.df.join(self.org)

        # Attributes
        self.date_index = self.df.index.copy()

        # Switch to numeric index
        self.df.index = list(range(len(self.df)))


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
        rsi = self.df.ta.rsi()
        self.df['RSI'] = rsi
        # rsi_high = 80
        # rsi_low = 20
        # self.df['RSI_high'] = rsi * (rsi>rsi_high)
        # self.df['RSI_low'] = rsi * (rsi<rsi_low)

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
        # self.df['LT_SMA40'] = self.df.ta.sma(length=40)
        # self.df['LT_SMA35'] = self.df.ta.sma(length=35)
        # self.df['LT_SMA30'] = self.df.ta.sma(length=30)
        # self.df['LT_SMA25'] = self.df.ta.sma(length=25)
        # self.df['LT_SMA20'] = self.df.ta.sma(length=20)
        # self.df['LT_SMA15'] = self.df.ta.sma(length=15)
        # self.df['LT_SMA10'] = self.df.ta.sma(length=10)
        # self.df['LT_SMA5'] = self.df.ta.sma(length=5)

        # Wavelets
        self.df['LT_close'] = self.df.close.copy()
        self.df['LT_RSI'] = self.df.RSI.copy()


        '''
        DROP NA AND SCALE
        '''
        # Drop NA rows
        self.df.dropna(inplace=True)

        # Copy df for original
        self.org = self.df[['close', 'high', 'low', 'open', 'volume']].copy()

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


    def make_wavelet(self, signals):
        # signal: (time_steps,)
        # coef: (scales, time_steps)
        scales = np.arange(1, self.wavelet_scales+1)
        out = np.zeros(shape=(len(scales), self.num_time_steps, len(signals.columns)))
        for idx, col in enumerate(signals.columns):
            signal = signals[col].diff().fillna(0).to_numpy()
            coef, _ = pywt.cwt(signal, scales, wavelet='gaus8') #gaus8 seems to give high res
            out[:, 0:coef.shape[1], idx] = abs(coef)
        return out #(scales, time_steps, num_LT_features)


    def remove(self):
        ''' Remove columns '''
        if self.remove_features:
            self.df.drop(self.remove_features, axis=1, inplace=True)
        
        for col in ['close', 'high', 'low', 'open', 'volume']:
            try:
                self.df.rename(columns={col: "_"+col}, inplace=True)
            except:
                pass





if __name__ == '__main__':
    
    data_cluster = DataCluster(
        dataset='realmix',
        remove_features=['close', 'high', 'low', 'open', 'volume'],
        num_stocks=1,
        num_time_steps=100
        )
    collection = data_cluster.collection

    # for dp in collection:
    #     print(f'{dp.ticker} has {len(dp.df)}')
        # df.index = list(range(len(df)))
        # steps = 90
        # start = random.randint(10,2000)
        # df = df.loc[start:start+steps]


    df = collection[0].df
    
    # df = df[0:300]

    # df.plot(subplots=True)
    # plt.show()