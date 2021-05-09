from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import random
from pathlib import Path
from tqdm import tqdm
import pywt
import cv2




class DataCluster:
    '''
    Contains a collection of data packs
    '''

    def __init__(
        self,
        dataset=None,
        remove_features=False,
        num_stocks=1,
        wavelet_scales=0,
        num_time_steps=0,
        verbose=True):

        self.num_time_steps = num_time_steps

        self.collection = list()

        if dataset == 'google':
            # self.collection.append(
            #     DataPack(dataframe=GOOG, remove_features=remove_features, wavelet_scales=wavelet_scales, num_time_steps=num_time_steps)
            #     )
            print('GOOG DATASET DEPRECATED -> QUIT')
            quit()

        elif dataset == 'sine':
            # df = GOOG
            # sin_array = np.linspace(0, 200*np.pi, len(df))
            # price = (np.sin(sin_array)*3 + 10) + np.random.normal(0,0.3,len(df))
            # df.Close = price
            # df.Open = price + np.random.normal(0,0.3,len(df))
            # df.High = price * 1.02
            # df.Low = price * 0.996
            # df.Volume *= df.Close
            # self.collection.append(
            #     DataPack(dataframe=df, remove_features=remove_features, wavelet_scales=wavelet_scales, num_time_steps=num_time_steps)
            #     )
            print('GOOG DATASET DEPRECATED -> QUIT')
            quit()

        elif dataset == 'random':
            # df = GOOG
            # price = np.ones(len(df))
            # for _ in range(5):
            #     sin_array = np.linspace(0, random.randint(50, 300)*np.pi, len(df))
            #     price += (np.sin(sin_array)*3) + np.random.normal(0,0.3,len(df))
            # df.Close = price
            # df.Open = price + np.random.normal(0,0.3,len(df))
            # df.High = price * 1.02
            # df.Low = price * 0.996
            # df.Volume *= df.Close
            # self.collection.append(
            #     DataPack(dataframe=df, remove_features=remove_features, wavelet_scales=wavelet_scales, num_time_steps=num_time_steps)
            #     )
            print('GOOG DATASET DEPRECATED -> QUIT')
            quit()

        elif dataset == 'realmix':

            # Specify the dir
            stock_folder = Path.cwd() / 'data' / 'stock'
            all_files = [x.stem for x in stock_folder.glob('*/')]

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
                        df = pd.read_csv(f'data/stock/{_file}.txt', delimiter = ",")
                    except: continue

                    # Resample for small dataframes
                    if len(df) < 600:
                        continue 

                    # All ok -> break
                    break
                
                if skip: continue

                # df.set_index('Date', inplace=True)
                # df.drop(['OpenInt'], axis=1, inplace=True)
                # df.index = pd.to_datetime(df.index)
                    
                self.collection.append(
                    DataPack(dataframe=df, ticker=_file, remove_features=remove_features, num_time_steps=num_time_steps, wavelet_scales=wavelet_scales)
                    )

        # Number of features
        # self.num_lt_features = self.collection[0].num_lt_features
        # self.num_st_features = self.collection[0].num_st_features



    def get_model_shape(self):
        ''' Returns the shape that will go into model '''

        # Sample a datapack
        dp = self.collection[0]

        # Arbitrary span for calculation
        span = (300,self.num_time_steps+299)
        
        # Request data process from datapack
        df_st, df_lt = dp.data_process(span)

        return df_st.shape, df_lt.shape





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
        self.st_scale_percent = 0.5 #fraction of original size
        self.lt_scale_percent = 0.5

        # Load data
        self.df = dataframe

        # Pre-process the data frame
        self.df_process()

        # Add original values to df
        self.org = self.df[['close', 'high', 'low', 'open', 'volume']].copy()

        # Save index as date
        self.date_index = self.df.index.copy()

        # Switch to numeric index
        self.df.index = list(range(len(self.df)))

        # Count features
        # self.count_features()


    def df_process(self):
        
        # Drop col
        self.df.drop(['OpenInt'], axis=1, inplace=True)

        # Set index to datetime
        self.df.set_index('Date', inplace=True)
        self.df.index = pd.to_datetime(self.df.index)
        
        # Rename columns
        self.df.columns= self.df.columns.str.lower()

        # Forward fill for missing dates
        self.df = self.df.asfreq(freq='1d', method='ffill')



    def data_process(self, span):
        '''
        Process data for feature extraction
        Output as numpy array
        '''

        # assert span[0]>0 , 'Negative span'

        # Slice the df according to the span
        # -50 due to nan values when calculating TIs
        _df = self.df.loc[span[0] - 50 : span[1]].copy()

        df_st = self.get_st_features(_df, span)
        df_lt = self.get_lt_features(_df, span)

        return df_st, df_lt



    def get_st_features(self, df, span):
        '''
        SHORT TERM
        '''

        # MACD
        df['MACD'] = df.ta.macd(fast=12, slow=26).MACDh_12_26_9
        
        # RSI signal
        rsi = df.ta.rsi()
        df['RSI'] = rsi
        # rsi_high = 80
        # rsi_low = 20
        # self.df['RSI_high'] = rsi * (rsi>rsi_high)
        # self.df['RSI_low'] = rsi * (rsi<rsi_low)

        # TRIX signal
        df['TRIX'] = df.ta.trix(length=14).TRIXs_14_9

        # Bollinger band
        length = 30
        bband = df.ta.bbands(length=length)
        bband['hlc'] = df.ta.hlc3()
        
        # Bollinger band upper signal
        bbu_signal = bband['hlc'] - bband['BBM_'+str(length)+'_2.0']
        bband['BBU_signal'] = (abs(bbu_signal) * (bbu_signal > 0) / bband['BBU_'+str(length)+'_2.0'])

        # Bollinger band lower signal
        bbl_signal = bband['BBM_'+str(length)+'_2.0'] - bband['hlc']
        bband['BBL_signal'] = (abs(bbl_signal) * (bbl_signal > 0)  / bband['BBL_'+str(length)+'_2.0'])
        
        df['BBU_signal'] = bband.BBU_signal
        df['BBL_signal'] = bband.BBL_signal

        # Slice again to remove nan values
        df = df.loc[span[0] : span[1]]

        # Scale all values
        scaler = MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])

        # Drop features that are not supposed to go into model
        df = df.drop(self.remove_features, axis=1)

        # Check null
        df.fillna(method='bfill', inplace=True)
        if df.isnull().sum().sum() > 0:
            print(self.ticker)
            print(df)
            print(span)
            print('FOUND NULL IN ST FEATURES - see data_loader -> get_st_features')
            assert False
        
        # Convert to numpy
        out = df.to_numpy()

        # Resize to increase fit/train speed
        height = int(out.shape[0] * self.st_scale_percent)
        out = cv2.resize(out, (out.shape[1], height), interpolation=cv2.INTER_AREA)

        return out


        
    def get_lt_features(self, df, span):
        '''
        LONG TERM
        long term features requires to have "LT_" in the beginning of the name
        '''

        # Wavelets as LT
        df['LT_close'] = df.close.copy()
        df['LT_RSI'] = df.RSI.copy()

        # Slice again to remove nan values
        df = df.loc[span[0] : span[1]]
        df = df[['LT_close', 'LT_RSI']]

        # Wavelet transform
        wt_trans = self.make_wavelet(df)

        return wt_trans



    def make_wavelet(self, signals):
        # Outputs wavelet transform, input pandas df
        # coef: (scales, time_steps)

        # Freq scales used in the transform
        scales = np.arange(1, self.wavelet_scales+1)

        # Allocate output
        out = np.zeros(shape=(len(scales), self.num_time_steps, len(signals.columns)))
        
        for idx, col in enumerate(signals.columns):
            signal = signals[col].diff().fillna(0).to_numpy()
            coef, _ = pywt.cwt(signal, scales, wavelet='gaus8') #gaus8 seems to give high res
            
            try:
                out[:, 0:coef.shape[1], idx] = abs(coef)
            except Exception as e:
                print(e)
                print('->>', coef.shape, idx)
                print('->>', out.shape)
                quit()

        # Scale all values
        scaler = MinMaxScaler()
        out = scaler.fit_transform( out.reshape(-1, out.shape[-1]) ).reshape(out.shape)

        # Resize to increase fit/train speed
        width = int(out.shape[1] * self.lt_scale_percent)
        height = int(out.shape[0] * self.lt_scale_percent)
        out = cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)

        return out #Format: (scales, time_steps, num_LT_features)



    def get_slice(self, span):
        ''' To get the close values for a certain span '''
        return self.df.loc[span[0] : span[1]].copy()



    def count_features(self):
        ''' Return number of features for long/short term df '''

        _span = (200,300) #Arbitrary span to be able to perform data process
        df_st, df_lt = self.data_process(_span)

        self.num_lt_features = len(df_lt.columns)
        self.num_st_features = len(df_st.columns)





if __name__ == '__main__':
    from environment import StockTradingEnv
    
    num_steps = 300
    wavelet_scales = 100
    dc = DataCluster(
        dataset='realmix',
        remove_features=['close', 'high', 'low', 'open', 'volume'],
        num_stocks=5,
        wavelet_scales=wavelet_scales,
        num_time_steps=num_steps
        )

    (st_shape, lt_shape) = dc.get_model_shape()
    print(st_shape)
    quit()
    collection = dc.collection

    env = StockTradingEnv(
        collection,
        look_back_window=num_steps,
        generate_est_targets=True
        )
    env.requested_target = 1
    obs = env.reset()
    


    # df = df[0:300]

    # df.plot(subplots=True)
    # plt.show()


    # def make_data():
    #     dc = DataCluster(
    #         dataset='realmix',
    #         remove_features=['close', 'high', 'low', 'open', 'volume'],
    #         num_stocks=0,
    #         num_time_steps=300
    #         )
    #     collection = dc.collection