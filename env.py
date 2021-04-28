from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm

from core import DataCluster, StockTradingEnv2

# wavelet_scales = 100
# num_time_Steps = 300
# data_cluster = DataCluster(
#     dataset='realmix',
#     remove_features=['close', 'high', 'low', 'open', 'volume'],
#     num_stocks=1,
#     wavelet_scales=wavelet_scales,
#     num_time_steps=num_time_Steps
#     )
# collection = data_cluster.collection

env = StockTradingEnv2()

obs = env.reset()





for i in tqdm(range(90)):
    obs, reward, done = env.step(0)
obs, reward, done = env.step(2)

print('-'*10)
env.render()

# df.Close.plot()
# plt.show()