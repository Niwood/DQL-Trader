from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import datetime
from matplotlib import pyplot as plt

from core import DataCluster, StockTradingEnv

wavelet_scales = 100
num_time_Steps = 300
data_cluster = DataCluster(
    dataset='realmix',
    remove_features=['close', 'high', 'low', 'open', 'volume'],
    num_stocks=1,
    wavelet_scales=wavelet_scales,
    num_time_steps=num_time_Steps
    )
collection = data_cluster.collection

env = StockTradingEnv(
    collection,
    look_back_window=num_time_Steps,
    static_initial_step=0,
    generate_est_targets=True)
env.requested_target = 1
obs = env.reset()



for i in range(10):
    obs, reward, done = env.step(0)
    print(env.df.loc[env.current_step, "close"])
obs, reward, done = env.step(2)

print('-'*10)
env.render()

# df.Close.plot()
# plt.show()