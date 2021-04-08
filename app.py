import streamlit as st
from streamlit import caching

import pandas as pd
import numpy as np
from backtesting.test import GOOG
from pathlib import Path
from datetime import datetime
import time
import json
import threading

import warnings
warnings.filterwarnings('ignore')


import numpy as np
from bokeh.plotting import figure
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# TO RUN: 
# streamlit run app.py



class App:
    def __init__(self):
        
        # Constants
        self.models_folder = Path.cwd() / 'models'

        # Refresh thread
        # self.refresh_thread = threading.Thread(target=self.refresh, args=(True,))

        # Render sidebar
        self.sidebar()

        # Title
        st.title(f'{self.page}')
        st.text(f'MODEL-ID: {self.model_id}') 

        # Page selector
        if self.page == 'Training':
            self.training()
        

    # def refresh(self, _refresh):
    #     while _refresh:
    #         time.sleep(5)
    #         st.experimental_rerun()
            # caching.clear_cache()


    def sidebar(self):
        st.sidebar.title('DQN Trader')

        # Page option
        self.page = st.sidebar.selectbox("Page", ("Training", "Model evaluation", "Inference"))

        # Model option
        all_model_date = [datetime.fromtimestamp(int(x.stem)) for x in self.models_folder.glob('*/')]
        all_model_date.reverse()
        self.model_date = st.sidebar.selectbox("Model", all_model_date)
        self.model_id = int(datetime.timestamp(self.model_date))

        # Moving average window
        self.ma_window = st.sidebar.slider('Moving average', 1, 100, 10)


        # Refresh
        if st.sidebar.button('Refresh'):
            # self.refresh_thread.start()
            pass
        # else:
        #     try: self.refresh_thread.join()
        #     except: pass
        

    def training(self):
        # @st.cache
        def load_estats():
            try:
                return pd.read_pickle(self.models_folder / str(self.model_id) / 'estats.pkl')
            except:
                return pd.DataFrame()
        def load_astats():
            try:
                return pd.read_pickle(self.models_folder / str(self.model_id) / 'astats.pkl')
            except:
                return pd.DataFrame()

        # Load data
        estats = load_estats()
        astats = load_astats()
        
        # Architecture
        with open(self.models_folder / str(self.model_id) / 'arch.txt', 'r') as reader:
            arch_text = reader.read()
        st.text(arch_text)

        # Meta data
        with open(self.models_folder / str(self.model_id) / 'metadata.json', 'r') as fp:
            metadata = json.load(fp)
        st.json(metadata)

        ### Episode statistics

        # Expander
        episode_statistics = st.beta_expander('Episode statistics')

        # REWARD
        episode_statistics.markdown(f'** Reward **')
        fig = go.Figure()
        # BUY
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.buyReward,
            mode='markers',
            opacity=0.2,
            line=dict(color='blue', width=1),
            name='BuyReward',
            showlegend=False
            ))
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.buyReward.rolling(window=self.ma_window).mean(),
            mode='lines',
            line=dict(color='blue',width=2,dash='dash'),
            name='BuyReward'
            ))
        # SELL
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.sellReward,
            mode='markers',
            opacity=0.2,
            line=dict(color='green', width=1),
            showlegend=False
            ))
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.sellReward.rolling(window=self.ma_window).mean(),
            mode='lines',
            line=dict(color='green',width=2,dash='dash'),
            name='SellReward'
            ))
        # HOLD
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.holdReward,
            mode='markers',
            opacity=0.2,
            line=dict(color='orange', width=1),
            showlegend=False
            ))
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.holdReward.rolling(window=self.ma_window).mean(),
            mode='lines',
            line=dict(color='orange',width=2,dash='dash'),
            name='HoldReward'
            ))
        # AVG
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.avgReward,
            mode='markers',
            opacity=0.2,
            line=dict(color='red', width=4),
            name='AvgReward',
            showlegend=False
            ))
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.avgReward.rolling(window=self.ma_window).mean(),
            mode='lines',
            line=dict(color='red',width=3),
            name='AvgReward'
            ))
        # AXES
        fig.update_xaxes(
            title_text = "Episode",
            zeroline = True,
            range = [0,len(estats)])
        fig.update_layout(template="plotly_white")
        episode_statistics.plotly_chart(fig, use_container_width=True)



        # RETURN
        episode_statistics.markdown(f'** Return on asset and Buy&Hold **')
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        # fig = go.Figure()
        # BUY AND HOLD
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.buyAndHold,
            mode='markers',
            opacity=0.2,
            line=dict(color='red', width=2),
            name='buyAndHold',
            showlegend=False
            ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.buyAndHold.rolling(window=self.ma_window).mean(),
            mode='lines',
            line=dict(color='red',width=2),
            name='buyAndHold'
            ), row=1, col=1)
        # NET WORTH CHANGE
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.netWorthChng,
            mode='markers',
            opacity=0.2,
            line=dict(color='blue', width=2),
            name='netWorthChng',
            showlegend=False
            ), row=1, col=1)
        # fig.add_trace(go.Scatter(
        #     x=estats.index,
        #     y=estats.netWorthChng.rolling(window=self.ma_window).mean() + estats.netWorthChng.rolling(window=self.ma_window).std()/2,
        #     mode='lines',
        #     fillcolor='indigo',
        #     line=dict(width=0),
        #     name='netWorthChng Upper Bound',
        #     showlegend=False
        # ))
        # fig.add_trace(go.Scatter(
        #     x=estats.index,
        #     y=estats.netWorthChng.rolling(window=self.ma_window).mean() - estats.netWorthChng.rolling(window=self.ma_window).std()/2,
        #     mode='lines',
        #     marker=dict(color='rgb(255,255,255,255.02)'),
        #     line=dict(width=0),
        #     fill='tonexty',
        #     name='netWorthChng Upper Bound',
        #     showlegend=False
        # ))
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.netWorthChng.rolling(window=self.ma_window).mean(),
            mode='lines',
            line=dict(color='blue',width=2),
            name='netWorthChng'
            ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=(estats.netWorthChng - estats.buyAndHold).rolling(window=self.ma_window).mean(),
            mode='lines',
            line=dict(color='green',width=2),
            name='MA delta'
            ), row=2, col=1)
            
        fig.update_xaxes(
            title_text = "Episode",
            zeroline = True,
            range = [0,len(estats)], row=2, col=1)
        fig.update_xaxes(
            zeroline = True,
            range = [0,len(estats)], row=1, col=1)
        fig.update_layout(template="plotly_white")
        episode_statistics.plotly_chart(fig, use_container_width=True)



        # Triggers
        episode_statistics.markdown(f'** Triggers **')
        estats['teomaxTrigger'] = estats.epsilon/3 + (1-estats.epsilon)
        fig = go.Figure()
        # EPSILON
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.epsilon,
            mode='lines',
            line=dict(color='pink', width=2, dash='dash'),
            name='epsilon',
            showlegend=True
            ))
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.teomaxTrigger,
            mode='lines',
            line=dict(color='pink', width=2),
            fill='tonexty',
            name='teomaxTrigger',
            showlegend=True
            ))
        # BUY
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.buyTrigger,
            mode='markers',
            opacity=0.5,
            line=dict(color='blue'),
            showlegend=False
            ))
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.buyTrigger.rolling(window=self.ma_window).mean(),
            mode='lines',
            line=dict(color='blue', width=2),
            name='buyTrigger'
            ))
        # SELL
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.sellTrigger,
            mode='markers',
            opacity=0.5,
            line=dict(color='green'),
            showlegend=False
            ))
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.sellTrigger.rolling(window=self.ma_window).mean(),
            mode='lines',
            line=dict(color='green', width=2),
            name='sellTrigger'
            ))
        # HOLD
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.holdTrigger,
            mode='markers',
            opacity=0.5,
            line=dict(color='orange'),
            showlegend=False
            ))
        fig.add_trace(go.Scatter(
            x=estats.index,
            y=estats.holdTrigger.rolling(window=self.ma_window).mean(),
            mode='lines',
            line=dict(color='orange', width=2),
            name='holdTrigger'
            ))

        # AXES
        fig.update_xaxes(
            title_text = "Episode",
            zeroline = True,
            range = [0,len(estats)])
        fig.update_layout(template="plotly_white")
        episode_statistics.plotly_chart(fig, use_container_width=True)



        # Balance
        episode_statistics.markdown(f'** Net worth **')
        _cols = ['amountBalance', 'amountAsset']
        episode_statistics.line_chart(estats[_cols])


        #### Episode table
        episode_table = st.beta_expander('Episode table')
        episode_table.table(estats)


        ##########################
        #### Assessment statistics
        assessment_statistics = st.beta_expander('Assessment statistics')

        # TRIGGER - ASSESSMENT
        assessment_statistics.markdown('** Trigger **')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=astats.index,
            y=astats.buyTrigger,
            mode='lines',
            line=dict(color='blue',width=2),
            name='BuyTrigger'
            ))
        fig.add_trace(go.Scatter(
            x=astats.index,
            y=astats.sellTrigger,
            mode='lines',
            line=dict(color='green',width=2),
            name='SellTrigger'
            ))
        fig.add_trace(go.Scatter(
            x=astats.index,
            y=astats.holdTrigger,
            mode='lines',
            line=dict(color='orange',width=2),
            name='HoldTrigger'
            ))
        fig.update_layout(template="plotly_white")
        assessment_statistics.plotly_chart(fig, use_container_width=True)

        # RETURN - ASSESSMENT
        assessment_statistics.markdown('** Return **')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=astats.index,
            y=astats.netWorthChng,
            mode='lines',
            line=dict(color='blue',width=2),
            name='NetWorthChng'
            ))
        fig.add_trace(go.Scatter(
            x=astats.index,
            y=astats.buyAndHold,
            mode='lines',
            line=dict(color='red',width=2),
            name='BuyAndHold'
            ))
        fig.update_layout(template="plotly_white")
        assessment_statistics.plotly_chart(fig, use_container_width=True)

        #### Assessment table
        assessment_table = st.beta_expander('Assessment table')
        assessment_table.table(astats)


if __name__ == '__main__':
    app = App()

