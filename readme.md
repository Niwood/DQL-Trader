# Deep Q Learning Trader


## Summary
Multi classification network trained as a two-headed 1D/2D ConvNet to be used as a recommender system for trade positions. The model uses Keras DNN framework and the StockTradingEnvironment is built on the superclass GYM from OpenAI. The features consistst of various technical indicators (RSI, Bollinger Band, TRIX, MACD) for the 1D ConvNet and a frequency decomposition spectrum using continuous wavelet transform (to retain the time domain of the frequency spectrum) using the real-valued Gaussian wavelet for the 2D ConvNet.

First, the network is pre-trained via supervised learning with target actions (buy/sell/hold) determined by Locally Weighted Scatterplot Smoothing (LOWESS) on the close value. The second and third gradient of LOWESS will reveal local max/min points, thresholding the extreme points above/below the q% quantile to only keep the most significant points.

Second, the network is trained via reinforcement learning as the policy network in Deep Q Learning utilizing the StockTradingEnvironment. The agent consistst of two copies of the network, one for prediction and another to act as the target network. Both networks are assigned the pre-trained weights from the supervised learning step. The reward the agent is given is the theoretical maximum return during a given finite time frame.

For each epoch (containing a certain amount of episodes) the model will be saved and reloaded to perform a full evaluation. The agent uses Experience Replay And Replay Memory to break any correlation between consecutive samples.

There is also an Streamlit application built for evaluating the model.

### Further Reading

---
## Credit
- StockTradingEnvironment
..

## TODO
- [x] Confusion matrix for pre-training and save
- [ ] Visualize weights during training
- [ ] Setup tests
- [x] 2D convNet for wavelets
- [ ] Deploy TF Lite model
- [ ] Change the data process pipeline as a callback from the env to the datapack
- [ ] Low utilization of the GPU... Whyyy


## Installation

```python
#pip install -r requirements.txt
```


## Usage

```python
TBD
```

Streamlit application
```python
streamlit run app.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)

