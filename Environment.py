import numpy as np
import gym
from gym import spaces
from sklearn import preprocessing
from cryptocompare import get_price as currentPrice


# Agent Environment
class GymEnvironment(gym.Env):
    # GYM environment setup
    MAX_TRADING_SESSION = 10000
    metadata = {'render.modes': ['live', 'file', 'none']}
    scaler = preprocessing.MinMaxScaler()
    viewer = None

    def __init__(self, df, init_balance=10000, init_btc_balance=0, lookback_window=50, commission=0.0025, serial=False):
        super(GymEnvironment, self).__init__()

        self.df = df  # Dataframe
        self.init_balance = init_balance  # initial account balance
        self.balance = init_balance  # initial account balance
        self.init_btc_balance = init_btc_balance  # initial BTC balance
        self.lookback_window = lookback_window  # time steps in the past the agent will observe at each step
        self.commission = commission  # flat commission from bitbns 0.25%
        self.serial = serial  # data frame will be traversed in random slices by default
        # Action space: buy, hold, sell (3), amounts for buy: 1/10, sell 3/10
        self.action_space = spaces.MultiDiscrete([3, 10])

        # Obseriving OHLCV values, trade history, and net worth
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, lookback_window + 1), dtype=np.float16)

    def _reset_session(self):
        self.current_step = 0
        if self.serial:
            self.steps_left = len(self.df) - self.lookback_window - 1
            self.frame_start = self.lookback_window

        else:
            self.steps_left = np.random.randint(1, self.MAX_TRADING_SESSION)
            self.frame_start = np.random.randint(self.lookback_window, len(self.df) - self.steps_left)

        self.df_subset = self.df[self.frame_start - self.lookback_window:self.frame_start + self.steps_left]

    def _next_observation(self):
        self.FromW = len(self.df_subset) - self.lookback_window - 1

        obsrv = np.array([
            self.df_subset['Open'].values[self.FromW:],
            self.df_subset['High'].values[self.FromW:],
            self.df_subset['Low'].values[self.FromW:],
            self.df_subset['Close'].values[self.FromW:],
            self.df_subset['Volume_(BTC)'].values[self.FromW:],
        ])

        scaled_history = self.scaler.fit_transform(self.ac_history)
        obsrv = np.append(obsrv, scaled_history[:, -(self.lookback_window + 1):], axis=0)

        return obsrv

    def _take_action(self, action, current_price):
        action_type = action[0]
        amount = action[1] / 10

        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0

        if action_type < 1:
            btc_bought = self.balance / current_price * amount
            cost = btc_bought * current_price * (1 + self.commission)
            self.btc_balance += btc_bought
            self.balance -= cost

        elif action_type < 2:
            btc_sold = self.btc_balance * amount
            sales = btc_sold * current_price * (1 - self.commission)
            self.btc_balance -= btc_sold
            self.balance += sales

        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({
                'step': self.frame_start + self.current_step,
                'amount': btc_sold if btc_sold > 0 else btc_bought,
                'total': sales if btc_sold > 0 else cost,
                'type': "sell" if btc_sold > 0 else "buy"
            })

        self.net_worth = self.balance + self.btc_balance * current_price
        self.ac_history = np.append(self.ac_history, [
            [self.net_worth],
            [btc_bought],
            [cost],
            [btc_sold],
            [sales]
        ], axis=1)

    def step(self, action):
        # current_price = float(self.df_subset['Close'].iloc[self.current_step]) + 0.01
        current_price = currentPrice('BTC', curr='USD').get('BTC').get('USD') + 0.01
        self._take_action(action, current_price)
        self._reset_session()
        obs = self._next_observation()
        reward = self.net_worth
        done = self.net_worth <= 0

        return obs, reward, done, {}

    def reset(self):
        self.net_worth = self.init_balance
        self.balance = self.init_balance
        self.btc_balance = self.init_btc_balance
        self._reset_session()
        self.ac_history = np.repeat([[self.net_worth], [0], [0], [0], [0]], self.lookback_window + 1, axis=1)
        self.trades = []

        return self._next_observation()

    def render(self, mode='human', **kwargs):
        return self.balance, self.btc_balance, self.net_worth

class InitialBalanceClass:
    bal = 1000.0
    b_bal = 0.0
    net_worth = bal

    def __init__(self):
        pass
    def return_bal(self):
        return self.bal

    def return_b_bal(self):
        return self.b_bal

    def return_net(self):
        return self.net_worth
