import sched
import time
import os
import numpy as np
import pandas as pd
import wget
import streamlit as st
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from Environment import GymEnvironment
from Environment import InitialBalanceClass

s = sched.scheduler(time.time, time.sleep)

#Load model
loadedModel = PPO.load("model/ppo_RL_trader.zip")


@st.cache
def importing_dataset():
    cwd = os.getcwd()
    filename = cwd + '/dataset' + '/' + 'Binance_BTCUSDT_1h.csv'  # get the full path of the file
    if os.path.exists(filename):
        os.remove(filename)
    wget.download('http://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_1h.csv', filename)
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df.reset_index(inplace=True)
        df.columns = ['Timestamp', 'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(USDT)', 'Trade_count']
        df = df.drop(index=0, axis=0)
        return df
    else:
        return 'File Not Found'


def exec_bot(sc):
    print('\n\n')
    DF = DummyVecEnv([lambda: GymEnvironment(importing_dataset(),
                                             init_balance=InitialBalanceClass().return_bal(),
                                             init_btc_balance=InitialBalanceClass().return_b_bal(),
                                             serial=True)])

    obsDF = DF.reset()
    action, _states = loadedModel.predict(obsDF)
    obs, rewards, done, info = DF.step(action)
    print('Initial USD Balance: {} Initial BTC Balance: {}'.format(InitialBalanceClass().return_bal(),
                                                                   InitialBalanceClass().return_b_bal()))
    newBal, newBTCbal, netW = DF.render()
    print('Actions: ', action)
    print('Current USD Balance: {} Current BTC Balance: {}'.format(newBal, newBTCbal))
    print('Current Net worth: ', netW)
    InitialBalanceClass.bal = newBal
    InitialBalanceClass.b_bal = newBTCbal
    InitialBalanceClass.net_worth = netW
    bhold.text('Account Balance: ' + str(newBal))
    bithold.text('BTC Balance: ' + str(newBTCbal))
    nhold.text('Net Worth: ' + str(netW))
    data.append([InitialBalanceClass().return_net()])
    graph.add_rows(np.array(data))
    s.enter(1, 1, exec_bot, (sc,))
    return 'Bot running....'


#s.enter(5, 1, exec_bot, (s,))
#s.run()
#"""
#Configure Stremlit app

user_input = st.number_input('Add amount ($)', 100, key='amtInp')

c1, c2, c3 = st.beta_columns(3)


bhold = c1.text('Account Balance: '+str(InitialBalanceClass().return_bal()))
bithold = c2.text('BTC Balance: '+str(InitialBalanceClass().return_b_bal()))
nhold = c3.text('Net Worth: '+str(InitialBalanceClass().return_net()))

c1, c2, c3 = st.beta_columns(3)

bal_button = c1.button('Add balance')
run_bot = c2.button('Run Bot')
stop_bot = c3.button('Stop Bot')

data = [[InitialBalanceClass().return_net()]]
graph = st.line_chart(np.array(data))

if bal_button:
    InitialBalanceClass.bal += user_input
    bhold.text('Account Balance: '+str(InitialBalanceClass().return_bal()))

if run_bot:
    s.enter(1, 1, exec_bot, (s,))
    s.run()

if stop_bot:
    st.stop()


#"""
