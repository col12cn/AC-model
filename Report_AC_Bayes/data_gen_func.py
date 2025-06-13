import numpy as np
import pandas as pd

#---------------
#---------------
def data_gen(env, fn_strategy, sample_size):    
    proceeds_list =np.array([]) #self.actual_proceeds
    trading_rate_list = np.array([]) #action/self.dt
    current_price_list =np.array([]) #self.price before step
    new_price_list = np.array([])#self.price after step
    for i in range(sample_size):
        env.state, info = env.reset()
        done = False
        while not done:
            action = np.array([fn_strategy(env.state,env.N, seed = env.seed)])
            current_price_list = np.append(current_price_list, float(env.price))
            trading_rate_list = np.append(trading_rate_list,float(action[0]/env.dt))
            env.state, reward, done, info = env.step(action)
            proceeds_list = np.append(proceeds_list,float(env.actual_proceeds/action[0]))
            new_price_list = np.append(new_price_list,float(env.price))
    price_change_list = current_price_list-new_price_list
    trading_cost_list = current_price_list - proceeds_list
    return trading_rate_list, trading_cost_list, price_change_list
#---------------
#---------------
def random_liquidation_strategy(state, N, seed = None):
    """
    Random liquidation strategy: liquidates a random number of shares within valid bounds at each step.
    
    :param remaining_shares: The current number of remaining shares.
    :return: Number of shares to sell.
    """
    remaining_shares, time_step, price, actual_proceeds =state
    if time_step == N:
        return remaining_shares
    else:
        np.random.seed(seed) 
        trade = np.random.uniform(1e3, remaining_shares)
        return trade

#---------------
#---------------
def extract(df, size, uniform = 1, seed = 45):
    if uniform == 1:
        range1 = np.linspace(0, df['Trading Rate'].max(),5)
        samp = pd.DataFrame()
        for i in range(4):
            condition = (df['Trading Rate'] >= range1[i]) & (df['Trading Rate'] <= range1[i+1])
            filtered_df = df[condition]
            sampled_df = filtered_df.sample(int(size/4), random_state = seed)
            samp = pd.concat([samp, sampled_df], axis=0, ignore_index=True)
    else:
        samp = df.sample(n=size)
    return samp

#---------------
#---------------

def price_change(sample):
    trading_rate = sample['Trading Rate'].to_numpy()
    price_change = sample['Price Change'].to_numpy()
    return trading_rate.reshape(-1,1), price_change.reshape(-1,1)      
