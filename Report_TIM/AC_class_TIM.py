import gym as gym
import numpy as np
class AlmgrenChrissEnv(gym.Env):
    """Custom Environment that follows the Gymnasium API for the Almgren-Chriss model as per the basic version in the paper 
    'Optimal Execution of Portfolio Transactions, Robert Almgren† and Neil Chriss, December 2000.' 
    This is a gymnasium environment to test it with RL more readily. Default impact models are linear but can be overloaded readily.
    AlmgrenChrissEnv is an inherited class of gym.env
    """
    #constructor of class
    def __init__(self, X0=1e6, N=10, T=10, sigma=0.95, param_temporary_impact=(0.0625,2.5e-6), param_permanent_impact=2.5e-7, initial_price=50,rndseed=None,drift =0, param_transient_impact = (2.5e-6, 2.5)):
        super(AlmgrenChrissEnv, self).__init__()
        """
        Initialize the model parameters.

        :param X0: Initial number of shares to liquidate.
        :param N: Number of time steps.
        :param T: Time horizon.
        :param sigma: Volatility of the asset price.
        :param_temporary_impact (eta): Temporary impact parameter. /currently defined as constants/ smaller if liquidity is high
        :param_permanent_impact (gamma in the paper): Permanent impact parameter. /currently defined as constants/  smaller if liquidity is high
        :param initial_price: Initial price of the asset.
        :param drift: should be linear to time step (expected return of the stock as modelled in black scholes)
        """
        # Initialize the model parameters
        self.X0 = X0
        self.N = N
        self.T = T
        self.sigma = sigma
        self.param_temporary_impact = param_temporary_impact
        self.param_permanent_impact = param_permanent_impact
        self.initial_price = initial_price
        self.drift = drift
        self.dt = self.T / self.N  # Time step size

        self.I = 0 # accumulation of market resilience
        
        self.param_transient_impact = param_transient_impact

        # Action space: The number of shares to liquidate at each step (continuous, gym.space.Box)
        self.action_space = gym.spaces.Box(low=0, high=self.X0, shape=(1,), dtype=np.float32) # only looking at selling inventory of X0 to 0, dimension is 1-D

        # Observation space: Remaining shares, time step, price, total proceeds etc.

        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0]), 
            high=np.array([self.X0, self.N, np.inf, np.inf]), # upper bounds of shares, time and price, which should be initial shares, total time steps and positive infinity
            dtype=np.float32
        )

        # Reset the environment state variables
        self.reset(seed = rndseed)

    def fn_permanent_price_impact(self, v):
        """Linear price impact function. Can ofc be overloaded"""
        return self.param_permanent_impact * v
    
    def fn_temporary_price_impact(self,v):
        """Linear price impact function. Can ofc be overloaded"""
        #[0] is bid-ask spread, [1] is temporary price coefficient
        return self.param_temporary_impact[0]*np.sign(v) + self.param_temporary_impact[1]* v
    
    def fn_transient_impact(self,v):
        """ Transient price impact function. Can ofc be overloaded"""
        #returns delta I - accumulating market resilience
        #[0] is the peak impact, [1] is decay function
        return (self.param_transient_impact[0]*v - self.param_transient_impact[1]*self.I) * self.dt 
    

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        Returns the initial observation and additional info.
        """
        super().reset(seed=seed)  # Set the seed for reproducibility # call parent class gym.env.reset 
   
        # state parameters: current inventory, current time, current price, current proceeds
        self.inventory = self.X0  # Reset remaining shares
        self.time_step = 0  # Reset time step
        self.price = self.initial_price  # Reset price
        self.total_actual_proceeds = 0  # Reset total proceeds
        self.actual_proceeds=0 # Reset actual proceeds
        self.done = False
        self.I = 0

        # Return the initial observation and info
        self.state = np.array([self.inventory, self.time_step, self.price, self.total_actual_proceeds, self.I], dtype=np.float32)
        info = {}
        
        return self.state, info

    def step(self, action: np.array): 
        """
        Apply the action (liquidate shares), update the state, and return the observation.
        
        :param action: firt entry is the number of shares to liquidate at this step.
        """
        # Clip action to ensure it’s within the allowed range
        action = np.clip(action, 0, self.inventory)

        # Calculate trading rate and update remaining shares
        trading_rate = action[0] / self.dt
        new_inventory = self.inventory - action[0] #subtract the liquidated shares from the inventory

        # Calculate actual proceeds from the action - temporary impact equation
        self.actual_proceeds = action[0] * (self.price - self.fn_temporary_price_impact(trading_rate)) 
        new_total_actual_proceeds = self.total_actual_proceeds + self.actual_proceeds

       #next, we update the prices resulting from the action:
        
        # price impact
        permanent_impact = self.fn_permanent_price_impact(trading_rate)
        
         #  noise normal distribution of 0 mean and variance 1
        xi =np.random.normal(0, 1)

        
        #update price:
        new_price = self.price  + (self.dt * self.drift) - (self.dt * permanent_impact) + (self.sigma * np.sqrt(self.dt)* xi) - self.fn_transient_impact(trading_rate)

        #update market resilience
        new_I = self.fn_transient_impact(trading_rate) + self.I

        #new_price = self.price  + self.dt * self.drift - self.dt * permanent_impact + self.sigma * np.sqrt(self.dt)* xi + transient impact

        # Update step count
        new_time_step = self.time_step + 1
        
        #update states and enter new time step, shoudl have all the parameters of state 
        self.time_step= new_time_step
        self.price = new_price
        self.total_actual_proceeds  = new_total_actual_proceeds
        self.inventory = new_inventory
        self.I = new_I
        
        # Check if the liquidation is done (i.e., no more shares to liquidate or last step)
        self.done= self.time_step > self.N

        # Calculate reward as negative implementation shortfall or the actual proceeds (capture)
        # Will need to redfine this if we incorporating drift
        ideal_proceeds = (self.X0 - self.inventory) * self.initial_price
        
        #ideal_proceeds = action[0] * self.initial_price

        implementation_shortfall = ideal_proceeds - self.total_actual_proceeds
        
        # Might wish to set the following to the actual_proceeds instead or to - implementation_shortfall
        # How do we take care of variance/ risk aversion parameter?

        #reward = actual_proceeds
        reward = -implementation_shortfall # We want to minimize shortfall

        # Observation: remaining shares, new current step, new current price
        self.state = np.array([self.inventory, self.time_step, self.price, self.total_actual_proceeds, self.I], dtype=np.float32)
        info = {}
        # Return observation, reward, done, truncated, and additional info
        return self.state, reward, self.done, info

    def render(self, mode='human'):
        """Render the environment state."""
        print(f"Step: {self.time_step}, Remaining Shares: {self.inventory}, Price: {self.price:.2f}, Total_Proceeds:{self.total_actual_proceeds}, Market_Resilience:{self.I}")
