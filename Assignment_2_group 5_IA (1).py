#!/usr/bin/env python
# coding: utf-8

# # Assignment 2 Solution
# GROUP 5
#SANSKRUTI MITTAL 
#KAUSHIK SANJIV
#XIANGLIN CHENG
#MINGHAO WANG
# ### Import modules

# In[1]:


import pandas            as pd   
import numpy             as np   
import scipy             as scp  
import matplotlib.pyplot as plt  
import os                as os   
from datetime import date as dd  # for dates
from scipy import optimize
from scipy import stats
import statsmodels.api as sm


# In[2]:


np.set_printoptions(precision=2) # Set precision of numpy arrays while printing uless specified otherwise
#%% Set directories and define files
os.chdir(os.path.abspath(''))   # Change directory to location of the script
datafile1 = 'industry_returns.xlsx'


# In[3]:


#%% Define helper function for Sharpe Ratio (Since this is used very frequently)
def Shratio(w,mu,Sigma,Rf): 
    portfolio_return = np.inner(mu, w) - Rf    # portfolio excess return
    portfolio_vol    = np.sqrt((w.T@Sigma@w))  # portfolio volatility
    return portfolio_return / portfolio_vol


# In[4]:


#%% Read data
returns_data = pd.read_excel(datafile1, usecols = "B:M", header = 1)
dates = pd.read_excel(datafile1, usecols = "A", header = 1)
dates.columns = ['dates']
dates['dates'] = pd.to_datetime(dates['dates'],format='%Y%m')


# In[5]:


# Set parameters
Rf = .04 / 12                         # monthly risk free rate
n = returns_data.shape[1]             # number of risky assets (columns)列
T = returns_data.shape[0]             # number of observations (rows)行
ret = returns_data / 100              # convert returns from percent to decimals
mu = ret.mean()                                 # mean returns 
Sigma = ret.cov()                            # covariance matrix


# ## Part 1
1. Using returns on the 12 industry portfolios in columns B-M, construct and plot the efficient frontier for a reasonable range of expected return values (e.g., 0 - 1.5% per month) 
# In[8]:


# Unconstrained optimization:
w0 = np.ones((n,1)) / n  # starting weights (1/12)
cons = ({'type' : 'eq', 'fun': lambda x: np.sum(x) - 1}) # portfolio weights sum to 1
sol = scp.optimize.minimize(lambda w: -Shratio(w, mu, Sigma, Rf), x0 = w0, constraints = cons)
w_uncon = sol.x
val_uncon = sol.fun
print(w_uncon)
print(val_uncon)


# In[9]:


# Construct EML (CML slope is sharp ratio)
A = (mu.transpose().dot(np.linalg.inv(Sigma))).dot(mu) 
B = mu.T.dot(np.linalg.solve(Sigma, np.ones((n,1))))
C = np.ones((n,1)).T.dot(np.linalg.solve(Sigma, np.ones((n,1))))
mu0 = np.arange(0,0.015,.001)
varR0 = (C*(mu0**2) - 2*B*mu0 + A) /( A*C-B**2)
stdR0 = np.sqrt(varR0)
dummy = np.arange(0,np.max(stdR0)+.01, .01)
Sharpe_ratio = -val_uncon                          # From the optimization above
Line_sharpe = Rf + Sharpe_ratio * dummy
plt.plot(stdR0.reshape((mu0.shape)), mu0, linewidth = .8)
plt.plot(dummy, Line_sharpe, color = 'green', linewidth = .8)
print('Answer:')
plt.show()

2. Report the weights of the tangency portfolio (12 by 1 vector). How large is this portfolio’s Sharpe ratio? 
# In[10]:


print('Answer:')
print(f'Annualized Sharpe ratio of tangency portfolio is {-np.sqrt(12) * val_uncon :.4}.')
print(f'Weights of tangency portfolio is {w_uncon}.')

3. uppose we constrain the weights on each of the industries to lie between zero and 25%. Compute the optimal weights and the associated Sharpe ratio of the portfolio that is subject to these limits on the portfolio weights. (Again, you can use the python code to get started).
# In[11]:


# 1.3: Constrained optimization
w0 = np.ones((n,1)) / n  # starting weights
cons = ({'type' : 'eq', 'fun': lambda x: np.sum(x) - 1}) # portfolio weights sum to 1
bnds = scp.optimize.Bounds(0,.25) # upper and lower bounds

sol = scp.optimize.minimize(lambda w: -Shratio(w, mu, Sigma, Rf), x0 = w0, constraints = cons, bounds=bnds)                             # Perfrom optimization here
w_con = sol.x
val_con = sol.fun
print('Answer:')
print(f'Annualized Sharpe ratio of tangency portfolio under constraints is {-np.sqrt(12) * val_con :.4}.')
print(f'Weights of tangency portfolio is {w_con}.')

4. Using data up to 2009:12, compute the portfolio weights on the 12 industries which maximize the Sharpe ratio. Then, using these weights, compute the actual Sharpe ratio on returns data for the remaining sample, 2010:01-2022:07. Also compute the Sharpe ratio for 2010:01-2022:07 for an equal-weighted portfolio that puts a weight of 1/12 in each industry. How do the two strategies (optimal Sharpe ratio weights vs. equal weights) compare in terms of performance?
# In[12]:


# 1.4: Training weights (Sample upto 2009)
ret_train = ret[dates['dates'].dt.year<2010]     # Training sample
ret_test = ret[dates['dates'].dt.year>=2010]     # Testing sample
mu = ret_train.mean()                          # mean returns of the training sample
Sigma = ret_train.cov()                       # covariance of the trainig sample
w0 = np.ones((n,1)) / n
cons = ({'type' : 'eq', 'fun': lambda x: np.sum(x) - 1}) # portfolio weights sum to 1
sol = scp.optimize.minimize(lambda w: -Shratio(w, mu, Sigma, Rf), x0 = w0, constraints = cons)                                            # Solve the optimization problem for training weights.
w_train = sol.x
val_train = sol.fun
print('Answer:')
print(f'Optimal portfolio weights of of training data is {w_train}.')
print(f'Sharp ratio is {-val_train}.')


# In[13]:


# Test data Sharpe Ratio
mu_test = ret_test.mean()                                           # mean returns of testing set
Sigma_test = ret_test.cov()                                        # covariance matrix of testing set
val_test = Shratio(w_train,mu_test,Sigma_test,Rf)*(12**0.5)                  # Use the Shratio with approopriate weights
print('Answer:')
print(f'Annualized Sharpe using optimal training weights is {val_test:.4}')


# In[14]:


# Equal weighted Sharpe Ratio
w_eq = np.ones((n)) / n  # Equal weights to all assets
val_test_eqweight = Shratio(w_eq,mu_test,Sigma_test,Rf)*(12**0.5)    # Use the Shratio with approopriate weights
print('Answer:')
print(f'Annualized Sharpe using equal weights is {val_test_eqweight:.4}')


# Thus the sharpe ratio with equal weights is better.

# # Part 2: Momentum

# In[15]:


#%% Part 2: Momentum
returns = returns_data.values
T = returns.shape[0]                   # Number of time periods
n = returns.shape[1]                   # Number of assets

avg_ret_11months = np.ones((T,n))      # Matrix with average returns for past 11 months for n firms
winner_portfolio_ret = np.ones((T,1))  # return of the winning portfolio
loser_portfolio_ret = np.ones((T,1))
winner_ind = np.ones((T,1))            # index  of winning industry in that month
loser_ind = np.ones((T,1))


# In[16]:


# loop over time series, start from 1950/01
burn = sum((dates['dates'].dt.year<1950) == True) # Number of time periods before 1950/01 that must be "burned"


# In[59]:


for i in range((burn),T): 
    # loop over industries
    for j in range(n):
        avg_ret_11months[i,j] = returns[i-12:i-1,j].mean()     # take average of previous 11 months with one-skip-month
        
        sorted_ret = np.sort(avg_ret_11months[i,:])
        order = np.argsort(avg_ret_11months[i,:])
        
        long = order[-1]                          # Index of the winner industry
        short = order[0]                         # Index of the loser industry
    
        winner_portfolio_ret[i] = returns[i,long]   
        loser_portfolio_ret[i] = returns[i,short]
    
        winner_ind[i] = long
        loser_ind[i] = short
    
spread = winner_portfolio_ret - loser_portfolio_ret

1. Compute the average returns of the Winner (W) and Loser (L) portfolios as well as that of the WML (winner minus loser) portfolio that goes long in winners and shorts losers. Also compute the standard deviation of the W, L, WML portfolios. What do you conclude?
# In[60]:


# 2.1: Winner Minus Loser
wl_ret = pd.DataFrame(data=np.hstack((winner_portfolio_ret,loser_portfolio_ret,spread)), columns=["winner_ret", "loser_ret",'spread'])
wl_ret = wl_ret.loc[burn:]
print('Answer:')
print(f'Annualized average return for winner, loser and spread is {12*wl_ret.mean()} percent')
print(f'Annualized standard deviation for winner, loser and spread is {np.sqrt(12)*wl_ret.std()} percent')

2. Plot the time-series of rolling 3- and 10-year average returns on the industry Winner minus Loser (WML) portfolio. Comment on what you see, in particular, has the performance of the WML portfolio deteriorated over time?
# In[61]:


# 2.2: Rolling time series (36 months)
print('Answer:')
wl_ret_rolling3 = wl_ret[["winner_ret", "loser_ret"]].rolling(36, min_periods=36).apply(lambda x: (np.prod(1.0+x/100)-1)/36*12)
wl_ret_rolling3['spread'] = wl_ret_rolling3.iloc[(35):,0]-wl_ret_rolling3.iloc[(35):,1]
wl_ret_rolling3['dates'] = dates[(burn+35):]
plt.plot(wl_ret_rolling3['dates'],wl_ret_rolling3['spread'])                        # Plot the spread
plt.show()


# In[62]:


# 2.2: Rolling time series (120 months)
print('Answer:')
wl_ret_rolling10 = wl_ret[["winner_ret", "loser_ret"]].rolling(120, min_periods=120).apply(lambda x: (np.prod(1.0+x/100)-1)/120*12)
wl_ret_rolling10['spread'] = wl_ret_rolling10.iloc[(119):,0]-wl_ret_rolling10.iloc[(119):,1]
wl_ret_rolling10['dates'] = dates[(burn+119):]
plt.plot(wl_ret_rolling10['dates'],wl_ret_rolling10['spread'])                        # Plot the spread
plt.show()

3. Which industries are particularly often included in the Winner portfolio?
# In[63]:


# 2.3: Winner industries
print('Answer:')
winner_ind_df = pd.DataFrame(data= winner_ind[(burn+1):], columns=['industry'])
winner_count = winner_ind_df['industry'].value_counts() #number of times in winner portfolio
winner_order = returns_data.columns[winner_count.index.astype(int)].to_list()
print(f'From most often to least ofter, winner portfolio consists of {winner_order}' )

4. Which industries are particularly often included in the Loser portfolio?
# In[64]:


# 2.4: Loser industries
print('Answer:')
loser_ind_df = pd.DataFrame(data= loser_ind[(burn+1):], columns=['industry'])
loser_count = loser_ind_df['industry'].value_counts()                        #number of times in loser portfolio
loser_order = returns_data.columns[loser_count.index.astype(int)].to_list()
print(f'From most often to least ofter, loser portfolio consists of {loser_order}' )

5. How large is the ‘turnover’ on the WML portfolio, i.e., how often are different industries rotated in and out of the WML portfolio?
# In[98]:


# 2.5: Turnover
winner_turn = winner_ind_df.diff().fillna(0).astype(bool).astype(int).squeeze()# Time series of number of 0 or 1. 1 if the winner changed from last time period
loser_turn = loser_ind_df.diff().fillna(0).astype(bool).astype(int).industry # Same for loser
turnover_rate = sum(winner_turn|loser_turn)/(T-burn)
print('Answer:')
print(f'Frequency that either winner or loser portfolio changes is {turnover_rate:.4}' )

6. Discuss if the WML portfolio is a desirable portfolio that can be implemented in practice. Use different measures of its risk such as volatility, skew and maximum drawdown as well as the expected return to support your conclusion.
# In[101]:


# 2.6 
print('Answer:')
volatility = np.std(spread)
print(f'The Volatility is {volatility:.4}%')
skew = scp.stats.skew(spread)
print('Answer:')
print(f'The Skew is {skew[0]:.4}')


# In[97]:


def maxdrawdown(x): 
    i = np.argmax(np.maximum.accumulate(x) - x) # end of period
    j = np.argmax(x[:i]) # start of the period
    percent_change = 100*( x[i] - x[j] ) / x[j]
    return percent_change
cumulativeret = np.cumprod(1+spread/100)
MaxDD = maxdrawdown(cumulativeret)
print('Answer:')
print(f'The Maximum Drawdown is {MaxDD:.4}%')


# In[82]:


ER = np.mean(spread)
print('Answer:')
print(f'The Expected Return is {ER:.4}%')

7. Regress the WML returns on an intercept and the Mkt-RF, SMB and HML Fama-French risk factors in columns O, P, and Q to obtain the WML portfolio’s alpha (risk-adjusted abnormal return). How large is the alpha and is it statistically significant? Is the alpha coming from the Winner or Loser portfolio returns or both?
# In[83]:


# 2.7: Alpha
factors = pd.read_excel(datafile1, usecols = "O:Q", header = 1) #3 factors
Rf = pd.read_excel(datafile1, usecols = "R", header = 0) #risk free rate
factors = factors[burn:] #3 factor


# In[84]:


# spread
mod = sm.OLS(wl_ret['spread'],sm.add_constant( factors))
res = mod.fit()
res.summary()
print('Answer:')
print(f'Alpha of wml is { res.params.const:.4}' )
print(f'P-value is {res.pvalues[0]}')


# In[85]:


# winner
mod = sm.OLS(wl_ret.iloc[:,0],sm.add_constant( factors)) 
res = mod.fit()
res.summary()
print('Answer:')
print(f'Alpha of winner is { res.params.const:.4}' )
print(f'P-value is {res.pvalues[0]}')


# In[86]:


# loser
mod = mod = sm.OLS(wl_ret.iloc[:,1],sm.add_constant( factors)) 
res = mod.fit()
res.summary()
print('Answer:')
print(f'Alpha of loser is { res.params.const:.4}' )
print(f'P-value is {res.pvalues[0]}')


# # PART 3

# # 3.1

# Climate investing supports technologies or companies that are likely to become important as the world transitions away from fossil fuels and carbon-intensive industries. Many funds and companies are investing in alternative energies, such as solar and wind power, that can replace fossil fuels. Another possible investment route is green initiatives, such as carbon offsets or electrical vehicles. According to data from Allied Market Research. the global renewable energy market was worth 881.7𝑏𝑖𝑙𝑙𝑖𝑜𝑛𝑖𝑛2020,𝑎𝑛𝑑𝑖𝑡′𝑠𝑒𝑥𝑝𝑒𝑐𝑡𝑒𝑑𝑡𝑜𝑟𝑒𝑎𝑐ℎ𝑎𝑙𝑚𝑜𝑠𝑡 2 trillion by 2030. The research also mentions that all renewable energy sources combined currently provide 7% of global energy demand. Companies with green energy initiatives can also be a great place to invest in for a climate change-targeted portfolio. These are companies with strong investments in carbon offsets, sustainable materials, meat substitutes, electric vehicles, or other low-carbon alternatives to existing technologies. A number of “green” financial indexes have existed for many years. These indexes fall into two broad groups: pure-play indexes that focus on renewable energy, clean technology, and/or environmental services. Thus far, the success of pure-play indexes has been limited
# 
# Hedging using weather derivatives
# 
# • Weather derivatives work like insurance, paying out contract holders if weather events occur or if losses are incurred due to certain weather-related events.
# 
# • Agriculture, tourism and travel, and energy are just a few of the sectors that utilize weather derivatives to mitigate the risks of weather.

# # 3.2

# Two assets that were strongly positively or negatively affected by news related to global warming:
# 
# (1)Insurance Companies: Hurricane Sandy in 2012 resulted in more than  70𝑏𝑖𝑙𝑙𝑖𝑜𝑛𝑖𝑛𝑒𝑐𝑜𝑛𝑜𝑚𝑖𝑐𝑙𝑜𝑠𝑠𝑒𝑠,𝑚𝑜𝑠𝑡𝑜𝑓𝑤ℎ𝑖𝑐ℎ𝑤𝑎𝑠𝑓𝑙𝑜𝑜𝑑−𝑟𝑒𝑙𝑎𝑡𝑒𝑑.𝐴𝑏𝑜𝑢𝑡 26 billion of that was insured.
# The nature of flood insurance is that because floods only impact certain low-lying areas, owners of homes above sea level don’t buy it; because there are fewer purchasers, flood insurance is significantly more expensive. It is not included in most insurance plans.
# 
# Rising sea levels and the potential for increased incidences of catastrophic flooding will likely drive up both premiums and payouts, putting a strain on the insurance industry. The National Flood Insurance Program, for example, collects about  3.6𝑏𝑖𝑙𝑙𝑖𝑜𝑛𝑖𝑛𝑝𝑟𝑒𝑚𝑖𝑢𝑚𝑠𝑒𝑣𝑒𝑟𝑦𝑦𝑒𝑎𝑟.𝑌𝑒𝑡𝑖𝑡𝑖𝑛𝑠𝑢𝑟𝑒𝑠𝑚𝑜𝑟𝑒𝑡ℎ𝑎𝑛 1.25 trillion in total assets.
# 
# Moreover, many insurance companies are starting to withdraw altogether from providing insurance to certain catastrophic markets, reveals a report released today by Ceres. The report states that last year just under a third of the $116 billion in worldwide losses from weather-related disasters were covered by insurance, according to data from the reinsurer Swiss Re. After Katrina struck, in 2005, insurance picked up 45 percent of the bill.
# 
# With more than 6.5 million homes in the U.S. at risk of storm damage and a total reconstruction value of nearly 1.5 trillion, according to Corelogic, a global property info and analytics provider, the insurance industry’s retreat is an alarming threat to local populations and public institutions.
# 
# (2)Beverage Industry Increased water shortages are among the biggest threats to the worldwide soft drink and bottled-water market, valued at $247 billion by an IBIS World report.
# Coca-Cola ’s 2013 10-K form stated that “changing weather patterns, along with the increased frequency or duration of extreme weather conditions, could impact the availability or increase the cost of key raw materials that the company uses to produce its products.”
# 
# Coca-Cola’s supply of sugarcane, sugar beets and other ingredients looks to be threatened as temperatures rise and extreme weather events occur more often. This past June, government authorities in India forced the closing of a Coca-Cola bottling plant in the country’s north after the company was accused of extracting too much groundwater.
