import numpy as np
import numpy.random as npr
from scipy.stats import norm
import tensorflow.keras as keras
import tensorflow.keras.backend as kb
import matplotlib.pyplot as plt

# Black Scholes pricing and delta functions
def BlackScholes(S0, r, sigma, T, K):
    d1 = 1 / (sigma * np.sqrt(T)) * (np.log(S0/K) + (r + sigma**2/2) * T)
    d2 = d1 - sigma * np.sqrt(T)
    return norm.cdf(d1)*S0 - norm.cdf(d2)*K*np.exp(-r*T)

def BlackScholesCallDelta(S0, r, sigma, T, K):
    d1 = 1 / (sigma * np.sqrt(T)) * (np.log(S0/K) + (r + sigma**2/2) * T)
    return norm.cdf(d1)

# Simulation parameters
T = 100         # number of time steps
S_0 = 1        # initial asset price
N = 10000       # number of simulation paths
time_horizon = 1.0   # total time

# Generate simulation-specific parameters: drift, volatility and strike
mu_vec = npr.uniform(0.05, 0.15, size=(N,1))       # drift per simulation
sigma_vec = npr.uniform(0.1, 0.5, size=(N,1))        # volatility per simulation
K_vec = npr.uniform(0.8, 1.2, size=(N,1))            # strike per simulation

# Generate Brownian increments and paths
xi = npr.normal(0, np.sqrt(1/T), (N, T))
W = np.apply_along_axis(np.cumsum, 1, xi)
W = np.concatenate((np.zeros((N,1)), W), axis=1)

# Create time grid and compute drift per simulation (linear from 0 to mu)
tim = np.linspace(0, time_horizon, T+1)  # shape (T+1,)
drift = mu_vec * tim  # broadcasts to (N, T+1)

# Simulate asset paths: S = S_0 * exp(drift + sigma*W)
S = S_0 * np.exp(drift + sigma_vec * W)
dS = np.diff(S, axis=1)  # increments; shape (N, T)

# Build training data inputs X:
# For each time step the input is [time, S_t, K]. (Note K is constant along each path.)
X = []
for i in range(T):
    timv = np.full((N, 1), tim[i])
    Sv = S[:, i].reshape(N, 1)
    X.append(np.concatenate((timv, Sv, K_vec), axis=1))

# Build the deep hedging network; input shape is now (3,)
inputs = []
predictions = []
layer1 = keras.layers.Dense(100, activation='relu')
layer2 = keras.layers.Dense(100, activation='relu')
layer3 = keras.layers.Dense(100, activation='relu')
layer4 = keras.layers.Dense(1, activation='sigmoid')
for i in range(T):
    sinput = keras.layers.Input(shape=(3,))
    x = layer1(sinput)
    x = layer2(x)
    x = layer3(x)
    spred = layer4(x)
    inputs.append(sinput)
    predictions.append(spred)
predictions = keras.layers.Concatenate(axis=-1)(predictions)
model = keras.models.Model(inputs=inputs, outputs=predictions)

# Set risk free rate and compute the option price per simulation under risk neutral pricing.
r = 0.01   # risk free rate
callprice = BlackScholes(S_0, r, sigma_vec, time_horizon, K_vec)  # shape (N,1)
callprice = callprice.flatten()  # shape (N,)

# Create training targets Y by concatenating:
#   - dS (shape (N, T))
#   - callprice (shape (N, 1)) repeated once per simulation
#   - strike K (shape (N, 1))
Y = np.concatenate((dS, callprice[:, None], K_vec), axis=1)  # shape (N, T+2)

# Custom loss function which extracts dS, callprice and strike from y_true.
def loss_call(y_true, y_pred):
    # y_true shape: (batch_size, T+2)
    # y_pred shape: (batch_size, T)
    dS_batch = y_true[:, :T]             # simulated increments
    callprice_batch = y_true[:, T]         # call price for each simulation
    K_batch = y_true[:, T+1]               # strike for each simulation
    hedge_sum = kb.sum(y_pred * dS_batch, axis=-1)
    S_T = S_0 + kb.sum(dS_batch, axis=-1)
    payoff = kb.maximum(S_T - K_batch, 0)
    error = callprice_batch + hedge_sum - payoff
    return kb.square(error)

epochs = 4
model.compile(optimizer='adam', loss=loss_call)
model.fit(X, Y, batch_size=100, epochs=epochs)

# Testing: Generate test data for a fixed time, strike and volatility.
# Instead of plotting hedge ratios, we now plot predicted payoff vs. actual option payoff.
t_test = 0.7
K_test = 1.0      # fixed test strike
sigma_test = 0.3    # fixed test volatility for pricing, delta computation remains unused here
dt = time_horizon - t_test  # remaining time to maturity

S_test_values = np.linspace(20, 100, T)  # range of asset prices at test time

# For each S_test value, create an input of shape (1,3): [t_test, S_test, K_test]
# and predict the hedge ratio.
X_test = []
for S_val in S_test_values:
    X_test.append(np.array([[t_test, S_val, K_test]]))
hedge_pred = model.predict(X_test)  # predicted hedge ratios; array shape (T, 1)
hedge_pred = hedge_pred.flatten()

# Compute the risk-neutral terminal asset price assuming S_T = S_test * exp(r * dt).
S_T_test = S_test_values * np.exp(r * dt)

# For each S_test value, compute the call option price using Black Scholes 
# for a remaining time dt with parameters: underlying S_test, fixed sigma_test and strike K_test.
callprice_test = np.array([BlackScholes(S, r, sigma_test, dt, K_test) for S in S_test_values])

# Predicted payoff (price + pnl) is: callprice + hedge*(S_T - S_test)
predicted_payoff = callprice_test + hedge_pred * (S_T_test - S_test_values)

# Actual option payoff: max(S_T - K_test, 0)
actual_payoff = np.maximum(S_T_test - K_test, 0)

# Plotting the results.
# plt.plot(S_test_values, predicted_payoff, label="Predicted payoff (price + pnl)")
# plt.plot(S_test_values, actual_payoff, "b--", label="Actual option payoff")
# plt.xlabel("S_t (spot price at time t_test)")
# plt.ylabel("Payoff")
# plt.title("Comparison of Predicted vs Actual Option Payoff")
# plt.legend()
# plt.show()

# ---------------------------------------------------------------
# Testing: Generate test data for a fixed time, strike and volatility.
# Here we compare the learned hedge ratio with the Black Scholes delta.
t_test = 0.7
K_test = 1.0     # fixed test strike
sigma_test = 0.3 # fixed test volatility for delta computation
S_test_values = np.linspace(0, 2, T)  # range of asset prices for testing

# For each S_test value, create an input of shape (1,3): [t_test, S_test, K_test]
X_test = []
for S_val in S_test_values:
    X_test.append(np.array([[t_test, S_val, K_test]]))

# Predict the learned hedge ratios.
Delta_learn = model.predict(X_test)
Delta_learn = Delta_learn.flatten()

# Compute Black Scholes delta for the test data (time to maturity is time_horizon - t_test).
Delta_BS = BlackScholesCallDelta(S_test_values, r, sigma_test, time_horizon - t_test, K_test)

plt.plot(S_test_values, Delta_learn, label="Learned hedge ratio")
plt.plot(S_test_values, Delta_BS, "b--", label="Black Scholes delta")
plt.xlabel("S_t (spot price)")
plt.ylabel("Hedge ratio")
plt.title("t = %1.2f, K = %1.2f" % (t_test, K_test))
plt.legend()
plt.show()