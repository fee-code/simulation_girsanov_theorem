import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
T = 1.0             
tau = 0.01          
N = int(T / tau)    
M = 5      # Number of paths         

mu = 2           
sigma = 0.4        
S0 = 100            

# Initialize
W = np.zeros((M, N + 1))
S_P = np.zeros((M, N + 1))
time_grid = np.linspace(0, T, N + 1)

# Simulate Wiener paths
for m in range(M):
    Z = np.random.normal(0, 1, N)
    increments = np.sqrt(tau) * Z
    W[m, 1:] = np.cumsum(increments)

# Simulate Bachelier paths using W
for m in range(M):
    S_P[m] = S0 + mu * time_grid + sigma * W[m]

# Shift the drift as a constant to apply Girsanov theorem
alpha = - mu / sigma 
print(alpha)

#Shift the Brownian motion
W_Q = W - alpha * time_grid  # Apply Girsanov shift

# Plot: Brownian Motion and shifted under P
plt.figure(figsize=(10, 5))
for m in range(M):
    plt.plot(time_grid, W[m], '--', label='Path of a Brownian Motion $(W_t)$' if m == 0 else "")
    plt.plot(time_grid, W_Q[m], '-', label='Path of the Shifted Brownian Motion $(W_t^\\mathbb{Q})$' if m == 0 else "")
plt.title("Paths of a Brownian Motion and Shifted Brownian Motion under $\mathbb{P}$")
plt.xlabel("Time")
plt.ylabel("W(t)")
plt.grid(True)
plt.legend()
plt.show()

# Simulate Brownian motion under Q (no drift)
W_QQ = np.zeros((M, N + 1))
for m in range(M):
    dW_Q = np.sqrt(tau) * np.random.randn(N)
    W_QQ[m, 1:] = np.cumsum(dW_Q)

# Simulate Bachelier process under Q, that is now a (relaxed) martingale
S_Q = S0 + sigma * W_QQ

# Plot Bachelier process under P and (relaxed) Martingale under Q
plt.figure(figsize=(10, 5))
for m in range(M):
    plt.plot(time_grid, S_P[m],'--' , label='Bachelier path $(S_t)$ under $\mathbb{Q}$' if m == 0 else "")
    plt.plot(time_grid, S_Q[m], '-', label='(relaxed) Martingale $(S_t)$ under $\mathbb{Q}$' if m == 0 else "")
plt.title("Bachelier Process Paths under $\mathbb{P}$ (with drift) and $\mathbb{Q}$ (no drift)")
plt.xlabel("Time")
plt.ylabel("S(t)")
plt.grid(True)
plt.legend()
plt.show()
