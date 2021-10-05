import numpy as np 
import matplotlib.pyplot as plt 

T = 5  # Time horizon
N = 5000  # 100, 10000

def f(x):
    return x

def brownian_mot(N,T):  # Function to simulate a Brownian motion
    mu, sigma = 0, np.sqrt(1/N) # Mean and standard deviation 
    s = np.random.normal(mu, sigma, N*T) # N*T normal distributed random numbers 
    B = np.zeros(N*T+1)  # Vector creation
    for i in range(N*T):
        B[i+1]=B[i]+s[i] # Increments of BM on each time step
    return B   # Return the Brownian motion
B = brownian_mot(N,T) # Brownian motion realization: in this exercise, we need the same realization of B for all the calculations

def stoch_alphaInt(a,f,N,T):
    Y = np.zeros(N*T+1) 
    for i in range(N*T):
        Y[i+1] = Y[i] + ( f(B[i]) + a*(f(B[i+1])-f(B[i])) ) * (B[i+1]-B[i])  # Stochastic integral approximation formula
    return Y

def det_alphaInt(a,f,N,T):
    Y = np.zeros(N*T+1) 
    for i in range(N*T):
        Y[i+1] = Y[i] + ( f(B[i]) + a*(f(B[i+1])-f(B[i])) ) / N
    return Y 

Y_calc1 = np.zeros(N*T+1) 
Y_calc2 = np.zeros(N*T+1) 
Y_calc3 = np.zeros(N*T+1) 
for i in range(N*T):
    Y_calc1[i+1] = (B[i+1]**2 - (i+1)/N) / 2
    Y_calc2[i+1] = (B[i+1]**2) / 2
    Y_calc3[i+1] = (B[i+1]**2 + (i+1)/N) / 2

x = np.linspace(0, T, N*T+1)   # X-axis discretization
plt.plot(x,det_alphaInt(0,f,N,T),label=r'$\alpha = 0$')
plt.plot(x,det_alphaInt(0.5,f,N,T),label=r'$\alpha = \frac{1}{2}$')
plt.plot(x,det_alphaInt(1,f,N,T),label=r'$\alpha = 1$')
plt.title("Visualization that "+ r'$\int_0^t B_s d^0 s = \int_0^t B_s d^{\frac{1}{2}} s = \int_0^t B_s d^1 s$')
plt.xlabel('t') # Labeling x-axis
plt.legend(loc='best') # Legend at the position where Python finds it the best
plt.show()

x = np.linspace(0, T, N*T+1)   # X-axis discretization
plt.plot(x,stoch_alphaInt(0,f,N,T),label=r'$\alpha = 0$')
plt.plot(x,Y_calc1,label=r'$\frac{1}{2}(B_t^2-t)$')
plt.plot(x,stoch_alphaInt(0.5,f,N,T),label=r'$\alpha = \frac{1}{2}$')
plt.plot(x,Y_calc2,label=r'$\frac{1}{2}B_t^2$')
plt.plot(x,stoch_alphaInt(1,f,N,T),label=r'$\alpha = 1$')
plt.plot(x,Y_calc3,label=r'$\frac{1}{2}(B_t^2+t)$')
plt.title("Visualization that "+ r'$\int_0^t B_s d^0 s \neq \int_0^t B_s d^{\frac{1}{2}} s \neq \int_0^t B_s d^1 s$')
plt.xlabel('t') # Labeling x-axis
plt.legend(loc='best') # Legend at the position where Python finds it the best
plt.show()
