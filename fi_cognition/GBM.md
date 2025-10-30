## Geometric Brownian motion

A geometric Brownian motion (GBM) (also known as exponential Brownian motion) is a continuous-time stochastic process in which
the logarithm of the randomly varying quantity follows a Brownian motion (also called a Wiener process) with drift.
It is an important example of stochastic processes satisfying a stochastic differential equation (SDE); 
in particular, it is used in mathematical finance to model stock prices in the Black–Scholes model. 

### equation
A stochastic process St is said to follow a GBM if it satisfies the following stochastic differential equation (SDE):

$$dS_t = \mu S_tdt + \sigma S_t d W_t$$

where $W_t$ is a Wiener process or Brownian motion, and $\mu$ ('the percentage drift') and $\sigma$ ('the percentage volatility') are constants.

The former parameter is used to model deterministic trends, while the latter parameter models unpredictable events occurring during the motion.

### solution
For an arbitrary initial value $S_0$ the above SDE has the analytic solution 

$$ S_t = S_0 \exp ((\mu-\frac{\sigma^2}{2})t + \sigma W_t)$$

more details see [GBM](https://www.mi.uni-koeln.de/wp-znikolic/wp-content/uploads/2017/05/4_Geometric_Brownian_Motion_28042017.pdf)
