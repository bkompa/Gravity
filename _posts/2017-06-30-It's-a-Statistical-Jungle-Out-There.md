---
layout: post
title: It's a (Statistical) Jungle Out There 
---
Whew, I hope you had enough time to digest my last blog post there. It's only been 6 weeks since my original post. Since then, I've wrangled MathJax to play nicely with Github blogs, so you should be able to enjoy the LaTeX the way it was meant to be seen. 

Anyways, enough about me and sorry for the delay. Today, I'll be covering chapter 2 of MLPP (Machine Learning: A Probabilistic Perspective by Kevin Murphy for those who aren't hip to my lingo). It's a basic overview of some statistical concepts and the myriad of statistical distributions that are common in machine learning. 

# Introduciton: Basic Review of Probability Concepts #
When we talk about a probability $p(A)$ we are considering the chance that the event $A$ happens. Quickly we can see that $0\leq p(A)\leq{1}$. $p(A)=1$ means that event $A$ will always happen, while $p(A)=0$ means event $A$ will never happen. Most events will lay somewhere in between.

A few more basic rules: <br/>
**Rule 1** <br/>
$p(A\cup{B}) = p(A) + p(B) - p(A\cap{B})$
<br/>
This states that the probability of A or ($\cup$) B is equal to the proabilities of $A$ and $B$ minus the probability that both $A$ and($\cap$) $B$ happen. This makes sense -- we don't want to double count an event in the space $A$ or double count an event in space $B$ because it's in both spaces. 

**Rule 2** <br/>
$p(A,B) = p(A|B)p(B)$ <br/>
This states that the joint probability of A and B is equal to the probability that A occurs, given($|$) that B occurs times the probability that $B$ happens. 
We can rearrange this rule and see $p(A|B) = \frac{p(A,B)}{p(B)}$. 

**Rule 3** <br/> 

Baye's rule is perhaps the most important relation in statistics. 
Baye's rule (or theorem) states: <br/>
$$p(X=x|Y=y) = \frac{p(Y=y|X=x)p(X=x)}{\sum_{x'}p(X=x')p(Y=y|X=x')} $$ <br/>

### Means, Moments, and Variances, Oh My! ### 

Before we get into the jungle of statistical distributions that exist, I'm going to take a detour and explain how to calculate _moments_ of a distribution. The $n^{th}$ moment of a distribution is defined as: $$ \int x^np(x)dx $$ Moments are important because, among other things, the first and second moments are the mean and variance. Therefore the mean of a distribution is: $$ \int xp(x)dx $$ and the variance is $$ \int x^2p(x)dx $$. Naturally, we have the standard deviation as the square root of the variance. 

# It's a Jungle Out There # 

Now I'll start our parade through the jungle of statistical distributions. There are so many out there, but a few keep popping up over and over again. We will start with discrete distributions and work our way up to continuous distributions. 

### Binomial and Bernoulli Distributions ### 

Imagine a coin that lands heads with a probability $\theta$.Let $X$ be the random variable representing the flip. Then we can use the Bernoulli distribution to describe this single coin flip: <br/>$$\mathrm{Ber}(x|\theta)= \theta^{x==1}(1-\theta)^{x==0}$$<br/>
<br/> If we now let $X$ be the number of heads in $N$ trials then $X\in\{0,1,...,N\}$ and we use a Binomial distribution to characterize the probabilities that $X$ takes on any particular value. The probability mass function of the Binomial distribution is: <br/>$$\mathrm{Bin}(X|n,\theta) = {n\choose x}{\theta}^x(1-\theta)^{n-x}$$<br/>
The mean of the distribution is $n\theta$ and the variance is $n\theta(1-\theta)$. Below you can see the probability mass function for a binomial distribution with $n=10$ and $\theta=.5$.



```python
%matplotlib inline
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

n, theta = 10, .5 

x = np.arange(binom.ppf(.01, n, theta), binom.ppf(1.00, n, theta))
ax.plot(x, binom.pmf(x, n, theta), 'bo', ms=8, label='binom pmf')
ax.vlines(x, 0, binom.pmf(x, n, theta), colors='b', lw=5, alpha=0.5)
plt.show()
```


![png](https://github.com/bkompa/bkompa.github.io/raw/master/images/StatisticalJungle_3_0.png)


### Multinomial and Multinoulli distributions ### 

Next, we can generalize the binomial and Bernoulli distributions when there is a more than 2 choices are what values $X_i$ can take. A classic exampmle of this is dice, which take 6 values. Assume $X$ is a vector with $K$ elements representing the total number of times the $k^{th}$ choice was hit. We write the multinomial pmf as: <br/>
$$ \mathrm{Mu}(x|n,\theta) = {n \choose {x_1,...,x_k}} \Pi^{K}_{j=1} {\theta}^{x_j}_{j} $$ <br/>where $$ {n \choose {x_1,...,x_k}} = \frac{n!}{\Pi^{K}_{j=1} x_{j}!}$$ In the case $n=1$, this is a multinoulli distribution. 

### Poisson Distribution ### 
The Poisson Distribution is often used to model events which rare events. Let $X$ be the counts of some event so $X\in \{0,1,...,\infty\}$ with a parameter $\lambda>0$. Then the pmf of a Poisson distribution is: <br/>$$\mathrm{Poi}(x=k|\lambda) = \frac{\mathrm{e}^{-\lambda}\lambda^k}{k!} $$ <br/> 

Below you can see a couple pmfs with $\lambda = .1, .5, 5, 10$.


```python
%matplotlib inline
import numpy as np
from scipy.stats import poisson 
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

lam = .1
x = np.arange(poisson.ppf(.01, lam), poisson.ppf(.99, n, lam))
ax.plot(x, poisson.pmf(x, lam), 'bo', ms=3, label='poisson pmf')

lam = .5
x2 = np.arange(poisson.ppf(.01, lam), poisson.ppf(.99, n, lam))
ax.plot(x2, poisson.pmf(x2, lam), 'ro', ms=3, label='poisson pmf')

lam = 5
x3 = np.arange(poisson.ppf(.01, lam), poisson.ppf(.99, n, lam))
ax.plot(x3, poisson.pmf(x3, lam), 'go', ms=3, label='poisson pmf')

lam = 10
x4 = np.arange(poisson.ppf(.01, lam), poisson.ppf(.99, n, lam))
ax.plot(x4, poisson.pmf(x4, lam), 'mo', ms=3, label='poisson pmf')

plt.show()
```


![png](https://github.com/bkompa/bkompa.github.io/raw/master/images/StatisticalJungle_5_0.png)


### The King of the Jungle: The Gaussian Distribution ### 
The Gaussian distribution (a.k.a. the Normal distribution) is quite possibly the most important distribution in all of statistics. It has some nice properties that make it very useful in statistics and machine learning. It also describes a lot of processes in life. Without further ado, here is the pmf in it's full glory: <br/>$$\mathcal{N}(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi{\sigma}^2}} \mathrm{e}^{\frac{-(x-\mu)^2}{2\sigma^2}} $$<br/>
Where $\mu$ is the mean and $\sigma$ is the standard deviation. See below for a classical standard normal: mean of 0, variance of 1


```python
%matplotlib inline
import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
ax.plot(x, norm.pdf(x),'b-', lw=5, alpha=0.5, label='norm pdf')
plt.show()
```



![png](https://github.com/bkompa/bkompa.github.io/raw/master/images/StatisticalJungle_7_0.png)


### Student's t distribution ### 

The Gaussian distribution has at least one problem: it needs to eat a hamburger. That's right, it's too skinny. Specifically, it's tails are too skinny. This means a Gaussian distribution is very sensitive to outliers. To solve this, we use a t distribution. It's pmf is: <br/>$$\mathcal{T}(x|\mu,\sigma^2,\nu)\propto (1+\frac{1}{\nu}(\frac{x-\mu}{\sigma})^2)^{\frac{-\nu-1}{2}}$$ <br/>
It has mean $\mu$ and variance $\frac{\nu\sigma^2}{(\nu-2)}$. $\nu$ is the "degrees of freedom" of the distribution. In common applications of the t distribution this is often 1 less than the sample size. 


```python
%matplotlib inline
import numpy as np
from scipy.stats import t 
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

df = 1
x = np.linspace(t.ppf(0.05,df), t.ppf(0.95,df), 100)
ax.plot(x, t.pdf(x,df),'b-', lw=5, alpha=0.5, label='t pdf')

df2 = 2
x2 = np.linspace(t.ppf(0.01,df2), t.ppf(0.99,df2), 100)
ax.plot(x2, t.pdf(x2,df2),'r-', lw=5, alpha=0.5, label='t pdf')

df3 = 10
x3 = np.linspace(t.ppf(0.001,df3), t.ppf(0.999,df3), 100)
ax.plot(x3, t.pdf(x3,df3),'g-', lw=5, alpha=0.5, label='t pdf')
plt.show()
```


![png](https://github.com/bkompa/bkompa.github.io/raw/master/images/StatisticalJungle_9_0.png)


### Gamma Distribution ### 

For shape and rate parameters $a,b>0$, the gamma distribution is defined as: <br/>$$\mathrm{Gamma}(x|a,b) = \frac{b^a}{\Gamma(a)} x^{a-1}\mathrm{e}^{-bx}$$<br/>
The gamma distribution has a variety of real world application like modeling rainfall or spikes in neuron activity. Several gamma distributions are shown below with varying shape parameters.

The gamma distribution is a generalization of the exponential distribution, which models the time in between events of  Poisson process. Simply let $a=1$ and $b=\lambda$. It is also the generalization of a chi-squared distribution with $\nu$ degree of freedom. Let $a=.5\nu$ and $b=.5$. 


```python
%matplotlib inline
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

a = 1
x = np.linspace(gamma.ppf(0.01,a), gamma.ppf(0.99,a), 100)
ax.plot(x, gamma.pdf(x,a),'b-', lw=5, alpha=0.5, label='gamma pdf')

a2 = 2
x2 = np.linspace(gamma.ppf(0.01,a2), gamma.ppf(0.99,a2), 100)
ax.plot(x2, gamma.pdf(x2,a2),'r-', lw=5, alpha=0.5, label='gamma pdf')

a3 = 10
x3 = np.linspace(gamma.ppf(0.01,a3), gamma.ppf(0.99,a3), 100)
ax.plot(x3, gamma.pdf(x3,a3),'g-', lw=5, alpha=0.5, label='gamma pdf')
plt.show()
```


![png](https://github.com/bkompa/bkompa.github.io/raw/master/images/StatisticalJungle_11_0.png)


### Beta Distribution ### 
The beta distribution's pmf is: <br/>$$\mathrm{Beta}(x|a,b)=\frac{1}{B(a,b)}x^{a-1}(1-x)^{b-1}$$<br/> Where $B(a,b)=\frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$. It is defined on $x\in[0,1]$. The beta distribution is very useful in Bayesian analysis since it's the conjugate prior of the binomial distribution -- but for more on that you'll have to wait for my next MLPP blog post. Below are some beta distributions, a recreation of Figure 2.10 in MLPP 


```python
%matplotlib inline
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

a,b = .1,.1
x = np.linspace(beta.ppf(0.3,a,b), beta.ppf(0.7,a,b), 100)
ax.plot(x, beta.pdf(x,a,b),'b-', lw=5, alpha=0.5, label='beta pdf')


a2,b2 = 1,1
x2 = np.linspace(beta.ppf(0.01,a2,b2), beta.ppf(0.99,a2,b2), 100)
ax.plot(x2, beta.pdf(x2,a2,b2),'r-', lw=5, alpha=0.5, label='beta pdf')

a3,b3 = 2,3
x3 = np.linspace(beta.ppf(0.01,a3,b3), beta.ppf(0.99,a3,b3), 100)
ax.plot(x3, beta.pdf(x3,a3,b3),'g-', lw=5, alpha=0.5, label='t pdf')


a4,b4 = 8,4
x4 = np.linspace(beta.ppf(0.01,a4,b4), beta.ppf(0.99,a4,b4), 100)
ax.plot(x4, beta.pdf(x4,a4,b4),'m-', lw=5, alpha=0.5, label='t pdf')

plt.show()
```


![png](https://github.com/bkompa/bkompa.github.io/raw/master/images/StatisticalJungle_13_0.png)


### Pareto Distribution ### 
The pmf of the Pareto distribution is: <br/>$$\mathrm{Pareto}(x|k,m)=km^kx^{-(k+1)}\mathcal{I}(x\geq m)$$<br/> Where $\mathcal{I}$ is my janky indicator function Latex solution because ipython notebook doesn't have mathbb and $m$ is a minimum value for $x$, and $k$ controls how much $x$ can exceed $m$.

# Next MLPP Post # 
I know I said a little further above I would post about conjugate priors in the next MLPP post but I'm actually going to cut off here with a little bit of Chapter 2 left. Next time I cover MLPP I'll go over:
- Joint Probability Models 
- Covariance and correlation 
- Multivariate distributions 
- Transformations of random variables 
- Monte Carlo Approximation 
- Information theory 
And maybe if I'm feeling spunky I'll do some of the Chapter 2 exercises. 

## Next post: CRLS Chapter 4 ## 
Monday, I'll be covering CRLS Chapter 4: Solving Recurrences. 


