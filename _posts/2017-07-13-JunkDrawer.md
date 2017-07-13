---
layout: post
title: The Junk Drawer 
---
### A Junk Drawer of Probability ### 
This post will cover a couple important topics in probability theory. These are formulas I should always know, but rarely completely remember. I hope after covering these they'll stick a bit more! 

#### Covariance and Correlation #### 

Covariance measures how linearly related two random variables are:  
$$cov[X,Y] = \mathbb{E}[(X-\mathbb{E}X)*(Y-\mathbb{E}Y)]$$  
Correlation is a normalized covariance:  
$$corr[X,Y] = \frac{cov[X,Y]}{\sqrt{\mathrm{var}X*\mathrm{var}Y}}$$  
This is helpful because correlation ranges between -1 and 1. With a one number, you can reasonably tell how related two linear variables are. 

#### Transforming Random Variables #### 

Assume you have a random variable $X$ that is a n dimensional. Let $f$ be a function $\mathbb{R}^n\rightarrow\mathbb{R}^n$ and $Y=f(x)$. Then we want to know how space where $X$ lives changes when we apply $f$. We can understand this by using the Jacobian of the inverse transformation $x=f^{-1}(y)$. Then:  
$$p_y(y) = p_x(x)|\mathrm{det} J_{y\rightarrow x}|$$  
In this way we have a probability distribution of $Y$. 

#### Monte Carlo Approximation #### 

Sometimes transforming random variables is very difficult. One alternative is to use Monte Carlo approximation. It involves taking a sample $S$ from the probability distribution $p_x(x$ and then applying our function $f$ to each element of the sample. Then we can estimate the mean of $Y=f(X)$ by looking at $\frac{1}{S}\sum_S f(x_i)$. We can estimate other statistics in this way as well. 

#### Entropy #### 
To measure the entropy of a discrete distribution $p_x$ with $K$ states we use the following formula:  
$$\mathbb{H}(X) = -\sum_{k=1}^K p(X=k)\log_{2}p(X=k)$$

#### KL Divergence #### 
Kullback-Leibler Divergence measues the relative entropy of two distributions $p$ and $q$. We define:  
$$\mathbb{KL}(p||q) = \sum_{k=1}^K p_k \log\frac{p_k}{q_k}$$  

#### Mutual Information #### 
An extension of KL Divergence is mutual information:  
$$\mathbb{I}(X;Y) = \mathbb{KL}(p(X,Y)||p(X)p(Y)) = \sum_x\sum_y p(x,y) \log\frac{p(X,Y)}{p(X)p(Y)}$$

### Closing the Drawer ###

These few formulas close out Chapter 2 of MLPP. In future posts, we will see how to apply them. Next MLPP post will cover Chapter 3: Generative models for discrete data. I think my next blog post will ocver CLRS Chapter 5: probabilistic analysis of algorithms 



```python

```
