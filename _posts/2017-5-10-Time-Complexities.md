---
layout: post
title: Time Complexities
---
### Welcome to my first real post ###
As stated in my introduction, I'll be covering CLRS and MLPP (not a standard acronym, but it will be for this blog!)

Let's get started with CLRS Chapter 3: Growth of Functions. 

We define $$\mathcal{\Theta}(g(n))=\{f(n):\exists c_1, c_2, n_0\geq 0, 0\leq {c}_{1}g(n)\leq f(n) \leq c_{2}g(n) \forall n\geq{n_0}\}$$

This means that \\(\mathcal{\Theta}\\) is the _set_ of functions that are bounded by scaled function \\(g(n)\\) above and below as \\(n\rightarrow \infty\\). Take as an example the function \\(2x^2\\) show in blue:


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-10, 10, 500, endpoint=True)
x2 = np.square(X)

plt.plot(X,2*x2)
plt.plot(X,4*x2)
plt.plot(X,.5*x2)
plt.show()
```


![](Images/TimeComplexities_1_0.png?raw=true)


We have bounded the blue function, \\(2x^2\\) with the function \\(g(x)=x^2\\) and constants \\(c_1=.5, c_2=4\\). That means \\(2x^2\in\mathcal(\Theta)(x^2)\\). Most people abused the notation here and just write \\(2x^2=\mathcal(\Theta)(x^2)\\). We now have an **asymptotically tight bound** for \\(2x^2\\).

We can define other types of bounds on functions: 

**Asymptotic upper bounds** 
$$O(g(n)) = \{f(n) : \exists c,n_0\geq{0}, 0{\leq}f(n)\leq{cg(n)} \forall n\geq n_0\}$$

**Asymptotic lower bounds**
$$\Omega(g(n)) = \{f(n) : \exists c,n_0\geq{0}, 0{\leq}cg(n)\leq f(n) \forall n\geq n_0\}$$

We can quickly see that if $f(n) \in O(g(n))$ and $f(n)\in\Omega(g(n)$ then $f(n)\in\Theta(g(n))$.

There are even tighter definitions available: 
$$o(g(n)) = \{f(n) : \forall c\geq{0},\exists n_0\geq{0}, 0{\leq}f(n)<{cg(n)} \forall n\geq n_0\}$$

$$\omega(g(n)) = \{f(n) : \forall c\geq{0},\exists n_0\geq{0}, 0{\leq}cg(n)< f(n) \forall n\geq n_0\}$$

There are two subtles differences here. Note that we are saying for every constant we can find an n. Also, in the last inequality, instead of being less than or equal to, we impose a stricter bound. 

I'll use this notation throughout this blog. This is just a quick introduction to time complexities. Next time I cover CLRS, we'll see how to actually derive time complexities for algorithms.


```python

```
