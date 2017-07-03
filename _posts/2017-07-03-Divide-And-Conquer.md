---
layout: post
title: Divide and Conquer
---
Welcome back, let's dive in to CLRS Chapter 4: Divide and Conquer. As always, check out the original source for full details. 

### Motivation ###

Algorithms can be complicated, but by breaking them down into smaller pieces, we can start to get a handle of what's going on. We will describe algorithms in terms of recurrences, $T(n)$, where $n$ is the input size. Here are some examples of recurrences:  
$$T(n) = 2T(\frac{n}{2})+\Theta(n)$$  
$$T(n) = T(\frac{2n}{3})+T(\frac{n}{3})+\Theta(n^2)$$
$$T(n) = \Theta(1)$$

The first example describes a recurrence that takes an input of size $n$, performs a linear operation on it, and the recurrs--it does the same process, but now on two inputs half as large. The second example describes a recurrence that takes an input of size $n$, performs quadratic opeartion on it, but then splits its input up a little differently. Finally, the last example is a constant time algorithm.  

There are three main ways we will solve this recurrences, which allows us to describe the Big O time complexity of our algorithm. See here if you a need a quick refresher on that.  

The **substitution method** is a guess and check way of solving recurrences. Once we do enough of these problems, you start to get a sense of what recurrence $T$ corresponds to which Big $\mathcal{O}$ running time.  

The **recursion tree method** visually expands recurrences into trees and then we can analyze the cost of each layer in the tree and the number of layers. 

The **master method** bounds recurrences in the form $T(n) = aT(n/b)+f(n)$. Pretty handy. 

### The maximum sub array problem ### 

It would be great if we could know the future. What would you do? Some people would go straight for the lottery get rich quick. But it might be too suspicious if you win several lotteries in a short period of time. Others might bet on big sports games, but those are relatively infrequent. Myself, I would go for the stock market. If I had the prices of a stock with the ticker BENK, how could I maximmize my profit? There's one restriction though: to avoid getting caught, I will only buy once and only sell once.  

The naive approach is the look at every combination of buy and sell dates to see what returns the maximum profit. Given $n$ dates, this approach takes ${n \choose 2}$ comparisons, which means this algorithm of maximizing profit is $\mathcal{O}(n^2)$.

What if we look at an array of successive differences? Consider an array of $n-1$ elements that represent the net change day to day. If we naively look at the sum of a subarray, which represents our profit, then we still have to consider $\mathcal{O}(n^2)$ subarrays, since we have 1 choice each for the start and end date and n-1 possibilities. 

#### Dividing #### 

Taking a step back, we the maximum subarray sum $A[i,...,j]$ of $A[low,...,high]$ must be in one of three locations in an array: 
$$A[low,...,mid]$$
$$A[mid,...,high]$$
$$A[i,...,mid,...,j]$$
The last scenario involves $low\leq{i}\leq{mid}\leq{j}\leq{high}$. We also note that the first two cases are recursive subproblems of the original subarray problem. Therefore, once we find a solutions to find the maximum subarray of that crosses the midpoint, we can combine the recursive solutions of the first two cases with the crossing case and have the maximum subarray solution. 

#### Conquering #### 

We can see that solution $A[i,...,mid,...,j]$ must be the sum of the maximum subarrays $A[i,...,mid]$ and $A[mid,...,j]$. Therefore we can find $A_\mathrm{crossing}$ in linear time:


```python
import math 
def find_A_crossing(A, low, mid, high):
    left_sum = -1.*math.inf 
    temp_sum = 0 
    for i in range(mid, low-1, -1):
        temp_sum = temp_sum + A[i]
        if temp_sum > left_sum: 
            left_sum = temp_sum 
            max_left = i
    right_sum = -1.*math.inf 
    temp_sum = 0 
    for j in range(mid,high+1):
        temp_sum = temp_sum + A[j]
        if temp_sum > right_sum: 
            right_sum = temp_sum 
            max_right = j
    total_sum = left_sum + right_sum 
    return max_left, max_right, total_sum 
```

We can combine our function above with some recursion and arrive at a solution for the maxmium subarray problem:


```python
import math 
def find_A_max(A, low, high):
    #for empty or 1-element A 
    if len(A)<2:
        return low, high, A 
    mid = math.floor((low+high)/2.)
    left_low, left_high, left_sum = find_A_max(A, low, mid)
    right_low, right_high, right_sum = find_A_max(A, mid, high)
    cross_low, cross_high, cross_sum = find_A_crossing(A, low, mid, high)
    
    max_sum = max(left_sum,max(right_sum,cross_sum))
    
    if left_sum == max_sum:
        return left_low, left_high, left_sum
    if right_sum == max_sum: 
        return right_low, right_high, right_sum
    if cross_sum == max_sum:
        return cross_low, cross_high, cross_sum
```

We can see this algorithm is $T(n) = T(n/2) + T(n/2) + \Theta(n)$. One $T(n/2)$ comes from solving the left half and right half subproblems, respectively. The $\Theta(n)$ comes from solving the crossing case in linear time. Therefore we can write that this algorithm is $T(n) = 2T(n/2) + \Theta(n)$. But what does that mean for the Big $\mathcal{O}$ running time for this algorithm? 

### The Master Method ### 

The master method bounds recurrences in the form $T(n) = aT(n/b)+f(n)$. This is directly applicable to our maximize subarray recurrence. There are three cases of the master method:  
Case 1: $f(n) \in O(n^{log_b{a}-\epsilon})$  
Then $T(n)$ is $\Theta(n^{log_b{a}})$  

Case 2: $f(n) \in \Theta(n^{log_b{a}})$  
Then $T(n)$ is $\Theta(n^{log_b{a}}lg n)$ 

Case 3: $f(n) \in \Omega(n^{log_b{a}+\epsilon})$  
Then $T(n)$ is $\Theta(f(n))$ provided $af(\frac{n}{b})\leq{cf(n)}$ for sufficiently large n and $c<1$. 

Let's apply this to our maximum subarray problem. $T(n) = 2T(n/2) + \Theta(n)$. Then $a=2$ and $b=2$ so Case 2 applies as $log_2{2}=1$. Therefore our solution to the maximum subarray problem is $\Theta(nlgn)$.

### Next Posts ###
Next time I cover CLRS, I'll be going over probabilistic analysis of algorithms. A cool intersection of statistics and computer science.  
The next post will continue my coverage of chapter 2 of MLPP. Looking forward to some Information Theory! 

