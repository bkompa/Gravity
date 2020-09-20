---
layout: post
title: Conformal Inference Tutorial 
---

Conformal inference is one way to obtain a prediction _region_ from a machine learning algorithm. It allows us to have guaranteed confidence sets for prediction and guaranteed confidence intervals for regression under very mild assumptions. Conformal inference provides error bounds on a per-sample basis. 

Using conformal inference, we can assess: 

*   How good is our classification/regression?
*   For classification, what is $P(\hat{y}=y)$? For regression, what is $\|\hat{y}-y\|$?
*   **Do we trust the model?**

Conformal inference can be used on most any classification or regression algorithm, including offline and online settings. 

This tutorial will be a quick introduction to conformal inference. We generally will follow the notation of [Shafer and Vovk 2008](http://jmlr.csail.mit.edu/papers/volume9/shafer08a/shafer08a.pdf). We additionally consider some of the examples raised in this [introduction to conformal inference](http://www.clrc.rhul.ac.uk/copa2017/presentations/CP_Tutorial_2017.pdf). 
















# Key Conepts

Conformal inference requires a **significance level** $\epsilon$ and a **nonconformity measure** $A$.  We consider our previous examples (i.e. training data) to be an **exchanagble** "bag", denoted $B$. $B$ consists of $\{z_i\}_{i=1}^N$ where this is the set of our previous observations $z_i=(x_i, y_i)$ for features $x_i$ and label/real value $y_i$ in the case of prediction/regression, respectively. 

Conformal inference can apply for most any classification or regression method. We denote such a method as the function $\hat{z}(B)$. This seems to only allow for predictions based on examples already in bag $B$, though. 

We can also consider functions that use the features of the current observation we are trying to classify/regress, i.e. $y_{N+1} = \hat{z}(B, x_{N+1})$. We use $\hat{z}(B, x_i)$ and $\hat{z}(B)$ interchangably unless specifically noted. 

## Exchangibility 
Bag $B$ must be **exchangable**. This means that all the examples $z_i\in B$ are drawn from the same (unknown) distribution $Q$. All i.i.d. variables are exchangable, but the converse is not true. A corollary is that conformal inference would not have guarantees when there is dataset shift. 

One common definition of exchangibility is that for any premutation $\tau$ of the numbers $\{1,...,N\}$, the joint distribution of $P(z_1,...z_N)$ is the same as $P(z_{\tau (1)},...,z_{\tau (N)})$. 

There are also game theoretic notions of exchanability that come up a lot in conformal inference literature. Checkout [Shafer and Vovk 2008](http://jmlr.csail.mit.edu/papers/volume9/shafer08a/shafer08a.pdf) for a great treatment of those. 

## Valid Prediction region

Applying the conformal procedure generates a **valid prediction region**, which is denoted for sample $i$: $$\Gamma_i^\epsilon$$

For prediction, $\Gamma_i^\epsilon$ is a set of labels such that for the true label $y_i$, $P(y_i\in \Gamma_i^\epsilon)=1-\epsilon$. 

For regression, $\Gamma_i^\epsilon$ is a confidence interval such that for true value $y_i$, $P(y_i\in \Gamma_i^\epsilon)=1-\epsilon$.

Valid prediction regions have the property that for $\epsilon_1\geq \epsilon_2$, $\Gamma_i^{\epsilon_1} \subseteq \Gamma_i^{\epsilon_2}$. 

## $1-\epsilon$ guarantees
Typically, we think of guarantees for confidence intervals in the way imagined by R.A. Fischer back around the turn of the 20th century. For example, let us try to estimate the true average weight of babies born before 32 weeks of gestation. We sample the weights of 1000 babies, construct a 95% confidence interval, and if we repeat this sampling and construction of confidence intervals, then the true average weight will be contained in 95% of the confidence intervals. This is an example of "offline" predictions.

In the offline scenario, we can just expect that, on average, $1-\epsilon$ of our predictions are correct. We can say nothing about the _next_ prediction we are about to make. 

Conformal inference gives us more powerful, more intuitive guarantees in online scenarios. For instance, say we are predicting the mortality of NICU patients after 1 week post-birth. Given training data bag $B$, conformal inference can guarantee that $P(y_i\in \Gamma_i^\epsilon)=1-\epsilon$ for the next patient $i$. The next sample predicted on is guaranteed to have a correct prediction $1-\epsilon$ of the time. 

### Why does this hold? 

These guarantees hold because of the [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers). In Shafer and Vovk, they informally describe the law of large numbers for $\epsilon$-rare events: 

Let $N$ be large and bag $B=\{z_i\}$ be exchanagble. Let $E_1,...E_N$ be $\epsilon$-rare events. Then the law of large number applies. 

$E_i$ is the event of a conformal prediction being incorrect. The law of large numbers guarantees at most an $\epsilon$ fraction of our predictions will be incorrect. 

### Is this actually useful? 

The utility of this guarantee depends on the efficiency of our valid prediction region $\Gamma_i^\epsilon$. If, for instance, a confidence interval is very wide or if the set of predicted labels is 0 and 1 in a binary classification problem, this is (not necessarily) very useful. 

In turn, the efficiency of our valid prediction region depends on the underlying distribution $z_i\sim Q$ and the nonconformity measure. 

However, if we think we know $Q$ (i.e. we have a good classifciation/prediction algorithm), then we can choose a nonconformoity measure that provides an efficient valid prediction region when we do indeed know $Q$. 


## Nonconformity measure 
Nonconformity measure $A(B, z)$ should be chosen to measure how different $z$ is from the examples in bag $B$. Most commonly, we just need to define a distance metric $d$ that measures the distance between predictions $\hat{z}(B)$ and a new sample, $z$. Thus, $A(B,z)=d(\hat{z}(B),z)$. 

The conformal algorithm is not affected by monotic changes in $A$. Thus, the metric we choose is not too important. The critical part is choosing a good predictor, $\hat{z}(B)$. 

We generally seek nonconformity measures that give low scores when applied to $z_i=(x_i,y_i)$ and high scores when applied to $z_i=(x_i, \neg y_i)$. When there are correct input-output pairs, there shouldn't be high nonconformity scores. 

Some common choices of $A$ include, the probability estimate for the correct class, the margin of probability estiamte for correct class (i.e. in SVMs), the distance to neighbors with same class, and the absolute error of a regression model. 


# Conformal Prediction 




Here, we'll discuss _inductive_ conformal prediction, which does not require retraining a model for every new data point we wish to classify. Originally, conformal inference was developed for transductive prediction, which requires retraining the algorithm. 

We'll apply conformal prediction the UCL ML handwritten numbers dataset available in scikit-learn. 

The steps of conformal prediction are: 

0. Divide the dataset
1. Train a learner on training data 
2. Calculate nonconformity scores on calibration data 
3. Determine valid prediction regions for new data points 




## Divide the Data Set 

For conformal prediction to work, we'll need a training and validation data set. This prevents us from violating any exchangability assumptions. We'll denote the training data as $Z_t$ and calibration data $Z_c$. The sizes of these datasets are $\|Z_t\|$ and $\|Z_c\|$, respectively. 


```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
```


```
# Load data from https://www.openml.org/d/554
digits = datasets.load_digits()
X = digits.images.reshape(len(digits.images), -1)
y = digits.target
```


```
n_samples = len(X)
print(X.shape)
```

    (1797, 64)



```
#Some example images 
for index, image in enumerate(X[:15]):
  plt.subplot(5, 3, index+1)
  plt.axis('off')
  plt.imshow(image.reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
```


![png](https://github.com/bkompa/bkompa.github.io/raw/master/images/Conformal_Inference_Tutorial_8_0.png)



```
#Split into training, calibration, and test datasets 
X_train_and_calibrate, X_test, y_train_and_calibrate, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, shuffle=True)

X_train, X_calibrate, y_train, y_calibrate = train_test_split(
    X_train_and_calibrate, y_train_and_calibrate, test_size=.25, 
    random_state=42, shuffle=True)
```


```
#Split into training, calibration, and test datasets 
print(X_train.shape, X_calibrate.shape)
```

    (1077, 64) (360, 64)


## Train a learner

We'll train an SVM to classify these digits. This is our _underlying model_. 


```
classifier = svm.SVC(gamma=0.001, probability=True, random_state = 42)
classifier.fit(X_train, y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
        max_iter=-1, probability=True, random_state=42, shrinking=True, tol=0.001,
        verbose=False)




```
print(f"Training Acc: {classifier.score(X_train, y_train)}")
print(f"Calibration Acc: {classifier.score(X_calibrate, y_calibrate)}")
print(f"Test acc: {classifier.score(X_test, y_test)}")
```

    Training Acc: 0.9972144846796658
    Calibration Acc: 1.0
    Test acc: 0.9861111111111112


## Calculate Nonconformity Scores

Next, we have to calculate nonconformity scores $\alpha_i$ for each $z_i$ in the calibration set


```
probability_predictions = classifier.predict_proba(X_calibrate)
alphas = np.array([1-p[idx] for p, idx in zip(probability_predictions, 
                                              y_calibrate)])
```


```
plt.scatter(np.arange(len(alphas)), np.sort(alphas))
plt.title('Sorted Nonconformity Scores')
plt.show()
```


![png](https://github.com/bkompa/bkompa.github.io/raw/master/images/Conformal_Inference_Tutorial_16_0.png)


## Determine Valid Prediction Region

Now we get to the real heart of the conformal algorithm. When classifying a new sample $x_i$ where we do not know the true label $y_i$ where there are $L$ possible labels, we calculate for $C=1,...,L$:
$$P_{\hat{y}^{C_i}} = \frac{|\{j=t+1,...t+c: \alpha_j \geq \alpha_i^{\hat{y}^C_i}\}|}{|Z_c|+1} $$

Let's break down what this means. First, $\alpha_i^{\hat{y}_i^C}$ is the nonconformity score for $\hat{z}_i = (x_i, \hat{y}_i)$ when $\hat{y}_i=C$. The numerator is the number of nonconformity scores in the calibration set that meet or exceed $\alpha_i^{\hat{y}_i^C}$. The denominator is the size of the calibration set plus one, as we have a new nonconformity score we are considering.

We set: $$\Gamma_i^\epsilon = \{C=1,..,L|P_{\hat{y}^{C_i}}>\epsilon\}$$

$P_{\hat{y}^{C_i}}$ is an empirical p-value. We set our valid prediction region to be the set of predicted labels with a p-value exceeding our significance level $\epsilon$



```
def valid_prediction_region(sample, 
                            calibration_alphas, 
                            algorithm, 
                            epsilon=0.05):
  sample_alphas = 1 - algorithm.predict_proba(sample)[0] 
  Z_c = len(calibration_alphas)
  p_values = np.array([np.sum(calibration_alphas>alpha_c)/(Z_c+1) for alpha_c 
                       in sample_alphas])
  return p_values, np.argwhere(p_values>epsilon)
```

Let's try this out. We'll take a few examples from our test set and see the valid prediction region. 


```
#Some example images 
N = 4
for index, image in enumerate(X_test[:N]):
  plt.subplot(2, 2, index+1)
  plt.axis('off')
  plt.imshow(image.reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
  p_values, prediction_region = valid_prediction_region(image.reshape(1,-1), 
                                                        alphas, 
                                                        classifier, 
                                                        0.05)
  plt.title(f"Img {index} Prediction Region: {prediction_region}")
```


![png](https://github.com/bkompa/bkompa.github.io/raw/master/images/Conformal_Inference_Tutorial_20_0.png)


But what happens if we mix images together to get examples that may be harder for the classifier to label? Let's try mixing Images 0 and 3. 


```
mix = (X_test[0]+X_test[3])/2 
plt.title('Mix of a 6 and a 7')
plt.imshow(mix.reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
```


![png](https://github.com/bkompa/bkompa.github.io/raw/master/images/Conformal_Inference_Tutorial_22_0.png)



```
p_values, prediction_region = valid_prediction_region(mix.reshape(1,-1), 
                                                      alphas, 
                                                      classifier, 
                                                      0.05)
print(f"Valid prediction region: {prediction_region}")
```

    Valid prediction region: []


What has happened? We have a null set as our valid prediction region! 


```
print(f'Label probabilities: {classifier.predict_proba(mix.reshape(1,-1))[0]}')
print(f'Nonconformity scores: {1-classifier.predict_proba(mix.reshape(1,-1))[0]}')
print(f'Maximum nonconformity score in the calibration set: {np.max(alphas)}')
```

    Label probabilities: [0.05673796 0.01519488 0.01492302 0.01777739 0.05952113 0.13716491
     0.18467143 0.07216876 0.26858631 0.17325424]
    Nonconformity scores: [0.94326204 0.98480512 0.98507698 0.98222261 0.94047887 0.86283509
     0.81532857 0.92783124 0.73141369 0.82674576]
    Maximum nonconformity score in the calibration set: 0.6406869203063865


The nonconformity scores for this messy sample are off the charts! The empirical p-value we calculated was 0 for every label because the nonconformity score for every label $C$ exceeded all observed nonconformity score in the calibration set. Thus, the numerator in $P_{\hat{y}^C_i}$ is 0. 

Let's try another mixing, this time with $\epsilon=0.01$


```
mix_frac = 0.53
id1, id2 = (1,2)
mix2 = (mix_frac*X_test[id1]+(1-mix_frac)*X_test[id2])
plt.title(f'Mix of a {y_test[id1]} and a {y_test[id2]}')
plt.imshow(mix2.reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

p_values, prediction_region = valid_prediction_region(mix2.reshape(1,-1), 
                                                      alphas, 
                                                      classifier, 
                                                      0.01)
print(f"Valid prediction region: {prediction_region}")
```


![png](https://github.com/bkompa/bkompa.github.io/raw/master/images/Conformal_Inference_Tutorial_28_0.png)


    Valid prediction region: [[3]
     [9]]


Now, our valid prediction region is $\{3, 9\}$. This time, our conservative $\epsilon$ level has ensured that we capture all possible labels for this example. 

## Conformal Prediction Algorithm Summary

0. Divide the dataset into training set $Z_t$ calibration set $Z_c$
1. Train a learner on training data 
2. Calculate nonconformity scores $\alpha_i$ on calibration data 
    * A common choice is $1-p(\hat{y}_i|x_i)$
3. Determine valid prediction regions for new data points 
    * For all labels $C$ calculate: 
        $$P_{\hat{y}^{C_i}} = \frac{|\{j=t+1,...t+c: \alpha_j \geq \alpha_i^{\hat{y}^C_i}\}|}{|Z_c|+1} $$
    $$\Gamma_i^\epsilon = \{C=1,..,L|P_{\hat{y}^{C_i}}>\epsilon\}$$

# Conformal Regression 




Conformal regression is largely the same idea as conformal prediction, but with a few tweaks to work for real valued $y_i$. We'll cover the basic algorithm here, but for a more detailed introduction see [this introduction](http://www.clrc.rhul.ac.uk/copa2017/presentations/CP_Tutorial_2017.pdf) or for a serious treatment, see [Lei et al.](http://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf)

## Conformal Regression Algorithm Summary

0. Divide the dataset into training set $Z_t$ calibration set $Z_c$
1. Train a learner $h$. 
2. Calculate nonconformity scores $\alpha_i$ on calibration data 
  * A common choice is $A(B, z_i) = |y_i-h(x_i)|$. 
  * This function needs to be partially invertible 
  * Save these scores in _descending_ order such that $\alpha_1\geq \alpha_2 \geq ... \geq \alpha_{|Z_c|}$
3. Fix a significance level $\epsilon$. Calculate $s=floor(\epsilon(|Z_c|+1))$. $s$ is the index of the $(1-\epsilon)$-percentile nonconformity score. 
4. The valid prediction region for a new sample $x_i$ is: $$\Gamma_i^\epsilon = h(x_i) \pm \alpha_s$$


# References 



Some references that were specifically consulted in creating this tutorial: 

* Linusson, H. An Introduction to Conformal Inference. 2017. http://www.clrc.rhul.ac.uk/copa2017/presentations/CP_Tutorial_2017.pdf

* Papadopoulos, H. Inductive Conformal Prediction:
Theory and Application to Neural Networks. 2008. http://cdn.intechweb.org/pdfs/5294.pdf

* Shafer, G. & Vovk V. A Tutorial on Conformal Prediction. JMLR. 2008. http://jmlr.csail.mit.edu/papers/volume9/shafer08a/shafer08a.pdf

Another reference not used but useful for a more detailed treatment of conformal regression: 

* Lei J. , G'Sell M., Rinaldo A., Tibshirani R., Wasserman L. Distribution-Free Predictive Inference for Regression. 2016. http://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf
