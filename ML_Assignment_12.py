#!/usr/bin/env python
# coding: utf-8

# # Assignment 12 Solutions

# SUBMITTED BY: YASH AJANKAR

# ##### 1. What is prior probability ? Give an example ?

# **Ans:** Prior probability shows the likelihood of an outcome in a given dataset. For example, in the mortgage case, `P(Y)` is the default rate on a home mortgage, which is `2%`. `P(Y|X)` is called the conditional probability, which provides the probability of an outcome given the evidence, that is, when the value of X is known.
# 
# "\"\\\"![0_V0GyOt3LoDVfY7y5.png](https://miro.medium.com/max/1400/1*ZM1ZhhgrtU7UnxRii04Clg.png)\""
# 

# ##### 2. What is posterior probability ? Give an example ?

# **Ans:**: Posterior probability is a revised probability that takes into account new available information. For example, let 
# there be two urns, urn A having 5 black balls and 10 red balls and urn B having 10 black balls and 5 red balls.
# 
# "\"\\\"![0_V0GyOt3LoDVfY7y5.png](https://miro.medium.com/max/1400/1*ZM1ZhhgrtU7UnxRii04Clg.png)\""

# ##### 3. What is likelihood probability ? Give an example ?

# **Ans:** Likelihood Function in Machine Learning and Data Science is the joint probability distribution(jpd) of the dataset given as a function of the parameter. Think of it as the probability of obtaining the observed data given the parameter values.
# 
# "\"\\\"![0_V0GyOt3LoDVfY7y5.png](https://miro.medium.com/max/501/1*hk9LX1qkkBmRDyJ-ntLbpw.png)\""

# ##### 4. What is Naïve Bayes classifier ? Why is it named so ?

# **Ans:** Naive Bayes is a simple and powerful algorithm for predictive modeling. Naive Bayes is called naive because it assumes that each input variable is independent. This is a strong assumption and unrealistic for real data; however, the technique is very effective on a large range of complex problems.
# 
# "\"\\\"![0_V0GyOt3LoDVfY7y5.png](https://miro.medium.com/proxy/1*ZW1icngckaSkivS0hXduIQ.jpeg)\""

# ##### 5. What is optimal Bayes classifier ?

# **Ans:** The Bayes Optimal Classifier is a probabilistic model that makes the most probable prediction for a new example. Bayes Optimal Classifier is a probabilistic model that finds the most probable prediction using the training data and space of hypotheses to make a prediction for a new data instance.
# 
# "\"\\\"![0_V0GyOt3LoDVfY7y5.png](data:image/png;base64,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)\""

# ##### 6. Write any two features of Bayesian learning methods ?

# **Ans:** A probability distribution over observed data for each possible hypothesis. New instances can be classified by combining the predictions of multiple hypotheses, weighted by their probabilities.
# 
# "\"\\\"![0_V0GyOt3LoDVfY7y5.png](https://slideplayer.com/slide/13615815/83/images/70/Features+of+Bayesian+Learning.jpg)\""
# 

# ##### 7. Define the concept of consistent learners ?

# **Ans:** **Consistent Learners:** A learner L using a hypothesis H and training data D is said to be a consistent learner if it  always outputs a hypothesis with zero error on D whenever H contains such a hypothesis. • By definition, a consistent learner must produce a hypothesis in the version space for H given D. 
# 
# "\"\\\"![0_V0GyOt3LoDVfY7y5.png](https://slideplayer.com/slide/13365780/80/images/11/Consistent+Learners.jpg)\""
# 

# ##### 8. Write any two strengths of Bayes classifier ?

# **Ans:** This algorithm works quickly and can save a lot of time. Naive Bayes is suitable for solving multi-class prediction problems. If its assumption of the independence of features holds true, it can perform better than other models and requires much less training data.
# 
#   * It is simple and easy to implement.
#   * It doesn't require as much training data.
#   * It handles both continuous and discrete data.
#   * It is highly scalable with the number of predictors and data points.
#   * It is fast and can be used to make real-time predictions.

# ##### 9. Write any two weaknesses of Bayes classifier ?

# **Ans:** The greatest weakness of the naïve Bayes classifier is that it relies on an often-faulty assumption of equally important and independent features which results in biased posterior probabilities.
# 
# 
#   * If your test data set has a categorical variable of a category that wasn’t present in the training data set, the Naive Bayes model will assign it zero probability and won’t be able to make any predictions in this regard. This phenomenon is called ‘Zero Frequency,’ and you’ll have to use a smoothing technique to solve this problem.
#   * This algorithm is also notorious as a lousy estimator. So, you shouldn’t take the probability outputs of ‘predict_proba’ too seriously. 
#   * It assumes that all the features are independent. While it might sound great in theory, in real life, you’ll hardly find a set of independent features. 
# 

# ##### 10. Explain how Naïve Bayes classifier is used for:
# 1. Text classification
# 2. Spam filtering
# 3. Market sentiment analysis

# **Ans:** Navie Bayes Classifier is used for:
# - **Text classification:**
# The Naive Bayes classifier is a simple classifier that classifies based on probabilities of events. It is the applied  commonly to text classification. With the training set, we can train a Naive Bayes classifier which we can use to automaticall categorize a new sentence.
# - **Spam filtering:**        
# Naive Bayes classifiers work by correlating the use of tokens (typically words, or sometimes other things), with spam and non-spam e-mails and then using Bayes' theorem to calculate a probability that an email is or is not spam.  It is one of the oldest ways of doing spam filtering, with roots in the 1990s.
# - **Market sentiment analysis:**    
# Market Sentiment analysis is a field dedicated to extracting subjective emotions and feelings from text. One common use of sentiment analysis is to figure out if a text expresses negative or positive feelings. Naive Bayes is a popular algorithm for classifying text.
