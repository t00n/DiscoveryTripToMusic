# Explanation

For the moment, our code obtains a Mean Absolute Percentage Error of 0.33 using the most optimal features. We have a script to find the optimal features which tries every possible combination and finds the one that minimizes the Mean Absolute Percentage Error using a 5-fold cross validation (using the same data as the perl script).
The algorithm used are a linear SVC (from scikit-learn) for multiclass classification (Performer, Instrument, Style) and a linear regression SVC (from scikit-learn) for linear regression (Year and Tempo). we also added the mfcc features and do an optimization for multiclass classification.
The next step will be to find a custom feature representation and maybe write our own implementation of SVC.

# Installation

With Python 3.3 :

```
    pip install -r requirements.txt
```

You can use virtualenv.

# Use

## crossvalidation using all features

```
    make test
```

Uses the perl script for cross validation

## crossvalidation using our own script

```
    make validation
```

This computes the Mean Absolute Percentage Error

## features selection

```
    make features
```

Selects the best features using our cross-validation function and minimizing the Mean Absolute Percentage Error. There are 5 features for the moment and activation is represented by a list of booleans.
Takes approximately 10-15 mins.

## the library python speech features

get MFCC (which is used to get mfcc features)

```
    git clone https://github.com/jameslyons/python_speech_features.git
    cd ./python_speech_features     
    sudo python setup.py install
     
```
