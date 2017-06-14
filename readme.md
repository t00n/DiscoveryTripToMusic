# Explanation

The goal of this project is to predict Performer, Instrument, Style, Year and Tempo in jazz songs from the Jazzomat Research Project (http://jazzomat.hfm-weimar.de/dbformat/dbcontent.html).
The algorithm used are a linear SVC (from scikit-learn) for multiclass classification (Performer, Instrument, Style) and a regression SVM (from scikit-learn) for regression (Year and Tempo). We use several features extracted from the midi files of the song such as minimum and maximum pitches, average silence duration, note duration and average density of note phrases. We also use the mfcc features and do an optimization for multiclass classification.

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
