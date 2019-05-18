# Hello There... General Datavity
![Code Quality](https://img.shields.io/pypi/status/Django.svg)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

   Feature Engineering is an area of Machine Learning that consists of creating new features from the ones that we already have. The new
features will be used to optimize the training for our machine learning model. Feature engineering has to be applied to every AI project
and the process is unique every time because the features will never be the same and the methods used will not necessarily be the same.
We start doing Feature Engineering by cleaning the data, this means replacing the missing data with zero, minimum number, maximum 
number, or the mean number of the feature. The reason why sometimes there will be missing data, is because there are times the dataset 
we are working with there will be some missing data. By having cleaned data, it will be more acceptable for the AI model and hence, will
make better predictions. 

  After cleaning the data, we can make them even more attractive as an input by applying several methods like linear scaling and 
clipping.The reason we alter the data is to make the features more similar to each other and that way, the model can make better 
predictions. There are many ways to do feature engineering, but this project focuses on the basics of it.

  Datavity is a language based on Feature Engineering, that handles the data that receives by applying them several methods that we
implemented. The methods consists on showing the features that the dataset has, replacing missing data with several options like zero 
and mean, transforming the data by applying several scaling methods, and applying the Root Mean Square Error (RMSE).
The functions for these methods are to make the features more acceptable for an AI model and see how well predicts the data,
after we apply these methods. Finally, Datavity gives you the opportunity to visualize how the data behaves after applying any method 
you call.

## Motivation

  The motivation for this project is to reduce the repetition of code that is done while doing Feature Engineering and searching for the 
best way of changing them. It will also be a language of having everything more compact and easy to use for people that are new to 
Feature Engineering and handling data. 

## Installation

Must have installed **Python3.6** and the following list of libraries used for Feature Engineering:
- string
- math
- IPython 
- matplotlib 
- pyplot
- numpy 
- pandas
- sklearn

Create a file in Desktop named Datavity_PL and save the project there.
Run the following commands in command prompt (Windows) or terminal (Ubuntu/MacOs).

* MacOs/Ubuntu
``` 
    cd Desktop/Datavity_PL/Datavity/src
    python3 terminal.py 
```
* Windows
```Shell
.\venv\Scripts\activate
```
```Shell
pip install -r requirements.txt
```
    
## Basic Language Syntax and Operations

The Feature Engineering tools that you can use with Datavity are:
* Handle missing data
* See the list of features
* Transform the data with scaling methods.
* Apply RMSE/ creates a scatter plot with a linear regression

### FEATURES

Prints the list of features that are used for the dataset imported.

```
>> FEATURES()
```

### CLEAN

The clean method changes the missing data (NaNs) to one of the four options, ZERO, MIN, MAX, MEAN. The output of this method
is a table with the first 20 rows of the dataframe where you can see the substitution of the NaNs.
```
>> CLEAN(ZERO || MEAN || MIN || MAX)
```

### TRANSFORM:

The transform method is for applying scaling methods to the data of the feature you selected, after you clean it first. The output of 
this method is three graphs, where the first one is the data of the feature unchanged, the second one is the data after applying linear
scaling, and the third one is the data after applying log scaling.

```
>> TRANSFORM(feature_name)
```

### RMSE:

The RMSE method applies the Root Mean Squared Error equation to the feature you want to predict (second one). The RMSE consists of 
calculating how far is the predicted value from the real one. the output is a scatter plot with linear regression, & prints the 
dataframe replaced with clean_method & RMSE in the terminal.

```
>> RMSE(MAX || MIN || MEAN || ZERO, feature_name, feature_name)
```


## Video Demonstration
[![Watch the video](https://img.youtube.com/vi/CD2_CZT9ltw/maxresdefault.jpg)](https://youtu.be/CD2_CZT9ltw)


##  Datavity Report

[Final-Report](https://drive.google.com/a/upr.edu/file/d/1bSyFR5kJeFgZNJ5eQpuhoILYY0k-cBTp/view?usp=sharing)

## Github Page
[Datavity-Github](https://github.com/JaiTorres13/Datavity)

## Authors 

[Jainel M. Torres Santos](https://github.com/JaiTorres13)  
[Cesar Justiniano](https://github.com/CesarJustiniano)  
[Fabian Guzman](https://github.com/fabianguzman)  

