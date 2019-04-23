#######################################
# IMPORTS
#######################################


import string
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics

# import tensorflow as tf
# from tensorflow.contrib.learn.python.learn import learn_io, estimator


#######################################
# CONSTANTS
#######################################


DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

#######################################
# DOWNLOAD DATASET & CREATE DATAFRAME
#######################################

# Set the output display to have one digit for decimal places, for display
# readability only and limit it to printing 15 rows.
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.max_rows = 20

# Provide the names for the columns since the CSV file with the data does
# not have a header row.
cols = ['symboling', 'losses', 'make', 'fuel-type', 'aspiration', 'num-doors',
        'body-style', 'drive-wheels', 'engine-location', 'wheel-base',
        'length', 'width', 'height', 'weight', 'engine-type', 'num-cylinders',
        'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio',
        'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
NUMERIC_FEATURES = ['compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price',
                    'symboling', 'losses', 'wheel-base', 'length', 'width', 'height', 'weight', 'engine-size']

# Load in the data from a CSV file that is comma seperated.
dataset = pd.read_csv('~/Desktop/Datavity_PL/Datavity/imports-85.csv',
                      sep=',', names=cols, header=None, encoding='latin-1')

for fe in NUMERIC_FEATURES:
    dataset[fe] = pd.to_numeric(dataset[fe], errors='coerce')


#############################################
# Intermediate Code
#############################################

def FEATURES():
    for feature in cols:
        print(feature)


def CLEAN(way):
    clean_data = dataset.copy()
    way_s = str(way).lower()
    if (way_s[-4:] == 'zero'):
        clean_data.fillna(0, inplace=True)

    elif (way_s[-4:] == 'mean'):
        for feature in NUMERIC_FEATURES:
            clean_data.fillna({feature: dataset[feature].mean()}, inplace=True)

    elif (way_s[-3:] == 'min'):
        for feature in NUMERIC_FEATURES:
            clean_data.fillna({feature: dataset[feature].min()}, inplace=True)

    else:
        for feature in NUMERIC_FEATURES:
            clean_data.fillna({feature: dataset[feature].max()}, inplace=True)

    ways = way_s[-4:]
    if (ways[0] == ':'):
        ways = ways[-3:]
    print('Replaced Nans with ' + ways)
    print(clean_data[1:20])
    return clean_data


#############################################################

# Linearly rescales to the range [0, 1]
def linear_scale(series):
    min_val = series.min()
    max_val = series.max()
    scale = 1.0 * (max_val - min_val)
    return series.apply(lambda x: ((x - min_val) / scale))


# Perform log scaling
def log_scale(series):
    return series.apply(lambda x: math.log(x + 1.0))


# Clip all features to given min and max
def clip(series, clip_to_min, clip_to_max):
    # You need to modify this to actually do the clipping versus just returning
    # the series unchanged

    return series.apply(lambda x: clip_to_min if x < clip_to_min else clip_to_max if x > clip_to_max else x)


def TRANSFORM(feature_name):
    valid_feature = str(feature_name).lower()
    if valid_feature not in NUMERIC_FEATURES:
        return False

    dataframe = dataset
    clip_min = -np.inf
    clip_max = np.inf
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.title(valid_feature)
    histogram = dataframe[valid_feature].hist(bins=50)

    plt.subplot(1, 3, 2)
    plt.title("linear_scaling")
    scaled_features = dataset.copy()
    scaled_features[valid_feature] = linear_scale(
        clip(dataframe[valid_feature], clip_min, clip_max))
    histogram = scaled_features[valid_feature].hist(bins=50)

    plt.subplot(1, 3, 3)
    plt.title("log scaling")
    log_normalized_features = dataset.copy()
    log_normalized_features[valid_feature] = log_scale(dataframe[valid_feature])
    histogram = log_normalized_features[valid_feature].hist(bins=50)
    plt.show()
    return True


def make_scatter_plot(dataframe, input_feature, target,
                      slopes=[], biases=[], model_names=[]):
    """ Creates a scatter plot of input_feature vs target along with the models.
  
  Args:
    dataframe: the dataframe to visualize
    input_feature: the input feature to be used for the x-axis
    target: the target to be used for the y-axis
    slopes: list of model weight (slope) 
    bias: list of model bias (same size as slopes)
    model_names: list of model_names to use for legend (same size as slopes)
  """
    # Define some colors to use that go from blue towards red
    cmap = cm.get_cmap("spring")
    colors = [cmap(x) for x in np.linspace(0, 1, len(slopes))]

    # Generate the Scatter plot
    x = dataframe[input_feature]
    y = dataframe[target]
    plt.ylabel(target)
    plt.xlabel(input_feature)
    plt.scatter(x, y, color='black', label="")

    # Add the lines corresponding to the provided models
    for i in range(0, len(slopes)):
        y_0 = slopes[i] * x.min() + biases[i]
        y_1 = slopes[i] * x.max() + biases[i]
        plt.plot([x.min(), x.max()], [y_0, y_1],
                 label=model_names[i], color=colors[i])
    if (len(model_names) > 0):
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def RMSE(way, x, y):
    LABEL3 = str(y).lower()
    INPUT_FEATURE3 = str(x).lower()

    if LABEL3 not in NUMERIC_FEATURES or INPUT_FEATURE3 not in NUMERIC_FEATURES:
        return False

    clean_data = CLEAN(way)
    way_s = str(way).lower()

    ways = way_s[-4:]
    if (ways[0] == ':'):
        ways = ways[-3:]

    # Plotting the new data set where rows where interpolated or estimated (filled)

    # model bias
    x3 = clean_data[INPUT_FEATURE3]
    y3 = clean_data[LABEL3]
    opt3 = np.polyfit(x3, y3, 1)
    y_pred3 = opt3[0] * x3 + opt3[1]
    opt_rmse3 = math.sqrt(metrics.mean_squared_error(y_pred3, y3))
    print("Root mean squared error for mean substitution")
    print(opt_rmse3)
    slope3 = opt3[0]
    bias3 = opt3[1]

    plt.ylabel(LABEL3)
    plt.xlabel(INPUT_FEATURE3)
    plt.scatter(clean_data[INPUT_FEATURE3], clean_data[LABEL3], c='black')
    plt.title('Scatter Plot when Nan subtittuted by ' + ways)
    make_scatter_plot(clean_data, INPUT_FEATURE3, LABEL3,
                      [slope3], [bias3], ["initial model"])
    plt.show()
    return True

# examples on how to run it in python
# FEATURES()
# TRANSFORM('losses')
# CLEAN('zero')
# RMSE('zero', 'prices', 'losses')

# examples on how to run it in datavity
# FEATURES()
# TRANSFORM(feature) where feature is any string
# CLEAN(ZERO) it can be also MIN, MAX, MEAN
# RMSE(ZERO feature feature) where feature is any string & first param can be also MIN, MAX, MEAN
