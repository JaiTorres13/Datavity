#######################################
# IMPORTS
#######################################


import string
import math

from IPython import display
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


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

#features that are numeric within the dataset
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

"""
		FEATURES(): prints the list of features from current dataset
"""
def FEATURES():
    for feature in cols:
        print(feature)


"""
		CLEAN(clean_method): Handles missing data inside the dataframe
			replacing Nans with MAX, MIN, MEAN, and ZERO
		param: clean_method -> can be MAX, MIN, MEAN, and ZERO
		returns: dataframe with Nans substituted with either MAX, MIN, MEAN, and ZERO
"""
def CLEAN(clean_method):
		clean_data = dataset.copy()
		clean_method_lower = str(clean_method).lower()
		if (clean_method_lower[-4:] == 'zero'):
				clean_data.fillna(0, inplace=True)

		elif (clean_method_lower[-4:] == 'mean'):
				for feature in NUMERIC_FEATURES:
						clean_data.fillna({feature: dataset[feature].mean()}, inplace=True)

		elif (clean_method_lower[-3:] == 'min'):
				for feature in NUMERIC_FEATURES:
						clean_data.fillna({feature: dataset[feature].min()}, inplace=True)

		else:
				for feature in NUMERIC_FEATURES:
						clean_data.fillna({feature: dataset[feature].max()}, inplace=True)

		clean_method_trimmed = clean_method_lower[-4:]
		if (clean_method_trimmed[0] == ':'):
				clean_method_trimmed = clean_method_trimmed[-3:]
		print('Replaced Nans with ' + clean_method_trimmed)
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

    return series.apply(lambda x: clip_to_min if x < clip_to_min else clip_to_max if x > clip_to_max else x)


"""
		TRANSFORM(feature_name): Applying feature scaling to a feature in cols
		param: feature_name -> Feature from the cols list
		return: 3 Histograms, where the first one is the feature unchanged, the second one is applying linear scaling 
						to the feature & the third one is applying log scaling to the feature.
"""
def TRANSFORM(feature_name):

    if feature_name not in NUMERIC_FEATURES:
        return False

    dataframe = dataset
    clip_min = -np.inf
    clip_max = np.inf
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.title(feature_name)
    histogram = dataframe[feature_name].hist(bins=50)

    plt.subplot(1, 3, 2)
    plt.title("linear_scaling")
    scaled_features = dataset.copy()
    scaled_features[feature_name] = linear_scale(
        clip(dataframe[feature_name], clip_min, clip_max))
    histogram = scaled_features[feature_name].hist(bins=50)

    plt.subplot(1, 3, 3)
    plt.title("log scaling")
    log_normalized_features = dataset.copy()
    log_normalized_features[feature_name] = log_scale(dataframe[feature_name])
    histogram = log_normalized_features[feature_name].hist(bins=50)
    plt.show()
    return True


#draws the scatter plot
def make_scatter_plot(dataframe, input_feature, target,
                      slopes=[], biases=[], model_names=[]):
   
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


"""
	RMSE(clean_method, x, y): calculates RMSE of the input feature (x) with the prediction label (y) & applying 
													  linear regression in the scatter plot
	param: clean_method -> MEAN,MAX,MIN,ZERO
				 x -> input feature, used to predict label
				 y -> label, what you want to predict
	return: a scatter plot with a linear regression, & prints the dataframe replaced with clean_method & RMSE in the terminal 
"""
def RMSE(clean_method, x, y):
    LABEL = y
    INPUT_FEATURE = x

    if LABEL not in NUMERIC_FEATURES or INPUT_FEATURE not in NUMERIC_FEATURES:
        return False

    clean_data = CLEAN(clean_method)
    clean_method_lower = str(clean_method).lower()

    clean_method_trimmed = clean_method_lower[-4:]
    if (clean_method_trimmed[0] == ':'):
        clean_method_trimmed = clean_method_trimmed[-3:]

    # Plotting the new data set where rows where interpolated or estimated (filled)

    # model bias
    x = clean_data[INPUT_FEATURE]
    y = clean_data[LABEL]
    opt = np.polyfit(x, y, 1)
    y_pred = opt[0] * x + opt[1]
    opt_rmse = math.sqrt(metrics.mean_squared_error(y_pred, y))
    print("Root mean squared error for " + clean_method_trimmed + " substitution")
    print(opt_rmse)
    slope = opt[0]
    bias = opt[1]

    plt.ylabel(LABEL)
    plt.xlabel(INPUT_FEATURE)
    plt.scatter(clean_data[INPUT_FEATURE], clean_data[LABEL], c='black')
    plt.title('Scatter Plot when Nan subtittuted by ' + clean_method_trimmed)
    make_scatter_plot(clean_data, INPUT_FEATURE, LABEL,
                      [slope], [bias], ["initial model"])
    plt.show()
    return True

# examples on how to run it in python
# FEATURES()
# TRANSFORM('losses')
# CLEAN('zero')
# RMSE('zero', 'price', 'losses')

# examples on how to run it in datavity
# FEATURES()
# TRANSFORM(feature) where feature is any string
# CLEAN(ZERO) it can be also MIN, MAX, MEAN
# RMSE(ZERO feature feature) where feature is any string & first param can be also MIN, MAX, MEAN
