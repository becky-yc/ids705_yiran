# -*- coding: utf-8 -*-
'''Sample script for solar array image classification

Author:       Kyle Bradbury
Date:         January 30, 2018
Organization: Duke University Energy Initiative
'''

'''
Import the packages needed for classification
'''


'''
Set directory parameters
'''
# Set the directories for the data and the CSV files that contain ids/labels
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from IPython.display import display
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2grey
from sklearn.preprocessing import StandardScaler
from sklearn import svm as svm
from skimage.filters import prewitt
import cv2
import os
os.chdir('/Users/N1/Desktop/705 - ML/Kaggle')

# Set the directories for the data and the CSV files that contain ids/labels
dir_train_images = './data/training/'
dir_test_images = './data/testing/'
dir_train_labels = './data/labels_training.csv'
dir_test_ids = './data/sample_submission.csv'

'''
Include the functions used for loading, preprocessing, features extraction,
classification, and performance evaluation
'''


def load_data(dir_data, dir_labels, training=True):
    ''' Load each of the image files into memory

    While this is feasible with a smaller dataset, for larger datasets,
    not all the images would be able to be loaded into memory

    When training=True, the labels are also loaded
    '''
    labels_pd = pd.read_csv(dir_labels)
    ids = labels_pd.id.values
    data = []
    for identifier in ids:
        fname = dir_data + identifier.astype(str) + '.tif'
        image = mpl.image.imread(fname)
        data.append(image)
    data = np.array(data)  # Convert to Numpy array
    if training:
        labels = labels_pd.label.values
        return data, labels
    else:
        return data, ids


# Load the data
data, labels = load_data(dir_train_images, dir_train_labels, training=True)


def preprocess_and_extract_features(data):
    '''Preprocess data and extract features

    Preprocess: normalize, scale, repair
    Extract features: transformations and dimensionality reduction
    '''
    # Here, we do something trivially simple: we take the average of the RGB
    # values to produce a grey image, transform that into a vector, then
    # extract the mean and standard deviation as features.

    # Make the image grayscale and extract each color
    dataR = np.array(data[:, :, 0])
    dataG = np.array(data[:, :, 1])
    dataB = np.array(data[:, :, 2])
    data = rgb2grey(data)

    # Lists for storage
    data0 = []
    data1 = []
    data2 = []
    data3 = []
    hog_feature_list = []
    for i in range(len(data)):
        # extract hog features of each sample
        hog_features = hog(data[i], block_norm='L2-Hys',
                           pixels_per_cell=(16, 16), feature_vector=True)
        hog_feature_list.append(hog_features)
    hog1 = np.array(hog_feature_list)
    return hog1

    dataR = np.array(dataR)
    dataG = np.array(dataG)
    dataB = np.array(dataB)

    for row in range(data.shape[0]):
        data_4loop = prewitt(data[row])
        data1.append(data_4loop)

    data = np.array(data1)

    vectorized_data = data.reshape(data.shape[0], -1)
    vectorized_dataR = dataR.reshape(dataR.shape[0], -1)
    vectorized_dataG = dataG.reshape(dataG.shape[0], -1)
    vectorized_dataB = dataB.reshape(dataB.shape[0], -1)

    # extract the mean and standard deviation of each sample as features
    feature_mean = np.mean(vectorized_data, axis=1)
    feature_std = np.std(vectorized_data, axis=1)
    feature_variance = np.var(vectorized_data, axis=1)

    feature_meanR = np.mean(vectorized_dataR, axis=1)
    feature_stdR = np.std(vectorized_dataR, axis=1)
    feature_varianceR = np.var(vectorized_dataR, axis=1)
    feature_minR = np.amin(vectorized_dataR, axis=1)

    feature_meanG = np.mean(vectorized_dataG, axis=1)
    feature_stdG = np.std(vectorized_dataG, axis=1)
    feature_varianceG = np.var(vectorized_dataG, axis=1)
    feature_minG = np.amin(vectorized_dataG, axis=1)

    feature_meanB = np.mean(vectorized_dataB, axis=1)
    feature_stdB = np.std(vectorized_dataB, axis=1)
    feature_varianceB = np.var(vectorized_dataB, axis=1)
    feature_minB = np.amin(vectorized_dataB, axis=1)
    feature_maxB = np.amax(vectorized_dataB, axis=1)

    features = np.stack((hog1, feature_variance,
                         feature_meanB, feature_varianceB), axis=-1)


def set_classifier():
    '''Shared function to select the classifier for both performance evaluation
    and testing
    '''
    # return RandomForestClassifier()
    # return KNeighborsClassifier(n_neighbors=10)
    # return SVC(probability=True)
    # return LogisticRegression()
    return svm.SVC(probability=True, C=10)


def cv_performance_assessment(X, y, k, clf):
    '''Cross validated performance assessment

    X   = training data
    y   = training labels
    k   = number of folds for cross validation
    clf = classifier to use

    Divide the training data into k folds of training and validation data.
    For each fold the classifier will be trained on the training data and
    tested on the validation data. The classifier prediction scores are
    aggregated and output
    '''
    # Establish the k folds
    prediction_scores = np.empty(y.shape[0], dtype='object')
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    for train_index, val_index in kf.split(X, y):
        # Extract the training and validation data for this fold
        X_train, X_val = X[train_index], X[val_index]
        y_train = y[train_index]

        # Train the classifier
        X_train_features = preprocess_and_extract_features(X_train)
        clf = clf.fit(X_train_features, y_train)

        # Test the classifier on the validation data for this fold
        X_val_features = preprocess_and_extract_features(X_val)

        #X_val_features   = pca.transform(X_val_features)

        cpred = clf.predict_proba(X_val_features)

        # Save the predictions for this fold
        prediction_scores[val_index] = cpred[:, 1]
    return prediction_scores


def plot_roc(labels, prediction_scores):
    fpr, tpr, _ = metrics.roc_curve(labels, prediction_scores, pos_label=1)
    auc = metrics.roc_auc_score(labels, prediction_scores)
    legend_string = 'AUC = {:0.3f}'.format(auc)

    plt.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
    plt.plot(fpr, tpr, label=legend_string)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid('on')
    plt.axis('square')
    plt.legend()
    plt.tight_layout()


'''
Sample script for cross validated performance
'''
# Set parameters for the analysis
num_training_folds = 20

# Load the data
data, labels = load_data(dir_train_images, dir_train_labels, training=True)

# Choose which classifier to use
clf = set_classifier()

# Perform cross validated performance assessment
prediction_scores = cv_performance_assessment(data, labels, num_training_folds, clf)

# Compute and plot the ROC curves
plot_roc(labels, prediction_scores)

'''
Sample script for producing a Kaggle submission
'''

produce_submission = True  # Switch this to True when you're ready to create a submission for Kaggle

if produce_submission:
    # Load data, extract features, and train the classifier on the training data
    training_data, training_labels = load_data(dir_train_images, dir_train_labels, training=True)
    training_features = preprocess_and_extract_features(training_data)
    clf = set_classifier()
    clf.fit(training_features, training_labels)

    # Load the test data and test the classifier
    test_data, ids = load_data(dir_test_images, dir_test_ids, training=False)
    test_features = preprocess_and_extract_features(test_data)
    test_scores = clf.predict_proba(test_features)[:, 1]

    # Save the predictions to a CSV file for upload to Kaggle
    submission_file = pd.DataFrame({'id':    ids,
                                    'score':  test_scores})
    submission_file.to_csv('submission.csv',
                           columns=['id', 'score'],
                           index=False)
