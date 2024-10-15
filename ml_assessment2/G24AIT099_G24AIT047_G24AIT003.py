import math
import random
import csv

# encode the sample data 
def encode_class(mydata):
    """
    This function encodes the class labels in the dataset into numerical values.

    Parameters:
    mydata (list): A list of data points, where each data point is a list of features. 
    The last element of each data point is the class label.

    Returns:
    list: The modified dataset with class labels encoded as numerical values. 
    The function also prints the total number of classes and their names.
    """
    classes = []
    # add any new class
    for i in range(len(mydata)):
        if mydata[i][-1] not in classes:
            classes.append(mydata[i][-1])
    # add data points
    for i in range(len(classes)):
        for j in range(len(mydata)):
            if mydata[j][-1] == classes[i]:
                mydata[j][-1] = i
    print("There are total of", len(classes), "classes, which are ", classes)
    return mydata



# Splitting the data
def splitting(mydata, ratio):
    """
    This function splits a dataset into training and testing sets based on a given ratio.

    Parameters:
    mydata (list): A list of data points, where each data point is a list of features.
    ratio (float): The ratio of training data to the total dataset. It should be a value between 0 and 1.

    Returns:
    tuple: A tuple containing two lists: the training set and the testing set.
    """
    train_num = int(len(mydata) * ratio)
    train = []
    # initially testset will have all the dataset
    test = list(mydata)
    while len(train) < train_num:
        # index generated randomly from range 0
        # to length of testset
        index = random.randrange(len(test))
        # from testset, pop data rows and put it in train
        train.append(test.pop(index))
    return train, test


# Group the data rows under each class yes or
# no in dictionary eg: dict[yes] and dict[no]
def groupUnderClass(mydata):
    """
    Group the data points in 'mydata' based on their class labels.

    Parameters:
    mydata (list): A list of data points, where each data point is a list of attribute values.
        The last element of each data point is the class label.

    Returns:
    dict: A dictionary where the keys are the class labels and the values are lists of data points belonging to each class.
    """
    dict = {}
    for i in range(len(mydata)):
        if (mydata[i][-1] not in dict):
            dict[mydata[i][-1]] = []
        dict[mydata[i][-1]].append(mydata[i])
    return dict
 


# Group the data rows under each class yes or
# no in dictionary eg: dict[yes] and dict[no]
def mean(numbers):
    """
    Calculate the mean (average) of a list of numbers.

    Parameters:
    numbers (list): A list of numerical values. The list can contain integers or floats.

    Returns:
    float: The mean (average) of the input list of numbers.
    """
    return sum(numbers) / float(len(numbers))

 
# Calculating Standard Deviation
def std_dev(numbers):
    """
    Calculate the standard deviation of a list of numbers.

    Parameters:
    numbers (list): A list of numerical values. The list can contain integers or floats.

    Returns:
    float: The standard deviation of the input list of numbers. The standard deviation is calculated
    as the square root of the variance, which is the average of the squared differences from the mean.
    In this implementation, a small constant (0.01) is added to the denominator to avoid division by zero.
    """
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1 + 0.01)
    return math.sqrt(variance)
 

def MeanAndStdDev(mydata):
    """
    Calculate the mean and standard deviation for each attribute in the dataset.

    Parameters:
    mydata (list): A list of data points, where each data point is a list of features.
    The last element of each data point is the class label.

    Returns:
    list: A list of tuples, where each tuple contains the mean and standard deviation
    of an attribute. The last tuple corresponds to the class label, which is removed from the result.
    """
    info = [(mean(attribute), std_dev(attribute)) for attribute in zip(*mydata)]
    del info[-1]
    return info
 
# find Mean and Standard Deviation under each class
def MeanAndStdDevForClass(mydata):
    """
    Calculate the mean and standard deviation for each attribute in the dataset grouped by class.

    Parameters:
    mydata (list): A list of data points, where each data point is a list of features.
    The last element of each data point is the class label.

    Returns:
    dict: A dictionary where the keys are the class labels and the values are lists of tuples.
    Each tuple contains the mean and standard deviation of an attribute for the corresponding class.
    The last tuple corresponds to the class label, which is removed from the result.
    """
    info = {}
    dict = groupUnderClass(mydata)  # Assuming groupUnderClass is a function defined elsewhere
    for classValue, instances in dict.items():
        info[classValue] = MeanAndStdDev(instances)  # Assuming MeanAndStdDev is a function defined elsewhere
    return info


# Calculate Gaussian Probability Density Function
def calculateGaussianProbability(x, mean, stdev):
     """
     Calculate the Gaussian probability density function (PDF) for a given value x,
     given the mean and standard deviation of a normal distribution.
 
     Parameters:
     x (float): The value for which the PDF is to be calculated.
     mean (float): The mean (average) of the normal distribution.
     stdev (float): The standard deviation of the normal distribution.
 
     Returns:
     float: The calculated Gaussian PDF value. If the standard deviation is zero,
     the function returns 1.0 if x is equal to the mean, and 0.0 otherwise.
     """
     if stdev == 0:  # Handle zero standard deviation
         return 1.0 if x == mean else 0.0
     expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
     return (1 / (math.sqrt(2 * math.pi) * stdev)) * expo


# Calculate Class Probabilities
def calculateClassProbabilities(info, test):
    """
    Calculate the class probabilities for each data point in the test set using Gaussian Naive Bayes.

    Parameters:
    info (dict): A dictionary containing the mean and standard deviation of each attribute for each class.
        The keys of the dictionary are the class labels, and the values are lists of tuples.
        Each tuple contains the mean and standard deviation of an attribute.
    test (list): A list of data points for which the class probabilities are to be calculated.
        Each data point is a list of attribute values.

    Returns:
    dict: A dictionary containing the class probabilities for each data point in the test set.
        The keys of the dictionary are the indices of the data points in the test set, and the values are dictionaries.
        Each inner dictionary contains the class probabilities for a single data point, with the class labels as keys.
    """
    probabilities = {}
    for classValue, classSummaries in info.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, std_dev = classSummaries[i]
            x = test[i]
            probabilities[classValue] *= calculateGaussianProbability(x, mean, std_dev)
    return probabilities



# Make prediction - highest probability is the prediction
def predict(info, test):
    """
    This function makes predictions for a set of test data points using Gaussian Naive Bayes.

    Parameters:
    info (dict): A dictionary containing the mean and standard deviation of each attribute for each class.
        The keys of the dictionary are the class labels, and the values are lists of tuples.
        Each tuple contains the mean and standard deviation of an attribute.
    test (list): A list of data points for which the class probabilities are to be calculated.
        Each data point is a list of attribute values.

    Returns:
    str: The predicted class label for the test data points. The class label is selected based on the highest probability.
    """
    probabilities = calculateClassProbabilities(info, test)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel



# returns predictions for a set of examples
def getPredictions(info, test):
    """
    This function makes predictions for a set of test data points using Gaussian Naive Bayes.

    Parameters:
    info (dict): A dictionary containing the mean and standard deviation of each attribute for each class.
        The keys of the dictionary are the class labels, and the values are lists of tuples.
        Each tuple contains the mean and standard deviation of an attribute.
    test (list): A list of data points for which the class probabilities are to be calculated.
        Each data point is a list of attribute values.

    Returns:
    list: A list of predicted class labels for the test data points.
    The class label is selected based on the highest probability.
    """
    predictions = []
    for i in range(len(test)):
        result = predict(info, test[i])
        predictions.append(result)
    return predictions
 
# Accuracy score
def accuracy_rate(test, predictions):
    """
    Calculate the accuracy rate of a classification model.

    Parameters:
    test (list): A list of test data points, where each data point is a list of attribute values.
    predictions (list): A list of predicted class labels for the test data points.

    Returns:
    float: The accuracy rate of the classification model, calculated as the percentage of correctly predicted class labels.
    """
    correct = 0
    for i in range(len(test)):
        if test[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(test))) * 100.0
 
 
 # add the data path in your system
filename = r'ion_binary_classification.csv'
# load the file and store it in mydata list
mydata = csv.reader(open(filename, "rt"))
mydata = list(mydata)
mydata = encode_class(mydata)

# List comprehension to remove the first element from each sublist
mydata = [i[1:] for i in mydata]
mydata = mydata[1:]


# Convert string numbers to float values 
for i in range(len(mydata)):
    mydata[i] = [float(x) for x in mydata[i]]


# split ratio = 0.8
# 70% of data is training data and 30% is test data used for testing
ratio = 0.8
train_data, test_data = splitting(mydata, ratio)
print('Total number of examples are: ', len(mydata))
print('Out of these, training examples are: ', len(train_data))
print("Test examples are: ", len(test_data))
 
# prepare model
info = MeanAndStdDevForClass(train_data)
 
# test model
predictions = getPredictions(info, test_data)
accuracy = accuracy_rate(test_data, predictions)


from collections import defaultdict


# Calculate the confusion matrix components: TP, FP, FN, TN
def confusion_matrix_components(test, predictions):
    """
    Calculate the confusion matrix components: TP, FP, FN, TN.

    Parameters:
    test (list): A list of test data points, where each data point is a list of attribute values.
    predictions (list): A list of predicted class labels for the test data points.

    Returns:
    dict: A dictionary containing the counts of true positive (TP), false positive (FP), 
    false negative (FN), and true negative (TN) for each class.
    """
    # Initialize counters for each class
    true_positive = defaultdict(int)
    false_positive = defaultdict(int)
    false_negative = defaultdict(int)
    true_negative = defaultdict(int)

    for i in range(len(test)):
        actual = test[i][-1]
        predicted = predictions[i]

        # True Positive (TP), False Positive (FP), False Negative (FN), True Negative (TN)
        if actual == predicted:
            true_positive[actual] += 1
            for class_value in set(true_positive.keys()).union(false_positive.keys()):
                if class_value != actual:
                    true_negative[class_value] += 1
        else:
            false_positive[predicted] += 1
            false_negative[actual] += 1
            for class_value in set(true_positive.keys()).union(false_positive.keys()):
                if class_value != actual and predicted != class_value:
                    true_negative[class_value] += 1

    return true_positive, false_positive, false_negative, true_negative


# Precision calculation
def precision_score(true_positive, false_positive):
    """
    Calculate the precision score for each class.

    Parameters:
    true_positive (dict): A dictionary containing the counts of true positive instances for each class.
    false_positive (dict): A dictionary containing the counts of false positive instances for each class.

    Returns:
    dict: A dictionary containing the precision score for each class.
    """
    precisions = {}
    for class_value in true_positive.keys():
        if true_positive[class_value] + false_positive[class_value] == 0:
            precisions[class_value] = 0.0
        else:
            precisions[class_value] = true_positive[class_value] / (true_positive[class_value] + false_positive[class_value])
    return precisions



# Recall calculation
def recall_score(true_positive, false_negative):
    """
    Calculate the recall score for each class.

    Parameters:
    true_positive (dict): A dictionary containing the counts of true positive instances for each class.
    false_negative (dict): A dictionary containing the counts of false negative instances for each class.

    Returns:
    dict: A dictionary containing the recall score for each class.
    """
    recalls = {}
    for class_value in true_positive.keys():
        if true_positive[class_value] + false_negative[class_value] == 0:
            recalls[class_value] = 0.0
        else:
            recalls[class_value] = true_positive[class_value] / (true_positive[class_value] + false_negative[class_value])
    return recalls


# F1 Score calculation
def f1_score(precision, recall):
    """
    Calculate the F1 score for each class.

    Parameters:
    precision (dict): A dictionary containing the precision score for each class.
    recall (dict): A dictionary containing the recall score for each class.

    Returns:
    dict: A dictionary containing the F1 score for each class.
    """
    f1_scores = {}
    for class_value in precision.keys():
        if precision[class_value] + recall[class_value] == 0:
            f1_scores[class_value] = 0.0
        else:
            f1_scores[class_value] = 2 * (precision[class_value] * recall[class_value]) / (precision[class_value] + recall[class_value])
    return f1_scores


# Calculate precision, recall, and F1-score for each class
def calculate_metrics(test, predictions):
    """
    Calculate precision, recall, and F1-score for each class in a classification model.

    Parameters:
    test (list): A list of test data points, where each data point is a list of attribute values.
    predictions (list): A list of predicted class labels for the test data points.

    Returns:
    None. The function prints the precision, recall, and F1-score for each class.

    The function calculates the confusion matrix components (TP, FP, FN, TN)
    using the `confusion_matrix_components` function.
    Then, it calculates the precision, recall, and F1-score for each class 
    using the `precision_score`, `recall_score`, and `f1_score` functions.
    Finally, it prints the precision, recall, and F1-score for each class.
    """
    tp, fp, fn, tn = confusion_matrix_components(test, predictions)
    precision = precision_score(tp, fp)
    recall = recall_score(tp, fn)
    f1 = f1_score(precision, recall)

    # Print results for each class
    for class_value in precision.keys():
        print(f"Class {class_value}:")
        print(f"Precision: {precision[class_value]:.2f}")
        print(f"Recall: {recall[class_value]:.2f}")
        print(f"F1-Score: {f1[class_value]:.2f}")
        print("-" * 30)


# Test the model with the new metrics
accuracy = accuracy_rate(test_data, predictions)
print(f"Accuracy of your model is: {accuracy:.2f}%")

# Print precision, recall, and F1-score
calculate_metrics(test_data, predictions)


"""
##### =================================================
##### Comparision Analysis Implementation with Sk-Learn

"""

# %pip install seaborn
# %pip install pandas
# %pip install numpy
# %pip install matplotlib
# %pip install scikit-learn

import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Load the dataset
file_path = 'ion_binary_classification.csv'
df = pd.read_csv(file_path)

# Drop column 'Unnamed: 0' permanently
df.drop('Unnamed: 0', axis=1, inplace=True)

# Display basic information and the first few rows of the dataset
display(df.head(2))


# Check for NaN values in the dataset
if df.isnull().values.any():
    print("Data contains NaN values. Please handle them before fitting the model.")
else:
    print("Data is not having Null valuse.")

# Convert the 'Class' column to binary values: 'good' becomes 1, 'bad' becomes 0
df['Class'] = df['Class'].apply(lambda x: 1 if x == 'good' else 0)
X = df.drop(columns=['Class']).values
y = df['Class'].values


# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Create the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)


# Output metrics

print('Accuracy : ', accuracy)
print('Precision : ', precision)
print('Recall : ', recall)
print('F1 : ', f1)
print('\n')
# Output confusion matrix
print('Confusion Matrix : \n', conf_matrix)


# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()