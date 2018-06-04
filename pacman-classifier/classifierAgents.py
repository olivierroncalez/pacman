# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
import operator
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.

# =============================================================================
# Start of user-defined functions
# =============================================================================

    # k_fold cross validation for model accuracy testing. Allows for insight that would be useful for model testing. This is specific for my Naive Bayes classifier
    def k_fold(self, data, target, K):
        scores = []

        # Shuffle the data
        shuffled_data = []
        shuffled_target = []
        new_index = [i for i in range(len(data))]

        # Inserting shuffled data
        for i in new_index:
            shuffled_data.append(data[i])
            shuffled_target.append(target[i])

        # Replace original data with shuffled data
        data = shuffled_data
        target = shuffled_target

        # Using list comprehension with step size to split data into train/test
        for a in range(K):
            slices_data = [data[i::K] for i in range(K)]
            slices_target = [target[i::K] for i in range(K)]
            validation_data = slices_data[a]
            validation_target = slices_target[a]

            slices_data.pop(a)
            slices_target.pop(a)

            learning_data = np.concatenate(slices_data)
            learning_target = np.concatenate(slices_target)

            # Learns the classifier on the learning data.
            self.probabilities, self.prior = self.Naive_Bayes_Train(learning_data, learning_target)

            # Obtains the score of the classifer based on the test set
            score = self.Bayes_score(validation_data, validation_target)[0]

            scores.append(score)


        # Calculates the mean of the K results and returns it.
        results = np.asarray(scores).mean()
        return results

    # This function runs a split validation score. It will train the classifier and test it using another defined function.
    def train_test_splitter(self, data, target, split = 0.2):

        # Shuffle the data
        shuffled_data = []
        shuffled_target = []
        new_index = [i for i in range(len(data))]

        # Inserting shuffled data
        for i in new_index:
            shuffled_data.append(data[i])
            shuffled_target.append(target[i])

        # Replace original data with shuffled data
        data = shuffled_data
        target = shuffled_target

        # Calculate the length of the testing data.
        d_len = int(len(data))* float(split)
        d_len = int(d_len)

        # Split the test and training data. Since the data is shuffled, I am scrapping the testing data from the top.
        X_test = data[:d_len]
        y_test = target[:d_len]

        # Training data
        X_learn = data[d_len + 1:]
        y_learn = target[d_len + 1:]

        # Runs the classifier to train the training data.
        self.probabilities, self.prior = self.Naive_Bayes_Train(X_learn, y_learn)

        # Utilizes the Bayes_score defined function to calculate the accuracy of classification
        score = self.Bayes_score(X_test, y_test)[0]

        return score

# =============================================================================
# End of user-defined functions
# =============================================================================

    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray

    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.

    # =============================================================================
    #                           NAIVE BAYES CLASSIFIER
    # =============================================================================
    # Creating a naive bayes classifier. This classifier was tested and compared against other scikit learn classifiers with 10-fold cross validation and was shown to have the highest accuracy and maintain high generalizability.

    ############################
    ### Naive Bayes Training ###
    ############################
    def Naive_Bayes_Train(self, data, target):

        # Calculating totals for each class in order to determine prior probabilities.
        prior = [0, 0, 0, 0]
        for i in target:
            if i == 0:
                prior[0] += 1
            if i == 1:
                prior[1] += 1
            if i == 2:
                prior[2] += 1
            if i == 3:
                prior[3] += 1

        # Building dictionary to split data based on classification (necessary for calculating probabilities)
        categorization = {0:[], 1:[], 2:[], 3:[]}

        index = 0
        # Splits the data into the dictionary based on class
        for i in target:
            if i == 0:
                categorization[0].append(data[index])
            if i == 1:
                categorization[1].append(data[index])
            if i == 2:
                categorization[2].append(data[index])
            if i == 3:
                categorization[3].append(data[index])
            index += 1

        # Dictionaries of probabilities - since this is binomial (1 or 0), the probability is getting a 1 for a feature given a particular class.
        probabilities = {0:[], 1:[], 2:[], 3:[]}

        # Iterating over the data by class and calcualting the sum (i.e., number of instances for a given attribute with 1).
        for key, value in categorization.iteritems():
            b = np.asarray(value).sum(axis = 0)
            for i in b:
                # Probability of getting a 1 for a given feature. Probabilities estimated using Laplace smoothing.
                probs = float(i+1)/float(len(categorization[key])+ 25)
                probabilities[key].append(probs) # Appending prob of 1 for per feature in class
        # This list of probabilities & prior is what the data is trained on. Simple enumeration.
        return probabilities, prior


    ###########################
    ### Naive Bayes Testing ###
    ###########################
    def Naive_Bayes_Test(self, features):

        results = {}
        # Iterating over key to calculate probabilities
        for key in range(len(self.probabilities)):

            # Creating empty list to store probability values for each class
            probability = []
            # Calculating prior probability
            proba = float(self.prior[key])/float(sum(self.prior))
            probability.append(proba)

            # iterating over every new feature to calculate probability of getting 1 for each key.
            for i in range(len(features)):

                # If it's not 1, the probability is 1 minus the probability of it being 1 (which is
                # what was calculated)
                if features[i] == 0:
                    probability.append(1 - float(self.probabilities[key][i]))

                # If 1, the probability based on learned data is that which was calculated.
                else:
                   probability.append(float(self.probabilities[key][i]))

            # Storing the results of the probability for that class and vector of features
            probability = np.prod(np.array(probability))
            results[key] = probability

        # Return the best move as a classification. One which maximizes the probability
        return max(results.iteritems(), key=operator.itemgetter(1))[0]


    ###########################################
    ### Accuracy metric  + confusion matrix ###
    ###########################################
    def Bayes_score(self, data_test, target_test):

        # Building an empty confusion matrix.
        confusion_matrix = np.zeros((4,4))

        # Empty list to store the results
        prediction = []
        # Obtaining predictions for every row in the data set and appending them.
        for i in range(len(data_test)):
            pred = self.Naive_Bayes_Test(data_test[i])
            prediction.append(pred)

        score = 0
        # Side by side comparison of the prediction with the target
        for a, b in zip(prediction, target_test):
            # Populating the confusion matrix
            confusion_matrix[a, b] += 1

        # Calculate the accuracy score as the sum of diagonals over the total matrix size
        for x in range(confusion_matrix[0].shape[0]):
            score += confusion_matrix[x, x]
        results = float(score)/float(confusion_matrix.sum())
        return results, confusion_matrix
    # =============================================================================
    #                     END OF NAIVE BAYES CLASSIFIER
    # =============================================================================


    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()


        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of integers 0-3 indicating the action
        # taken in that state.


    # =============================================================================
    # Start: Running the classifier
    # =============================================================================


        # Train test split with 0.2 for my own classifier. This code will run the classifier and test it, returning the score of that split.
        self.split_score = self.train_test_splitter(self.data, self.target, 0.2)

        # Custom built cross-validation score for my NBayes classifier
        self.cross_val_score = self.k_fold(self.data, self.target, 10)


        # Learning built classifier with all the data
        self.probabilities, self.prior = self.Naive_Bayes_Train(self.data, self.target)


        self.score = 0 # This allows us to only print the metrics once (see getAction)
        # Calculating training score
        self.training_score, self.matrix = self.Bayes_score(self.data, self.target)



        # Using scikit learn metrics to compare algorithms
        clf = BernoulliNB().fit(self.data, self.target)
        self.scikit_score = clf.score(self.data, self.target)
        self.scikit_cross_val = cross_val_score(clf, self.data, self.target, cv = 10).mean()
        self.scikit_matrix = confusion_matrix(self.target, clf.predict(self.data))

    # =============================================================================
    # End: Running the classifier
    # =============================================================================


    # Tidy up when Pacman dies
    def final(self, state):
        print "I'm done!"

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state) # This is the test data

        # =============================================================================
        # Calling classifier & metrics
        # =============================================================================

        # Calculating learning metrics for fine tuning of hyperparameters (if needed).
        if self.score == 0:
            print "The training accuracy for the full set of data is: %.2f" %(self.training_score)
            print "\nThe 0.2 split validation resulted in an accuracy of: %.2f" %(self.split_score)
            print "\nThe 10-fold cross validation score for the full set of data is: %.2f" %(self.cross_val_score)
            print  "\nMy classifer confusion matrix is: \n %s" %(self.matrix)
            # Comparing metrics with those from scikit learn
            print "\nThe training accuracy for SCIKIT is: %.2f" %(self.scikit_score)
            print "\nThe 10-fold cross validation score for SCIKIT is: %.2f" %(self.scikit_cross_val)
            print "\nThe SCIKIT confusion matrix is: \n%s\n" %(self.scikit_matrix)
            self.score = 1



        # =============================================================================
        # Classification of new instances
        # =============================================================================

        # Using Naive Bayes to classify feature vector
        classification = self.Naive_Bayes_Test(features)
        # Turning the classifcation into a move.
        best_move = self.convertNumberToMove(classification)


        # Get the actions we can try.
        legal = api.legalActions(state)

        # Checks to see if it's possible to make that move. If not, it will try the next best move based on the probability. If no moves are possible it will return a random move.
        if best_move in legal:
            return api.makeMove(best_move, legal)
        else: return api.makeMove(random.choice(legal), legal)
