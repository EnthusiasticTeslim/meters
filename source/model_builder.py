
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import warnings
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


from source.data_preprocess import DataPreprocessing
warnings.filterwarnings("ignore")


class ModelBuilder(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__(*args, **kwargs)

    def dt(self, X_train, X_test, y_train, y_test):
        #Create DT model
        DT_classifier = MLPClassifier()

        #Train the model
        DT_classifier.fit(X_train, y_train)

        #Test the model
        DT_predicted = DT_classifier.predict(X_test)

        error = 0
        for i in range(len(y_test)):
            error += np.sum(DT_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        #get performance
        self.accuracy = accuracy_score(y_test, DT_predicted)

        return DT_classifier
    
    def ann(self, X_train, X_test, y_train, y_test, hidden_lay_sizes, learn_rate_init, iteration):

        ## params: Create a new jupyter notebook 
        # to try different values for the hidden_layer_sizes, learning_rate_init, and max_iter hyperparameters (

        # Create ANN model
        if (hidden_lay_sizes == None) and (learn_rate_init == None) and (iteration == None):
                
            ann_classifier = MLPClassifier()
                
        else:
            if (hidden_lay_sizes != None):
                    
                ann_classifier = MLPClassifier(
                                                hidden_layer_sizes = hidden_lay_sizes
                                               )
            if (learn_rate_init != None):
                    
                ann_classifier = MLPClassifier(
                                                learning_rate_init = learn_rate_init
                                               )
                
            if (iteration != None):
                ann_classifier = MLPClassifier(
                                               max_iter = iteration
                                               )
                

        # Train the model
        ann_classifier.fit(X_train, y_train)

        # Test the model
        ytrain_predicted = ann_classifier.predict(X_train)
        ytest_predicted = ann_classifier.predict(X_test)

        # get performance
        self.trainScores = accuracy_score(y_train, ytrain_predicted)
        self.testScores = accuracy_score(y_test, ytest_predicted)

        return ann_classifier 
    
    