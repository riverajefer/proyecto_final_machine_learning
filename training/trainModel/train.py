# Importar librerías
from datetime import datetime, date 
import pandas as pd
import numpy as np
from scipy import stats
import datetime
import re
import scipy
import nltk
from enum import Enum
from nltk.corpus import stopwords
import pickle

# Librerías de visualiación
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns

# Librerías de sklearn
import sklearn
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC        
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

class TrainModel():

    def __init__(self):
        self.df = None
        self.X = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    
    # Limpia el texto
    def __cleantext(self, data):
        data = re.sub(r'@[A-Za-z0-9]+', '', data) # remove @mentions
        data = re.sub(r'#', '', data)# remove # tag
        data = re.sub(r'RT[\s]+', '', data) # remove the RT
        data = re.sub(r'https?:\/\/\S+', '', data) # remove links
        data = re.sub('(\\\\u([a-z]|[0-9])+)', ' ', data) # remove unicode characters
        data = re.sub(r'"', '', data)
        data = re.sub(r':', '', data)

        return data

    def __getMetricsModel(self, y_test=None, y_pred=None):
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print('fscore: ', metrics.f1_score(y_test, y_pred, average="macro"))
        print('precision: ',metrics.precision_score(y_test, y_pred, average="macro"))
        print('recall: ',metrics.recall_score(y_test, y_pred, average="macro"))          

    
    def __cleanData(self):
        print('Clean data')
        self.df['Text'] = self.df['Text'].apply(self.__cleantext)


    def __loadData(self):
        print('Load Data')
        self.df = pd.read_csv('http://45.90.109.111/api/emotion.csv')
        self.__cleanData()
    
    
    def __transformerFeatures(self):
        print('Transformer Features')
        nltk.download('stopwords')
        vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
        X = vectorizer.fit_transform(self.df['Text']).toarray()

        tfidfconverter = TfidfTransformer()
        self.X = tfidfconverter.fit_transform(X).toarray()
        
        enc = OrdinalEncoder()
        self.df["Emotion_code"] = enc.fit_transform(self.df[["Emotion"]])

        #refactor
        vec_file = 'vectorizer.pickle'
        pickle.dump(vectorizer, open(vec_file, 'wb'))


    def __split(self):
        print('Split data')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.df['Emotion_code'], test_size=0.2, random_state=42)

    
    def __preparate(self):
        print('Preparate')
        self.__loadData()
        self.__transformerFeatures()
        self.__split()


    def __preditTest(self, model):
        print('Predict')
        y_pred  = model.predict(self.X_test)
        self.__getMetricsModel(y_test=self.y_test, y_pred = y_pred)        
    
    
    def onFitLogisticRegression(self):
        print('OnFit')

        self.__preparate()

        print('On Training...')
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
        clf.fit(self.X_train, self.y_train)

        mod_file = 'LogisticRegression.model'
        pickle.dump(clf, open(mod_file, 'wb'))        

        self.__preditTest(model=clf)


    def onDecisionTreeClassifier(self):
        print('OnFit')

        self.__preparate()

        print('On Training DecisionTreeClassifier...')
        clf = DecisionTreeClassifier()
        clf.fit(self.X_train, self.y_train)

        mod_file = 'DecisionTreeClassifier.model'
        pickle.dump(clf, open(mod_file, 'wb'))        

        self.__preditTest(model=clf)


    def onDecisionTreeClassifierGrid(self):
        print('OnFit')

        self.__preparate()

        print('On Training DecisionTreeClassifierGrid...')
        rfc=DecisionTreeClassifier(random_state=1)

        param_grid = {'criterion':['gini','entropy'], 'max_depth' : [3,5,7,20]}
        clf = GridSearchCV(rfc, param_grid=param_grid, cv=5)
        clf.fit(self.X_train, self.y_train)
        
        mod_file = 'DecisionTreeClassifierGrid.model'
        pickle.dump(clf, open(mod_file, 'wb'))        

        self.__preditTest(model=clf)


    def onMLPClassifier(self):
        print('OnFit')
        self.__preparate()

        print('On Training MLPClassifier...')
        clf = MLPClassifier(random_state=1, hidden_layer_sizes=(100,100,100), max_iter=300, alpha=0.0001)
        clf.fit(self.X_train, self.y_train)
        
        mod_file = 'MLPClassifier.model'
        pickle.dump(clf, open(mod_file, 'wb'))        

        self.__preditTest(model=clf)

    def onRandomForestClassifier(self):
        print('OnFit')
        self.__preparate()

        print('On Training RandomForestClassifier...')
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(self.X_train, self.y_train)
        
        mod_file = 'RandomForestClassifier.model'
        pickle.dump(clf, open(mod_file, 'wb'))        

        self.__preditTest(model=clf)

    

    