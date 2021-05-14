# Importar librerías
import pandas as pd
import numpy as np
import re
import os
import nltk
from enum import Enum
import pickle
from nltk.corpus import stopwords

# Librerías de sklearn
import sklearn
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


class PreparateData():
    
    def __init__(self):
        self.df = pd.read_csv('http://45.90.109.111/api/emotion.csv')
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

    
    def __cleanData(self):
        print('Clean data')
        self.df['Text'] = self.df['Text'].apply(self.__cleantext)

   
    def __transformerFeatures(self):
        print('Transformer Features')
        #nltk.download('stopwords')
        vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
        X = vectorizer.fit_transform(self.df['Text']).toarray()

        tfidfconverter = TfidfTransformer()
        self.X = tfidfconverter.fit_transform(X).toarray()
        
        enc = OrdinalEncoder()
        self.df["Emotion_code"] = enc.fit_transform(self.df[["Emotion"]])


    def __preparate(self):
        print('Preparate...')
        self.__cleanData()
        self.__transformerFeatures()


    def onSplit(self):
        self.__preparate()
        print('Split data')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.df['Emotion_code'], test_size=0.2, random_state=42)

        return self.X_test, self.y_test




class MetricsModelPredict():

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        folderBin = '//bin//'
        self.pathBin  = ''.join([dirname, folderBin])
        print(self.pathBin)        
  

    def __getMetricsModel(self, y_test=None, y_pred=None):
        metrics_model = [
            {
                'metric': 'accuracy_score',
                'value':  metrics.accuracy_score(y_test, y_pred),
            },
            {
                'metric': 'f1score',
                'value':  metrics.f1_score(y_test, y_pred, average="macro"),
            },
            {
                'metric': 'precision',
                'value':   metrics.precision_score(y_test, y_pred, average="macro"),
            },
            {
                'metric': 'recall',
                'value':  metrics.recall_score(y_test, y_pred, average="macro")
            },
        ]

        return metrics_model   


    def __loadModel(self, nameModel):
        print(nameModel)
        return pickle.load(open(self.pathBin + str(nameModel), 'rb'))

    
    def onGetMetricsModel(self, nameModel, split):
        print('onGetMetricsModel')

        X_test, y_test = split
        loadModel = self.__loadModel(nameModel)

        y_pred  = loadModel.predict(X_test)
        metrics_result = self.__getMetricsModel(y_test=y_test, y_pred = y_pred)        

        print(metrics_result)
        
        return metrics_result


 
    


    