from joblib import dump, load
import pandas as pd
import numpy as np
import re
import os
import nltk
import pickle
from enum import Enum
from nltk.corpus import stopwords
from pydantic import BaseModel
from pathlib import Path
import pathlib
import sklearn.metrics as metrics

# Con el string con la emoción retorna el emoji
class EmotionTextToEmoji(str, Enum):
    happy = '\U0001F600'
    sadness = '\U0001F625'
    anger = '\U0001F620'
    fear  = '\U0001F628'
    love  = '\U0001F970'
    surprise = '\U0001F632'
   
    def __str__(self):
      return self.value


class ModelPredict():

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        folderBin = '//bin//'
        self.pathBin  = ''.join([dirname, folderBin])
        print(self.pathBin)
 

    # Con el código númerico retorna el string con la emoción
    def __emotionCodeToText(self, code):
        selector = {
            2.0: 'happy',
            4.0: 'sadness',
            0.0: 'anger',
            1.0: 'fear',
            3.0: 'love',
            5.0: 'surprise',
        }

        return selector.get(code)


    def __predictEmotionText(self, model = None, text='love', vectorizer=None):
        code_predict = model.predict(vectorizer.transform([text]))
        code_predict = code_predict[0]
        emotion_text = self.__emotionCodeToText(code_predict)

        print(f'Emotion: { emotion_text} | {str(EmotionTextToEmoji[emotion_text])}') 

        return emotion_text

    
    def __loadedVectorizer(self):
        loaded_vectorizer = pickle.load(open(self.pathBin + 'vectorizer.pickle', 'rb'))

        return loaded_vectorizer
    
    
    def onPredictLogisticRegression(self, text):
        loaded_vectorizer = self.__loadedVectorizer()

        loaded_model = pickle.load(open(self.pathBin + 'LogisticRegression.model', 'rb'))
        emotion_text = self.__predictEmotionText(model = loaded_model, vectorizer = loaded_vectorizer, text = text)

        return emotion_text


    def onPredictDecisionTreeClassifier(self, text):
        loaded_vectorizer = self.__loadedVectorizer()

        loaded_model = pickle.load(open(self.pathBin + 'DecisionTreeClassifier.model', 'rb'))
        emotion_text = self.__predictEmotionText(model = loaded_model, vectorizer = loaded_vectorizer, text = text)

        return emotion_text


    def onPredictMLPClassifierPredic(self, text):
        loaded_vectorizer = self.__loadedVectorizer()

        loaded_model = pickle.load(open(self.pathBin + 'MLPClassifier.model', 'rb'))
        emotion_text = self.__predictEmotionText(model = loaded_model, vectorizer = loaded_vectorizer, text = text)

        return emotion_text


    def onRandomForestClassifier(self, text):
        loaded_vectorizer = self.__loadedVectorizer()

        loaded_model = pickle.load(open(self.pathBin + 'RandomForestClassifier.model', 'rb'))
        emotion_text = self.__predictEmotionText(model = loaded_model, vectorizer = loaded_vectorizer, text = text)

        return emotion_text


