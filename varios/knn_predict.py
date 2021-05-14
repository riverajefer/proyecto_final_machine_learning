# Importar librerías
from joblib import dump, load
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from enum import Enum
from nltk.corpus import stopwords


# Librerías de sklearn
from sklearn.feature_extraction.text import CountVectorizer

print('Inicio !!')

# Con el código númerico retorna el string con la emoción
def EmotionCodeToText(code):
    selector = {
        2.0: 'happy',
        4.0: 'sadness',
        0.0: 'anger',
        1.0: 'fear',
        3.0: 'love',
        5.0: 'surprise',
    }

    return selector.get(code)


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


# función para predecir el texto a emoción
def predict_emotion_text(model = None, text='love', vectorizer=None):
  code_predict = model.predict(vectorizer.transform([text]))
  code_predict = code_predict[0]
  emotion_text = EmotionCodeToText(code_predict)

  print(f'Emotion: { emotion_text} | {str(EmotionTextToEmoji[emotion_text])} ')      


loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
loaded_model = pickle.load(open('classification.model', 'rb'))
predict_emotion_text(model = loaded_model, vectorizer = loaded_vectorizer, text = "my pc is broken but i am really very happy as i got a new pc from my wife ")

print('Listo !')