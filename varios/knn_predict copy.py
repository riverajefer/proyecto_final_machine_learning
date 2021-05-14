# Importar librerías
from joblib import dump, load
import pandas as pd
import numpy as np
import re
import nltk
from enum import Enum
from nltk.corpus import stopwords

# Librerías de visualiación
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Librerías de sklearn
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report

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

# Limpia el texto
def cleantext(data):
    data = re.sub(r'@[A-Za-z0-9]+', '', data) # remove @mentions
    data = re.sub(r'#', '', data)# remove # tag
    data = re.sub(r'RT[\s]+', '', data) # remove the RT
    data = re.sub(r'https?:\/\/\S+', '', data) # remove links
    data = re.sub('(\\\\u([a-z]|[0-9])+)', ' ', data) # remove unicode characters
    data = re.sub(r'"', '', data)
    data = re.sub(r':', '', data)
    return data


def get_metrics_model(y_test=None, y_pred=None):
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print('fscore: ', metrics.f1_score(y_test, y_pred, average="macro"))
    print('precision: ',metrics.precision_score(y_test, y_pred, average="macro"))
    print('recall: ',metrics.recall_score(y_test, y_pred, average="macro"))    


def plot_confusion_matrix(y_test=None, y_pred=None):
    labels = df['Emotion'].unique()
    cm1 = pd.DataFrame(confusion_matrix(y_test, y_pred), index = labels, columns = labels)

    plt.figure(figsize = (10, 8))
    sns.heatmap(cm1, annot = True, cbar = False, fmt = 'g')
    plt.ylabel('Valores actuales')
    plt.xlabel('Valores predichos')
    plt.show() 


print('Inico proceso !')

df = pd.read_csv('http://45.90.109.111/api/emotion.csv')

df['Text'] = df['Text'].apply(cleantext)

plt.figure(figsize = (20,7))
sns.set_theme(style="darkgrid")
ax = sns.countplot(x="Emotion", data=df, palette="Set3", dodge=False)

#nltk.download('stopwords')
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(df['Text']).toarray()

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

enc = OrdinalEncoder()
df["Emotion_code"] = enc.fit_transform(df[["Emotion"]])

X_train, X_test, y_train, y_test = train_test_split(X, df['Emotion_code'], test_size=0.2, random_state=42)

print('Cargando modelo...')
model_log = load('/root/mlds/models/model_log.joblib')


y_pred  = model_log.predict(X_test)
get_metrics_model(y_test=y_test, y_pred = y_pred)


print(metrics.classification_report(y_test, y_pred))

plot_confusion_matrix(y_test=y_test, y_pred=y_pred)


print('Predicioendo con el modelo cargado...')
predict_emotion_text(model = model_log, vectorizer = vectorizer, text = "my pc is broken but i am really very happy as i got a new pc from my wife ")

print('Listo !')