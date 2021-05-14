from trainModel.train import TrainModel

TRAIN_MODEL = TrainModel()

print('Inicio')

#TRAIN_MODEL.onFitLogisticRegression()
#TRAIN_MODEL.onDecisionTreeClassifier()
#TRAIN_MODEL.onMLPClassifier()
TRAIN_MODEL.onRandomForestClassifier()