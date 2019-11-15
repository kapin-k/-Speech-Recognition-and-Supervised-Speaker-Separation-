import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy
from sklearn import metrics

CSVFILE='./speakerresult_1.csv'
test_df=pd.read_csv(CSVFILE)

actualValue=test_df['actual']
predictedValue=test_df['predicted']

actualValue=actualValue.values
predictedValue=predictedValue.values

#Confusion matrix
cmt=confusion_matrix(actualValue,predictedValue)
print (cmt)

#Precision
precision=metrics.precision_score(actualValue,predictedValue,average='macro')
print('Precision',precision)

#Recall
recall=metrics.recall_score(actualValue,predictedValue,average='macro')
print('Recall',recall)

#F1-score
F1score=metrics.f1_score(actualValue,predictedValue,average='macro')
print('F1score',F1score)
