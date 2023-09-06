# Projects
from IPython.core.compilerop import CachingCompiler
#step1: import library
import pandas as pd
#step 2 : import data
cancer = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv')
#step 3 :
cancer.head()
#step 4 :
cancer.info()
#step 5 :
cancer.describe()
#step 6 : define target (y) and features (x)
#step 7 :
cancer.columns
#step 8 :
y = cancer['diagnosis']
#step 9 :
x = cancer.drop(['id','diagnosis','Unnamed: 32'],axis=1)
#step 10 : train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=2529)
#step 11 : check shape of train and test sample
x_train.shape,x_test.shape,y_train.shape,y_test.shape
#step 12 : select model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000)
#step 13 : train or fit model
model.fit(x_train,y_train)
#step 14 :
model.intercept_
#step 15 :
model.coef_
#step 16 :predict model
y_pred = model.predict(x_test)
#step 17 :
y_pred
#step 18 : model accuracy
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#step 19 :
confusion_matrix(y_test,y_pred)
#step 20 :
accuracy_score(y_test,y_pred)
#step 21 :
print(classification_report(y_test,y_pred))
