from fileinput import filename
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg' , 'plas','pres','skin','test','mass','pedi','age','class']

dataframe = pd.read_csv(url,names= names)
array = dataframe.values
x = array[:,0:8]
y = array[:,8]

test_size = 0.2
random_state = 101

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=test_size , random_state=random_state)
model = LogisticRegression()
model.fit(x_train , y_train)

#accuracy
result = model.score(x_test , y_test)
print(result)

#save the model
filename = 'predict_79.pkl' #model extension shoud be pkl or sav
joblib.dump(model , filename)
