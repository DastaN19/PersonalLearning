import pickle

#load the model
model = pickle.load(open('predict_79.pkl','rb'))
data = model.predict([[1,1,1,1,1,1,1,1]])

if data[0] == 0:
    print('non diabetic')
else:
    print('Diabetic')