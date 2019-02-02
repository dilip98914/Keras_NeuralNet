from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#random seed for reproducibility
np.random.seed(7)

#loading diabetic dataset of 5 years
dataset=np.loadtxt('dia_set.csv',delimiter=',')

#splitting into input(X) and ouptput(Y)
X=dataset[:,0:8]
Y=dataset[:,8]

#create models and add layers
model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))#output layer

#compile model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#call the function to fit to the dataset/TRAINING
model.fit(X,Y,epochs=100,batch_size=10)

#evaluate the model
scores=model.evaluate(X,Y)
print("\n%s:------ %.2f%%" %(model.metrics_names[1],scores[1]*100))
