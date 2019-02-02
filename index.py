from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
import numpy as np

#random seed for reproducibility
np.random.seed(2)

#loading diabetic dataset of 5 years
dataset=np.loadtxt('dia_set.csv',delimiter=',')

#splitting into input(X) and ouptput(Y)
X=dataset[:,0:8]
Y=dataset[:,8]

#split X,Y into train and test 
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


#create models and add layers
model=Sequential()
model.add(Dense(15,input_dim=8,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dropout(.2))
model.add(Dense(1,activation='sigmoid'))#output layer

#compile model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#call the function to fit to the dataset/TRAINING
model.fit(x_train,y_train,epochs=1000,batch_size=20,validation_data=(x_test,y_test))

#evaluate the model
scores=model.evaluate(X,Y)
print("\n%s:------ %.2f%%" %(model.metrics_names[1],scores[1]*100))

model.save('weights.h5')