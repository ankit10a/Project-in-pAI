import os
import random
import pickle
import cv2
import numpy as np
from keras import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from sklearn.model_selection import train_test_split

categories = ['with_mask', 'without_mask']

data = []

for category in categories:
    #path combine by the help of os modules
    path = os.path.join('train',category)

    # categories labelling
    label = categories.index(category)

    for file in os.listdir(path):
        image_path = os.path.join(path,file)
        img = cv2.imread(image_path)
        #resizing the training images
        img = cv2.resize(img,(224,224))

        data.append([img,label])


#check data length
print('data',len(data))

random.shuffle(data)

x=[]
y=[]

for features,label in data:
    x.append(features)
    y.append(label)

x = np.array(x)
y = np.array(y)

print("x--->",x.shape)
print("Y",y.shape)

x = x/255

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

print("training DAta",x_train.shape)

print("test data",x_test.shape)

vgg = VGG16()
print("vgg-->",vgg.summary())

model = Sequential()


for layer in vgg.layers[:-1]:
    model.add(layer)

print("model-->",model.summary())

for layer in model.layers:
    layer.trainable = False

model.add(Dense(1,activation="sigmoid"))

#print model summary for the verification
print("model-->",model.summary())

#model Training an optimize configuration
model.compile(optimizer="Adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=5,validation_data=(x_test,y_test))

#to generate pickle file
pickle.dump(model,open('mask.pkl',"wb"))

#to save the model in to keras
model.save("maskModel.keras")
