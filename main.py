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
        #read image of category
        image_path = os.path.join(path,file)
        #store the image to variable
        img = cv2.imread(image_path)
        #resizing the training images
        img = cv2.resize(img,(224,224))
        #append the images into list which store the image and label
        data.append([img,label])

#check data length
print('data',len(data))
# data shuffle use is to prevent to generate the bais
random.shuffle(data)

x=[]
y=[]

# sotre the image feature and label in different list
for features,label in data:
    x.append(features)
    y.append(label)

x = np.array(x)
y = np.array(y)

print("x--->",x.shape)
print("Y",y.shape)
#standard and scale the image variable
x = x/255

#training data from the model
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

print("training DAta",x_train.shape)

print("test data",x_test.shape)

#transfer learning of the algorithm start

#initialize the VGG16 Class CNN
vgg = VGG16()
print("vgg-->",vgg.summary())

#initialize the sequential model
model = Sequential()

#remove the last layer of vgg 16 and add the sequentail layer
for layer in vgg.layers[:-1]:
    model.add(layer)

print("model-->",model.summary())

#Freezing the layer to prevent update the weight on training
for layer in model.layers:
    layer.trainable = False

model.add(Dense(1,activation="sigmoid"))

#print model summary for the verification
print("model-->",model.summary())

#model compilation Training an optimize configuration
model.compile(optimizer="Adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(x_train,y_train,epochs=5,validation_data=(x_test,y_test))

#to generate pickle file
pickle.dump(model,open('mask.pkl',"wb"))

#to save the model in to keras
model.save("maskModel.keras")
