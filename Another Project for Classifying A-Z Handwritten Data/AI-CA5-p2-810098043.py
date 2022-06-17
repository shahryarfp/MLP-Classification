#!/usr/bin/env python
# coding: utf-8

# # AI CA5 Phase 2

# In[37]:


import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers
import datetime
from sklearn import metrics
from keras.regularizers import l2


# In[2]:


# from google.colab import drive
# drive.mount('/content/drive')


# ## Phase 1

# ### Reading Dataset

# In[3]:


dataset_ = pd.read_csv('A_Z Handwritten Data.csv')


# ### Organizing Datas

# In[4]:


y = dataset_['0']
del dataset_['0']
x = dataset_


# In[5]:


x = x.values.tolist()
new_x = pd.DataFrame()
new_x['value'] = x
x = new_x


# In[6]:


x


# In[7]:


y


# In[8]:


print('num of datas:', len(x))
classes = []
for i in range(len(y)):
    if y[i] not in classes:
        classes.append(y[i])
print('num of classes:', len(classes))


# In[9]:


classes_count_dict = {}
for i in range(len(y)):
    if y[i] not in classes_count_dict:
        classes_count_dict[y[i]] = 1
    else:
        classes_count_dict[y[i]] += 1

y_axis = list(classes_count_dict.keys())
x_axis = list(classes_count_dict.values())
plt.barh(y_axis,x_axis)
plt.ylabel('Class')
plt.xlabel('Count')
plt.show()


# In[10]:


shown_classes = []
for i in range(len(x)):
    if y[i] not in shown_classes:
        print('class',y[i])
        plt.imshow(np.array(x.loc[i].value).reshape(28,28))
        plt.show()
        shown_classes.append(y[i])


# ### Deviding into train & test

# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)
x_train = np.array(list(x_train.value))
x_test = np.array(list(x_test.value))


# ### One hot encoding for y:<br>
# we are doing it because in the last layer of network we have neurons according to the number of classes. So due to the fact that each neuron must have a output value, we have to assign 0 to these neurons and 1 to the correct neuron.

# In[12]:


y_train = tf.keras.utils.to_categorical(y_train, len(classes))
y_test = tf.keras.utils.to_categorical(y_test, len(classes))


# ## Phase 2

# ### Creating the Model

# In[93]:


model = tf.keras.Sequential()
activation_func = 'relu'
model.add(layers.Flatten(input_shape = (28*28, 1)))
model.add(layers.Dense(200, activation=activation_func))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(100, activation=activation_func))
model.add(layers.Dropout(0.25))
#last layer
model.add(layers.Dense(len(classes), activation='softmax'))
print(model.summary())


# ### Compiling the Model

# In[94]:


loss_func = 'categorical_crossentropy'
optimizer_func = 'SGD'
model.compile(loss=loss_func, optimizer=optimizer_func, metrics=['accuracy'])


# ## Phase 3

# ### Normalizing datas

# In[13]:


x_train = x_train/255
x_test = x_test/255


# ### Part 0:

# ### Training the Model

# In[96]:


start = datetime.datetime.now()
trainedModel = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data = (x_test, y_test))
end = datetime.datetime.now()


# In[97]:


print('Training Duration: ', end-start)


# ### Evaluating the Model

# In[98]:


history = trainedModel.history
loss = history['loss']
val_loss = history['val_loss']
acc = history['accuracy']
val_acc = history['val_accuracy']

plt.xlabel('Epochs')
plt.ylabel('loss')
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss','val_loss'])
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc','val_acc'])


# ### Train metrics

# In[99]:


# train metrics
y_pred = model.predict(x_train)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_train_temp = []
for i in range(len(y_train)):
    train_index = np.argmax(y_train[i])
    y_train_temp.append(train_index)
    
#calculating metrics
print('for train data:')
print(metrics.classification_report(y_train_temp, y_pred_temp, digits=3))


# ### Test metrics

# In[100]:


# test metrics
y_pred = model.predict(x_test)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_test_temp = []
for i in range(len(y_test)):
    test_index = np.argmax(y_test[i])
    y_test_temp.append(test_index)
    
#calculating metrics
print('for test data:')
print(metrics.classification_report(y_test_temp, y_pred_temp, digits=3))


# ### Using LeakyRelu:

# In[101]:


#creating the model
model = tf.keras.Sequential()
activation_func = 'LeakyReLU'
model.add(layers.Flatten(input_shape = (28*28, 1)))
model.add(layers.Dense(200, activation=activation_func))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(100, activation=activation_func))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(len(classes), activation='softmax'))

#compiling the model
loss_func = 'categorical_crossentropy'
optimizer_func = 'SGD'
model.compile(loss=loss_func, optimizer=optimizer_func, metrics=['accuracy'])

#training the model
start = datetime.datetime.now()
trainedModel = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data = (x_test, y_test))
end = datetime.datetime.now()

print('Training Duration: ', end-start)

#evaluating the model
history = trainedModel.history
loss = history['loss']
val_loss = history['val_loss']
acc = history['accuracy']
val_acc = history['val_accuracy']
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss','val_loss'])
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc','val_acc'])


# In[102]:


y_pred = model.predict(x_test)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_test_temp = []
for i in range(len(y_test)):
    test_index = np.argmax(y_test[i])
    y_test_temp.append(test_index)
    
#calculating metrics
print('for test data:')
print(metrics.classification_report(y_test_temp, y_pred_temp, digits=3))


# #### So it seems that relu is woring better in this network

# ### Part 1 (Optimizer):

# ### Effect of Momentum:<br>
# Float hyperparameter >= 0 that accelerates gradient descent in the relevant direction and dampens oscillations. Defaults to 0, i.e., vanilla gradient descent. momentum is an easy and quick way to improve upon standard Stochastic Gradient Descent for optimizing Neural Network models.<br> 
# #### Update rule for parameter w with gradient g when momentum is 0:<br>
# w = w - learning_rate * g<br>
# <br>
# #### Update rule when momentum is larger than 0:<br>
# velocity = momentum * velocity - learning_rate * g<br>
# w = w + velocity<br>

# #### Momentum = 0.5

# In[14]:


#creating the model
model = tf.keras.Sequential()
activation_func = 'relu'
model.add(layers.Flatten(input_shape = (28*28, 1)))
model.add(layers.Dense(200, activation=activation_func))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(100, activation=activation_func))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(len(classes), activation='softmax'))

#compiling the model
loss_func = 'categorical_crossentropy'
optimizer_func = tf.keras.optimizers.SGD(lr=0.01, momentum=0.5)
model.compile(loss=loss_func, optimizer=optimizer_func, metrics=['accuracy'])

#training the model
start = datetime.datetime.now()
trainedModel = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data = (x_test, y_test))
end = datetime.datetime.now()

print('Training Duration: ', end-start)

#evaluating the model
history = trainedModel.history
loss = history['loss']
val_loss = history['val_loss']
acc = history['accuracy']
val_acc = history['val_accuracy']
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss','val_loss'])
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc','val_acc'])


# In[15]:


# train metrics
y_pred = model.predict(x_train)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_train_temp = []
for i in range(len(y_train)):
    train_index = np.argmax(y_train[i])
    y_train_temp.append(train_index)
    
#calculating metrics
print('for train data:')
print(metrics.classification_report(y_train_temp, y_pred_temp, digits=3))


# In[16]:


# test metrics
y_pred = model.predict(x_test)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_test_temp = []
for i in range(len(y_test)):
    test_index = np.argmax(y_test[i])
    y_test_temp.append(test_index)
    
#calculating metrics
print('for test data:')
print(metrics.classification_report(y_test_temp, y_pred_temp, digits=3))


# #### Momentum = 0.9

# In[18]:


#creating the model
model = tf.keras.Sequential()
activation_func = 'relu'
model.add(layers.Flatten(input_shape = (28*28, 1)))
model.add(layers.Dense(200, activation=activation_func))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(100, activation=activation_func))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(len(classes), activation='softmax'))

#compiling the model
loss_func = 'categorical_crossentropy'
optimizer_func = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss=loss_func, optimizer=optimizer_func, metrics=['accuracy'])

#training the model
start = datetime.datetime.now()
trainedModel = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data = (x_test, y_test))
end = datetime.datetime.now()

print('Training Duration: ', end-start)

#evaluating the model
history = trainedModel.history
loss = history['loss']
val_loss = history['val_loss']
acc = history['accuracy']
val_acc = history['val_accuracy']
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss','val_loss'])
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc','val_acc'])


# In[19]:


# train metrics
y_pred = model.predict(x_train)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_train_temp = []
for i in range(len(y_train)):
    train_index = np.argmax(y_train[i])
    y_train_temp.append(train_index)
    
#calculating metrics
print('for train data:')
print(metrics.classification_report(y_train_temp, y_pred_temp, digits=3))


# In[20]:


# test metrics
y_pred = model.predict(x_test)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_test_temp = []
for i in range(len(y_test)):
    test_index = np.argmax(y_test[i])
    y_test_temp.append(test_index)
    
#calculating metrics
print('for test data:')
print(metrics.classification_report(y_test_temp, y_pred_temp, digits=3))


# #### Momentum = 0.98

# In[21]:


#creating the model
model = tf.keras.Sequential()
activation_func = 'relu'
model.add(layers.Flatten(input_shape = (28*28, 1)))
model.add(layers.Dense(200, activation=activation_func))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(100, activation=activation_func))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(len(classes), activation='softmax'))

#compiling the model
loss_func = 'categorical_crossentropy'
optimizer_func = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.98)
model.compile(loss=loss_func, optimizer=optimizer_func, metrics=['accuracy'])

#training the model
start = datetime.datetime.now()
trainedModel = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data = (x_test, y_test))
end = datetime.datetime.now()

print('Training Duration: ', end-start)

#evaluating the model
history = trainedModel.history
loss = history['loss']
val_loss = history['val_loss']
acc = history['accuracy']
val_acc = history['val_accuracy']
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss','val_loss'])
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc','val_acc'])


# In[23]:


# train metrics
y_pred = model.predict(x_train)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_train_temp = []
for i in range(len(y_train)):
    train_index = np.argmax(y_train[i])
    y_train_temp.append(train_index)
    
#calculating metrics
print('for train data:')
print(metrics.classification_report(y_train_temp, y_pred_temp, digits=3))


# In[24]:


# test metrics
y_pred = model.predict(x_test)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_test_temp = []
for i in range(len(y_test)):
    test_index = np.argmax(y_test[i])
    y_test_temp.append(test_index)
    
#calculating metrics
print('for test data:')
print(metrics.classification_report(y_test_temp, y_pred_temp, digits=3))


# It is obvious that with using momentum, the training decreased and the accuracy increased!<br>
# Between three momentum that we analyzed the best one was momentum=0.9<br>
# For momentum = 0.5 & 0.98 results was not satisfying!<br>
# Here is the link to why momentum=0.9 is better: https://towardsdatascience.com/why-0-9-towards-better-momentum-strategies-in-deep-learning-827408503650

# ### Adam Optimizer

# In[25]:


#creating the model
model = tf.keras.Sequential()
activation_func = 'relu'
model.add(layers.Flatten(input_shape = (28*28, 1)))
model.add(layers.Dense(200, activation=activation_func))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(100, activation=activation_func))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(len(classes), activation='softmax'))

#compiling the model
loss_func = 'categorical_crossentropy'
optimizer_func = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=loss_func, optimizer=optimizer_func, metrics=['accuracy'])

#training the model
start = datetime.datetime.now()
trainedModel = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data = (x_test, y_test))
end = datetime.datetime.now()

print('Training Duration: ', end-start)

#evaluating the model
history = trainedModel.history
loss = history['loss']
val_loss = history['val_loss']
acc = history['accuracy']
val_acc = history['val_accuracy']
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss','val_loss'])
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc','val_acc'])


# In[26]:


# train metrics
y_pred = model.predict(x_train)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_train_temp = []
for i in range(len(y_train)):
    train_index = np.argmax(y_train[i])
    y_train_temp.append(train_index)
    
#calculating metrics
print('for train data:')
print(metrics.classification_report(y_train_temp, y_pred_temp, digits=3))


# In[27]:


# test metrics
y_pred = model.predict(x_test)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_test_temp = []
for i in range(len(y_test)):
    test_index = np.argmax(y_test[i])
    y_test_temp.append(test_index)
    
#calculating metrics
print('for test data:')
print(metrics.classification_report(y_test_temp, y_pred_temp, digits=3))


# Accuracy for Adam ==> 0.977<br>
# Accuracy for SGD with Momentum=0.9 ==> 0.978<br>

# ### Part 2(epoch):

# In[28]:


#creating the model
model = tf.keras.Sequential()
activation_func = 'relu'
model.add(layers.Flatten(input_shape = (28*28, 1)))
model.add(layers.Dense(200, activation=activation_func))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(100, activation=activation_func))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(len(classes), activation='softmax'))

#compiling the model
loss_func = 'categorical_crossentropy'
optimizer_func = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=loss_func, optimizer=optimizer_func, metrics=['accuracy'])

#training the model
start = datetime.datetime.now()
trainedModel = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data = (x_test, y_test))
end = datetime.datetime.now()

print('Training Duration: ', end-start)

#evaluating the model
history = trainedModel.history
loss = history['loss']
val_loss = history['val_loss']
acc = history['accuracy']
val_acc = history['val_accuracy']
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss','val_loss'])
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc','val_acc'])


# In[29]:


# train metrics
y_pred = model.predict(x_train)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_train_temp = []
for i in range(len(y_train)):
    train_index = np.argmax(y_train[i])
    y_train_temp.append(train_index)
    
#calculating metrics
print('for train data:')
print(metrics.classification_report(y_train_temp, y_pred_temp, digits=3))


# In[30]:


# test metrics
y_pred = model.predict(x_test)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_test_temp = []
for i in range(len(y_test)):
    test_index = np.argmax(y_test[i])
    y_test_temp.append(test_index)
    
#calculating metrics
print('for test data:')
print(metrics.classification_report(y_test_temp, y_pred_temp, digits=3))


# As we can see we had a little improvement in accuracy<br><br>
# The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters.<br>
# Exactly like humans when we want to learn a book we read it more than once, the Network need to iterate on a dataset more than once to update its weights so it can increase its accuracy.<br><br>
# 
# Using mode epochs is not helpful allways! it could result in overfitting. some times the network will memorize the train datas so we shouldn't use more epochs allways and we should mind this problem. There is some simple methods to avoid overfitting when using large number of epochs like early-stopping.

# ### Part 3(Loss Function):

# In[31]:


#creating the model
model = tf.keras.Sequential()
activation_func = 'relu'
model.add(layers.Flatten(input_shape = (28*28, 1)))
model.add(layers.Dense(200, activation=activation_func))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(100, activation=activation_func))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(len(classes), activation='softmax'))

#compiling the model
loss_func = 'mse'
optimizer_func = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=loss_func, optimizer=optimizer_func, metrics=['accuracy'])

#training the model
start = datetime.datetime.now()
trainedModel = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data = (x_test, y_test))
end = datetime.datetime.now()

print('Training Duration: ', end-start)

#evaluating the model
history = trainedModel.history
loss = history['loss']
val_loss = history['val_loss']
acc = history['accuracy']
val_acc = history['val_accuracy']
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss','val_loss'])
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc','val_acc'])


# In[32]:


# train metrics
y_pred = model.predict(x_train)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_train_temp = []
for i in range(len(y_train)):
    train_index = np.argmax(y_train[i])
    y_train_temp.append(train_index)
    
#calculating metrics
print('for train data:')
print(metrics.classification_report(y_train_temp, y_pred_temp, digits=3))


# In[33]:


# test metrics
y_pred = model.predict(x_test)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_test_temp = []
for i in range(len(y_test)):
    test_index = np.argmax(y_test[i])
    y_test_temp.append(test_index)
    
#calculating metrics
print('for test data:')
print(metrics.classification_report(y_test_temp, y_pred_temp, digits=3))


# Accuracy for MSE ==> 0.970<br>
# Accuracy for Categorical Crossentropy ==> 0.977<br><br>
# MSE is not good for Classificaion problems.<br><br>
# There are two reasons why Mean Squared Error(MSE) is a bad choice for binary classification problems:<br>
# First, using MSE means that we assume that the underlying data has been generated from a normal distribution (a bell-shaped curve). In Bayesian terms this means we assume a Gaussian prior. While in reality, a dataset that can be classified into two categories (i.e binary) is not from a normal distribution but a Bernoulli distribution.<br>
# Secondly, the MSE function is non-convex for binary classification. In simple terms, if a binary classification model is trained with MSE Cost function, it is not guaranteed to minimize the Cost function. This is because MSE function expects real-valued inputs in range(-∞, ∞), while binary classification models output probabilities in range(0,1) through the sigmoid/logistic function.<br><br>
# On a final note, MSE is a good choice for a Cost function when we are doing Linear Regression (i.e fitting a line through data for extrapolation). In the absence of any knowledge of how the data is distributed assuming normal/gaussian distribution is perfectly reasonable.<br><br>
# Source: https://towardsdatascience.com/why-using-mean-squared-error-mse-cost-function-for-binary-classification-is-a-bad-idea-933089e90df7#:~:text=There%20are%20two%20reasons%20why,we%20assume%20a%20Gaussian%20prior

# ### Part 4(Regularization):

# ### Without dropout:

# In[34]:


#creating the model
model = tf.keras.Sequential()
activation_func = 'relu'
model.add(layers.Flatten(input_shape = (28*28, 1)))
model.add(layers.Dense(200, activation=activation_func))
model.add(layers.Dense(100, activation=activation_func))
model.add(layers.Dense(len(classes), activation='softmax'))

#compiling the model
loss_func = 'categorical_crossentropy'
optimizer_func = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=loss_func, optimizer=optimizer_func, metrics=['accuracy'])

#training the model
start = datetime.datetime.now()
trainedModel = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data = (x_test, y_test))
end = datetime.datetime.now()

print('Training Duration: ', end-start)

#evaluating the model
history = trainedModel.history
loss = history['loss']
val_loss = history['val_loss']
acc = history['accuracy']
val_acc = history['val_accuracy']
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss','val_loss'])
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc','val_acc'])


# As it is obvious without dropout we have overfitting problem

# In[35]:


# train metrics
y_pred = model.predict(x_train)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_train_temp = []
for i in range(len(y_train)):
    train_index = np.argmax(y_train[i])
    y_train_temp.append(train_index)
    
#calculating metrics
print('for train data:')
print(metrics.classification_report(y_train_temp, y_pred_temp, digits=3))


# In[36]:


# test metrics
y_pred = model.predict(x_test)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_test_temp = []
for i in range(len(y_test)):
    test_index = np.argmax(y_test[i])
    y_test_temp.append(test_index)
    
#calculating metrics
print('for test data:')
print(metrics.classification_report(y_test_temp, y_pred_temp, digits=3))


# ### Adding Regularization<br>
# Weight regularization provides an approach to reduce the overfitting of a deep learning neural network model on the training data and improve the performance of the model on new data, such as the holdout test set.

# In[38]:


#creating the model
model = tf.keras.Sequential()
activation_func = 'relu'
model.add(layers.Flatten(input_shape = (28*28, 1)))
model.add(layers.Dense(200, activation=activation_func, kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)))
model.add(layers.Dense(100, activation=activation_func, kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)))
model.add(layers.Dense(len(classes), activation='softmax'))

#compiling the model
loss_func = 'categorical_crossentropy'
optimizer_func = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=loss_func, optimizer=optimizer_func, metrics=['accuracy'])

#training the model
start = datetime.datetime.now()
trainedModel = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data = (x_test, y_test))
end = datetime.datetime.now()

print('Training Duration: ', end-start)

#evaluating the model
history = trainedModel.history
loss = history['loss']
val_loss = history['val_loss']
acc = history['accuracy']
val_acc = history['val_accuracy']
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss','val_loss'])
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc','val_acc'])


# As we can see the overfitting problem is decreased!

# In[39]:


# train metrics
y_pred = model.predict(x_train)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_train_temp = []
for i in range(len(y_train)):
    train_index = np.argmax(y_train[i])
    y_train_temp.append(train_index)
    
#calculating metrics
print('for train data:')
print(metrics.classification_report(y_train_temp, y_pred_temp, digits=3))


# In[40]:


# test metrics
y_pred = model.predict(x_test)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_test_temp = []
for i in range(len(y_test)):
    test_index = np.argmax(y_test[i])
    y_test_temp.append(test_index)
    
#calculating metrics
print('for test data:')
print(metrics.classification_report(y_test_temp, y_pred_temp, digits=3))


# ### Adding Dropout<br>
# The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.

# In[43]:


#creating the model
model = tf.keras.Sequential()
activation_func = 'relu'
model.add(layers.Flatten(input_shape = (28*28, 1)))
model.add(layers.Dense(200, activation=activation_func, kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(100, activation=activation_func, kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(len(classes), activation='softmax'))

#compiling the model
loss_func = 'categorical_crossentropy'
optimizer_func = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=loss_func, optimizer=optimizer_func, metrics=['accuracy'])

#training the model
start = datetime.datetime.now()
trainedModel = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data = (x_test, y_test))
end = datetime.datetime.now()

print('Training Duration: ', end-start)

#evaluating the model
history = trainedModel.history
loss = history['loss']
val_loss = history['val_loss']
acc = history['accuracy']
val_acc = history['val_accuracy']
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss','val_loss'])
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc','val_acc'])


# As we can see the overfitting problem solved!

# In[44]:


# train metrics
y_pred = model.predict(x_train)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_train_temp = []
for i in range(len(y_train)):
    train_index = np.argmax(y_train[i])
    y_train_temp.append(train_index)
    
#calculating metrics
print('for train data:')
print(metrics.classification_report(y_train_temp, y_pred_temp, digits=3))


# In[45]:


# test metrics
y_pred = model.predict(x_test)
y_pred_temp = []
for i in range(len(y_pred)):
    pred_index = np.argmax(y_pred[i])
    y_pred_temp.append(pred_index)
    
y_test_temp = []
for i in range(len(y_test)):
    test_index = np.argmax(y_test[i])
    y_test_temp.append(test_index)
    
#calculating metrics
print('for test data:')
print(metrics.classification_report(y_test_temp, y_pred_temp, digits=3))


# In[ ]:




