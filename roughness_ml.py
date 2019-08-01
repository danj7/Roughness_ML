import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

#to save model.summary()
from contextlib import redirect_stdout

#Setting a seed for reproducibility
np.random.seed(7)

#dimensions of input image of interface (a square)
length = 128

#amount of images per zeta value
Ninterfaces = 2000

#zetas to test: min value, max value, step
#zetas = np.arange(0.1, 1.01, 0.1)
zetas = np.arange(0.05, 1.05, 0.05)

#proportion of train data to use as validation
test_size_proportion = 0.2

#other params
batch_size = 64
num_classes = len(zetas)
epochs = 10

#Generating train and validate sets
from interfaces.interface_generation import *
interfs_train, interfs_valid, zetas_train, zetas_valid = generate_train_validate_set(Ninterfaces, length,
                                                                                     zetas, test_size_proportion)
#reshaping data to 'channels_last' format
interfs_train = interfs_train.reshape(interfs_train.shape[0], length, length, 1)
interfs_valid = interfs_valid.reshape(interfs_valid.shape[0], length, length, 1)
input_shape = (length, length, 1)

# convert class vectors to binary class matrices
zetas_train = keras.utils.to_categorical(zetas_train*num_classes-1, num_classes)
zetas_valid = keras.utils.to_categorical(zetas_valid*num_classes-1, num_classes)
#the *num_classes-1 is because .to_categorical() requires array of ints starting from 0.

#Model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#Compiling model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.adam(),
              metrics=['accuracy'])

#Training and evaluating
model.fit(interfs_train, zetas_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(interfs_valid, zetas_valid))

score = model.evaluate(interfs_valid, zetas_valid, verbose=0)

#Plotting the training and validation accuracy curve
history = model.history
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(acc) + 1)

#Train and validation accuracy
plt.plot(epochs_range, acc, 'b', label='Training accurarcy')
plt.plot(epochs_range, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()
plt.savefig('train_and_val_acc.png')

#Train and validation loss
plt.figure()
plt.plot(epochs_range, loss, 'b', label='Training loss')
plt.plot(epochs_range, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.savefig('train_and_val_loss.png')

#testing with some new interfaces
#reseeding the RNG
np.random.seed()
#choosing 5 random zeta values and generating 2 interfaces per zeta
random_zetas = np.random.choice(zetas, 5)
interfs_test, zetas_test = generate_interfaces(2, length, random_zetas)

#reshaping for prediction and saving predictions
interfs_test_pred = interfs_test.reshape(10, length, length, 1)
predictions = model.predict(interfs_test_pred)

#extracting predicting zetas
zetas_pred = []
for i in range(len(interfs_test)):
    zetas_pred.append(np.round(zetas[predictions[i].argmax()],2))
#convert to numpy array
zetas_pred = np.array(zetas_pred)

#fraction of zetas the model got right in this Test.
# print('zetas_test = ',zetas_test)
# print('zetas_test dtype =', zetas_test.dtype)
# print('zetas_pred = ',zetas_pred)
# print('zetas_pred dtype =', zetas_pred.dtype)
# print('resta =', zetas_test-zetas_pred)
# print('resta array =', zetas_test[np.abs(zetas_test-zetas_pred)<1e-2])
# bool_array = np.array(zetas_test==zetas_pred.astype('float32'))
# print('bool_array= ', bool_array)
fractional_accuracy = 1.0 * len(zetas_test[np.abs(zetas_test-zetas_pred)<1e-2])
fractional_accuracy /= len(zetas_test)
print('fractional_accuracy =',fractional_accuracy)
print('Percentage accuracy in testing = '+ str(fractional_accuracy*100)+ '%')
#calculating rms error
rms_error = np.sqrt( ((zetas_pred - zetas_test)**2).sum() / len(zetas_test) )
print('RMS Error: '+str(np.round(rms_error, 2)))

#Save the model and model information
model.save('model_roughness_ml.h5')
with open('model_results.txt', 'a') as file:
    with redirect_stdout(file):
        model.summary()
    file.write('Length of square: '+str(length)+'\n')
    file.write('Interfaces per zeta: '+str(Ninterfaces)+'\n')
    file.write('Number of Epochs: '+str(epochs)+'\n')
    file.write('Validation loss: '+str(score[0])+'\n')
    file.write('Validation accuracy: '+str(score[1])+'\n')
    file.write('Classes: '+str(len(zetas))+' zeta values = '+str(zetas)+'\n')
    file.write('zetas_test = '+str(zetas_test)+'\n')
    file.write('zetas_pred = '+str(zetas_pred)+'\n')
    file.write('Test fractional accuracy: '+str(fractional_accuracy)+'\n')
    file.write('RMS Error: '+str(np.round(rms_error, 2))+'\n')
    file.write('*****************************************************************\n')
