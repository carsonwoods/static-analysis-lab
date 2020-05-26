import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend
import json

# Sets the level of verbosity for Tensorflow.
# 0 = No logs. 1 = Logs.
verbose_level = 1

if verbose_level == 0:
    print('''Tensorflow verbosity level set to 0. Training logs will not be printed.''')
if verbose_level == 1:
    print('''Tensorflow verbosity level set to 1. Training logs will be printed.''')


######################
# Preprocessing Data #
######################

# Combine all words into training/validation strings
training_strings = []
validation_strings = []


# Stores values corresponding to languages.
# 0 is memory_leak
# 1 is heap_overflow
# 2 is stack_overflow
# 3 is command_injection
training_target = []
validation_target = []

memory_leak_lines = []
heap_overflow_lines = []
stack_overflow_lines = []
command_injection_lines = []


# Load Imporant Data
print("Loading Memory Leak Data...")
with open("../json_data/memory_leak.json", 'r', encoding = "ISO-8859-1") as f:
    lines = f.readlines()
    for line in lines:
        if "\"function\" : " in line:
            temp = (line.split(":")[1]).strip()
            temp = temp.strip(",")
            training_strings.append(temp.strip("\""))
            training_target.append(0);
print("Done.")

print("Loading Heap Overflow Data...")
with open("../json_data/heap_overflow.json", 'r', encoding = "ISO-8859-1") as f:
    lines = f.readlines()
    for line in lines:
        if "\"function\" : " in line:
            temp = (line.split(":")[1]).strip()
            temp = temp.strip(",")
            training_strings.append(temp.strip("\""))
            training_target.append(1);
print("Done.")

print("Loading Stack Overflow Data...")
with open("../json_data/stack_overflow.json", 'r', encoding = "ISO-8859-1") as f:
    lines = f.readlines()
    for line in lines:
        if "\"function\" : " in line:
            temp = (line.split(":")[1]).strip()
            temp = temp.strip(",")
            training_strings.append(temp.strip("\""))
            training_target.append(2);
print("Done.")

print("Loading Command Injection Data...")
with open("../json_data/command_injection.json", 'r', encoding = "ISO-8859-1") as f:
    lines = f.readlines()
    for line in lines:
        if "\"function\" : " in line:
            temp = (line.split(":")[1]).strip()
            temp = temp.strip(",")
            training_strings.append(temp.strip("\""))
            training_target.append(3);
print("Done.")

count = 1
for i, element in enumerate(training_strings):
    if count == 10:
        validation_strings.append(training_strings.pop(i))
        validation_target.append(training_target.pop(i))
        count = 0
    count += 1

print(len(training_strings))
print(len(training_target))

print(len(validation_target))
print(len(validation_strings))

"""
count = 1
for word in memory_leak_file:
    # Does preprocessing for training_strings and target labels
    cleanedWord = word.strip()

    if (len(cleanedWord) == 5):
        # Reads in english words from file
        if count == 10:
            count = 1
            validation_strings.append(cleanedWord)
            validation_target.append(0)
        else:
            count = count + 1
            training_strings.append(cleanedWord)
            training_target.append(0)

count = 1
for word in germanFile:
    # Does preprocessing for training_strings and target labels
    cleanedWord = word.strip()


    if (len(cleanedWord) == 5):
        # Reads in english words from file
        if count == 10:
            count = 1
            validation_strings.append(cleanedWord)
            validation_target.append(1)
        else:
            count = count + 1
            training_strings.append(cleanedWord)
            training_target.append(1)



# Convert strings to ordinal values
training_ordinal = []
validation_ordinal = []

for word in training_strings:
    ordinal_temp = []
    for char in word:
        ordinal_temp.append(ord(char))
    training_ordinal.append(ordinal_temp)

for word in validation_strings:
    ordinal_temp = []
    for char in word:
        ordinal_temp.append(ord(char))
    validation_ordinal.append(ordinal_temp)


# Training data/label pair respectively
x_train = np.asarray(x_training, dtype=np.float32)
y_train = np.asarray(y_training, dtype=np.float32)

# Test data/label pair respectively
x_test = np.asarray(x_testing, dtype=np.float32)
y_test = np.asarray(y_testing, dtype=np.float32)

# Reshape arrays to fit model
x_train = x_train.reshape(x_train.shape[0], 57, 1)
x_test = x_test.reshape(x_test.shape[0], 57, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

######################
#     Build Model    #
######################

model = Sequential()
model.add(Flatten(input_shape=(57, 1)))

model.add(Dense(units=48, activation='sigmoid'))
model.add(Dense(units=48, activation='sigmoid'))
model.add(Dense(units=48, activation='sigmoid'))
model.add(Dense(units=48, activation='sigmoid'))
model.add(Dense(units=48, activation='sigmoid'))
model.add(Dense(units=48, activation='sigmoid'))

model.add(Dense(units=2, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(x_train, y_train,
          validation_data=[x_test, y_test],
          batch_size=500,
          epochs=250,
          verbose=verbose_level)

score = model.evaluate(x_test, y_test, verbose=0)

# Print Total accuracy and loss
print('Test loss:', score[0])
print('Test accuracy:', score[1])


"""