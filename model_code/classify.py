# Static Analysis Lab Solution
# Written by Carson Woods
# University of Tennessee at Chattanooga
# 2020

# This is not the only way that the lab can be accomplished
# though this solution yeilds 99% accuracy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend
import numpy as np
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


print("Loading Memory Leak Data...")
with open("../json_data/memory_leak.json", 'r', encoding = "ISO-8859-1") as f:
    memory_leak_json = json.loads(f.read())
    for function in memory_leak_json["functions"]:
        if len(function["function"]) > 20:
            training_strings.append(function["function"][0:20])
            training_target.append(0);
    del memory_leak_json
print("Done.")


print("Loading Heap Overflow Data...")
with open("../json_data/heap_overflow.json", 'r', encoding = "ISO-8859-1") as f:
    heap_overflow_json = json.loads(f.read())
    for function in heap_overflow_json["functions"]:
        if len(function["function"]) > 20:
            training_strings.append(function["function"][0:20])
            training_target.append(1);
    del heap_overflow_json
print("Done.")

print("Loading Stack Overflow Data...")
with open("../json_data/stack_overflow.json", 'r', encoding = "ISO-8859-1") as f:
    stack_overflow_json = json.loads(f.read())
    for function in stack_overflow_json["functions"]:
        if len(function["function"]) > 20:
            training_strings.append(function["function"][0:20])
            training_target.append(2);
    del stack_overflow_json
print("Done.")

print("Loading Command Injection Data...")
with open("../json_data/command_injection.json", 'r', encoding = "ISO-8859-1") as f:
    command_injection_json = json.loads(f.read())
    for function in command_injection_json["functions"]:
        if len(function["function"]) > 20:
            training_strings.append(function["function"][0:20])
            training_target.append(3);
    del command_injection_json
print("Done.")


count = 1
for i, element in enumerate(training_strings):
    if count == 10:
        validation_strings.append(training_strings.pop(i))
        validation_target.append(training_target.pop(i))
        count = 0
    count += 1

"""
print(len(training_strings))
print(len(training_target))
print(len(validation_strings))
print(len(validation_target))
"""

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

# Free up unneeded variables to curb memory consumption
del ordinal_temp
del training_strings
del validation_strings

# Convert list to numpy array
x_train = np.array(training_ordinal)
x_test = np.array(validation_ordinal)
y_train = np.array(training_target)
y_test = np.array(validation_target)

# Free up unneeded variables to curb memory consumption
del training_target
del validation_target

# Reshape data
x_train = x_train.reshape(x_train.shape[0], 20, 1)
x_test = x_test.reshape(x_test.shape[0], 20, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 4)
y_test = keras.utils.to_categorical(y_test, 4)


######################
#     Build Model    #
######################

model = Sequential()
model.add(Flatten(input_shape=(20, 1)))

model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=4, activation='softmax'))


model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=500,
          epochs=250,
          verbose=verbose_level)

score = model.evaluate(x_test, y_test, verbose=verbose_level)

# Print Total accuracy and loss
print('Test loss:', score[0])
print('Test accuracy:', score[1])
