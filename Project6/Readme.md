~~~python
# Define the model
model = Sequential()

model.add(Convolution2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3)))# 32
model.add(Activation('relu'))
model.add(Convolution2D(48, 3, 3))#  30
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(10, 1, 1, activation='relu')) #30
model.add(MaxPooling2D(pool_size=(2, 2))) #15
model.add(Convolution2D(96, 3, 3, border_mode='same')) #15
model.add(Activation('relu')) 
model.add(Convolution2D(96, 3, 3))  # 13
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(10, 1, 1, activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) #6
model.add(Convolution2D(192, 3, 3, border_mode='same')) #6
model.add(Activation('relu'))
model.add(Convolution2D(192, 3, 3)) # 4
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #2
model.add(Convolution2D(10, 2))
model.add(Flatten())
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
~~~

#### Model 1 

Dense layer was removed, Altered Dropouts, Inserted 1x1 for kernel scale up 

---



#### Model 2

~~~python
# Define the model
model = Sequential()

model.add(Convolution2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3)))# 32
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(48, 3, 3))#  30
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Convolution2D(10, 1, 1, activation='relu')) #30
model.add(MaxPooling2D(pool_size=(2, 2))) #15
model.add(Convolution2D(96, 3, 3, border_mode='same')) #15
model.add(Activation('relu')) 
model.add(BatchNormalization())
model.add(Convolution2D(96, 3, 3))  # 13
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Convolution2D(10, 1, 1, activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) #6
model.add(Convolution2D(192, 3, 3, border_mode='same')) #6
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(192, 3, 3)) # 4
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))# Added DropOut here
model.add(MaxPooling2D(pool_size=(2, 2))) #2
model.add(Convolution2D(10, 2))
model.add(Flatten())
model.add(Activation('softmax'))

#Added Learning Rate
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.003 * 1/(1 + 0.319 * epoch), 10)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.003), metrics=['accuracy'])
~~~

Still Overfitting too much and Loss is too much and Val acc caps at 75 % will add more kernels next models and add layers

~~~txt

Epoch 1/50
390/390 [==============================] - 34s 87ms/step - loss: 1.6387 - acc: 0.4612 - val_loss: 2.1994 - val_acc: 0.4110
Epoch 2/50
390/390 [==============================] - 28s 73ms/step - loss: 1.2109 - acc: 0.5959 - val_loss: 1.1141 - val_acc: 0.6139
Epoch 3/50
390/390 [==============================] - 28s 73ms/step - loss: 1.0021 - acc: 0.6569 - val_loss: 1.2570 - val_acc: 0.5723
Epoch 4/50
390/390 [==============================] - 28s 73ms/step - loss: 0.8793 - acc: 0.6935 - val_loss: 0.8791 - val_acc: 0.6976
Epoch 5/50
390/390 [==============================] - 28s 73ms/step - loss: 0.7716 - acc: 0.7281 - val_loss: 1.2168 - val_acc: 0.6099
Epoch 6/50
390/390 [==============================] - 28s 73ms/step - loss: 0.7015 - acc: 0.7561 - val_loss: 0.9393 - val_acc: 0.6991
Epoch 7/50
390/390 [==============================] - 28s 72ms/step - loss: 0.6473 - acc: 0.7729 - val_loss: 0.7533 - val_acc: 0.7407
Epoch 8/50
390/390 [==============================] - 28s 73ms/step - loss: 0.6045 - acc: 0.7859 - val_loss: 0.7128 - val_acc: 0.7574
Epoch 9/50
390/390 [==============================] - 28s 73ms/step - loss: 0.5619 - acc: 0.8023 - val_loss: 0.7294 - val_acc: 0.7553
Epoch 10/50
390/390 [==============================] - 28s 73ms/step - loss: 0.5369 - acc: 0.8119 - val_loss: 0.8538 - val_acc: 0.7239
Epoch 11/50
390/390 [==============================] - 28s 73ms/step - loss: 0.4996 - acc: 0.8244 - val_loss: 0.9576 - val_acc: 0.7029
Epoch 12/50
390/390 [==============================] - 28s 73ms/step - loss: 0.4734 - acc: 0.8328 - val_loss: 0.7290 - val_acc: 0.7607
Epoch 13/50
390/390 [==============================] - 28s 73ms/step - loss: 0.4507 - acc: 0.8409 - val_loss: 0.8170 - val_acc: 0.7478
Epoch 14/50
390/390 [==============================] - 28s 73ms/step - loss: 0.4241 - acc: 0.8502 - val_loss: 0.7548 - val_acc: 0.7578
Epoch 15/50
390/390 [==============================] - 28s 72ms/step - loss: 0.4030 - acc: 0.8565 - val_loss: 0.8604 - val_acc: 0.7447
Epoch 16/50
390/390 [==============================] - 28s 72ms/step - loss: 0.3792 - acc: 0.8649 - val_loss: 0.8092 - val_acc: 0.7516
Epoch 17/50
390/390 [==============================] - 28s 73ms/step - loss: 0.3598 - acc: 0.8716 - val_loss: 0.7002 - val_acc: 0.7843
Epoch 18/50
390/390 [==============================] - 28s 72ms/step - loss: 0.3407 - acc: 0.8788 - val_loss: 0.6988 - val_acc: 0.7764
Epoch 19/50
390/390 [==============================] - 28s 72ms/step - loss: 0.3232 - acc: 0.8849 - val_loss: 0.8280 - val_acc: 0.7621
Epoch 20/50
390/390 [==============================] - 28s 72ms/step - loss: 0.3020 - acc: 0.8905 - val_loss: 0.7487 - val_acc: 0.7738
Epoch 21/50
390/390 [==============================] - 28s 73ms/step - loss: 0.2932 - acc: 0.8942 - val_loss: 0.7312 - val_acc: 0.7798
Epoch 22/50
390/390 [==============================] - 28s 73ms/step - loss: 0.2769 - acc: 0.9015 - val_loss: 0.7781 - val_acc: 0.7767
Epoch 23/50
390/390 [==============================] - 28s 72ms/step - loss: 0.2601 - acc: 0.9060 - val_loss: 0.7885 - val_acc: 0.7649
Epoch 24/50
390/390 [==============================] - 28s 73ms/step - loss: 0.2489 - acc: 0.9120 - val_loss: 0.8518 - val_acc: 0.7639
Epoch 25/50
390/390 [==============================] - 28s 73ms/step - loss: 0.2368 - acc: 0.9161 - val_loss: 0.8228 - val_acc: 0.7699
Epoch 26/50
390/390 [==============================] - 28s 73ms/step - loss: 0.2373 - acc: 0.9138 - val_loss: 0.7175 - val_acc: 0.7866
Epoch 27/50
390/390 [==============================] - 28s 72ms/step - loss: 0.2240 - acc: 0.9200 - val_loss: 0.8716 - val_acc: 0.7731
Epoch 28/50
390/390 [==============================] - 28s 73ms/step - loss: 0.2131 - acc: 0.9235 - val_loss: 0.8263 - val_acc: 0.7776
Epoch 29/50
390/390 [==============================] - 28s 73ms/step - loss: 0.2044 - acc: 0.9270 - val_loss: 0.8279 - val_acc: 0.7884
Epoch 30/50
390/390 [==============================] - 28s 73ms/step - loss: 0.2043 - acc: 0.9260 - val_loss: 0.9083 - val_acc: 0.7604
Epoch 31/50
390/390 [==============================] - 28s 73ms/step - loss: 0.1926 - acc: 0.9301 - val_loss: 0.8054 - val_acc: 0.7909
Epoch 32/50
390/390 [==============================] - 28s 73ms/step - loss: 0.1798 - acc: 0.9354 - val_loss: 0.8419 - val_acc: 0.7749
Epoch 33/50
390/390 [==============================] - 28s 73ms/step - loss: 0.1837 - acc: 0.9337 - val_loss: 0.8448 - val_acc: 0.7872
Epoch 34/50
390/390 [==============================] - 28s 73ms/step - loss: 0.1707 - acc: 0.9382 - val_loss: 0.7970 - val_acc: 0.7861
Epoch 35/50
390/390 [==============================] - 28s 72ms/step - loss: 0.1740 - acc: 0.9378 - val_loss: 1.0913 - val_acc: 0.7553
Epoch 36/50
390/390 [==============================] - 28s 73ms/step - loss: 0.1623 - acc: 0.9418 - val_loss: 0.8851 - val_acc: 0.7882
Epoch 37/50
390/390 [==============================] - 28s 72ms/step - loss: 0.1601 - acc: 0.9426 - val_loss: 0.9438 - val_acc: 0.7728
Epoch 38/50
390/390 [==============================] - 28s 72ms/step - loss: 0.1499 - acc: 0.9457 - val_loss: 0.8644 - val_acc: 0.7905
Epoch 39/50
390/390 [==============================] - 28s 73ms/step - loss: 0.1537 - acc: 0.9461 - val_loss: 0.8453 - val_acc: 0.7993
Epoch 40/50
390/390 [==============================] - 28s 73ms/step - loss: 0.1486 - acc: 0.9465 - val_loss: 0.9152 - val_acc: 0.7906
Epoch 41/50
390/390 [==============================] - 28s 73ms/step - loss: 0.1445 - acc: 0.9479 - val_loss: 0.9319 - val_acc: 0.7839
Epoch 42/50
390/390 [==============================] - 28s 72ms/step - loss: 0.1441 - acc: 0.9493 - val_loss: 0.8821 - val_acc: 0.7836
Epoch 43/50
390/390 [==============================] - 28s 72ms/step - loss: 0.1395 - acc: 0.9500 - val_loss: 0.9630 - val_acc: 0.7683
Epoch 44/50
390/390 [==============================] - 28s 73ms/step - loss: 0.1368 - acc: 0.9507 - val_loss: 0.9073 - val_acc: 0.7881
Epoch 45/50
390/390 [==============================] - 28s 73ms/step - loss: 0.1343 - acc: 0.9524 - val_loss: 0.8670 - val_acc: 0.7991
Epoch 46/50
390/390 [==============================] - 28s 73ms/step - loss: 0.1334 - acc: 0.9520 - val_loss: 0.9133 - val_acc: 0.7924
Epoch 47/50
390/390 [==============================] - 28s 73ms/step - loss: 0.1297 - acc: 0.9540 - val_loss: 0.9065 - val_acc: 0.7899
Epoch 48/50
390/390 [==============================] - 28s 72ms/step - loss: 0.1292 - acc: 0.9542 - val_loss: 0.9242 - val_acc: 0.7931
Epoch 49/50
390/390 [==============================] - 28s 73ms/step - loss: 0.1228 - acc: 0.9564 - val_loss: 0.9585 - val_acc: 0.7845
Epoch 50/50
390/390 [==============================] - 28s 73ms/step - loss: 0.1277 - acc: 0.9546 - val_loss: 0.9676 - val_acc: 0.7898
Model took 1422.80 seconds to train

~~~

Visiable difference by adding steps and BN. Will add more features in next model and increase epochs

---

#### Model 3



~~~Python
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(32, 32, 3)))# 32
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Convolution2D(64, 3, 3))#  30
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3,, border_mode='same'))#  30
model.add(Activation('relu'))


model.add(MaxPooling2D(pool_size=(2, 2))) #15

model.add(Dropout(0.25))
model.add(Convolution2D(96, 3, 3, border_mode='same')) #15
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Convolution2D(128, 3, 3))  # 13
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(128, 3, 3,border_mode='same'))  # 13
model.add(MaxPooling2D(pool_size=(2, 2))) #6

model.add(Dropout(0.25))
model.add(Convolution2D(192, 3, 3, border_mode='same')) #6
model.add(Activation('relu'))
model.add(Convolution2D(256, 3, 3)) # 4
model.add(Activation('relu'))
model.add(Convolution2D(512, 3, 3)) # 2
# model.add(MaxPooling2D(pool_size=(2, 2))) #2 

model.add(Dropout(0.25))


model.add(Convolution2D(10, 2))
model.add(Flatten())
model.add(Activation('softmax'))


from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.007 * 1/(1 + 0.319 * epoch), 10)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.003), metrics=['accuracy'])

~~~

~~~txt
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1033: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1020: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.

Epoch 1/100
390/390 [==============================] - 44s 112ms/step - loss: 14.4857 - acc: 0.0994 - val_loss: 14.5063 - val_acc: 0.1000
Epoch 2/100
390/390 [==============================] - 37s 94ms/step - loss: 14.5060 - acc: 0.1000 - val_loss: 14.5063 - val_acc: 0.1000
Epoch 3/100
390/390 [==============================] - 37s 95ms/step - loss: 14.5060 - acc: 0.1000 - val_loss: 14.5063 - val_acc: 0.1000
Epoch 4/100
390/390 [==============================] - 37s 94ms/step - loss: 14.5067 - acc: 0.1000 - val_loss: 14.5063 - val_acc: 0.1000
Epoch 5/100
390/390 [==============================] - 37s 94ms/step - loss: 14.5042 - acc: 0.1001 - val_loss: 14.5063 - val_acc: 0.1000
Epoch 6/100
390/390 [==============================] - 37s 94ms/step - loss: 14.5089 - acc: 0.0998 - val_loss: 14.5063 - val_acc: 0.1000
Epoch 7/100
390/390 [==============================] - 37s 94ms/step - loss: 14.5017 - acc: 0.1003 - val_loss: 14.5063 - val_acc: 0.1000
Epoch 8/100
390/390 [==============================] - 37s 94ms/step - loss: 14.5069 - acc: 0.1000 - val_loss: 14.5063 - val_acc: 0.1000
Epoch 9/100
390/390 [==============================] - 37s 94ms/step - loss: 14.5146 - acc: 0.0995 - val_loss: 14.5063 - val_acc: 0.1000
Epoch 10/100
390/390 [==============================] - 37s 94ms/step - loss: 14.4941 - acc: 0.1008 - val_loss: 14.5063 - val_acc: 0.1000
Epoch 11/100
249/390 [==================>...........] - ETA: 12s - loss: 14.5432 - acc: 0.0977
~~~

Accuracy just became shit, removing `border_mode='same`

___

#### MODEL 5 

Removed `border_mode =same` and altered RF a little bit.

~~~python
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(32, 32, 3)))# 32
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Convolution2D(64, 3, 3))#  30
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))#  28
model.add(Activation('relu'))


model.add(MaxPooling2D(pool_size=(2, 2))) #14

model.add(Dropout(0.25))
model.add(Convolution2D(96, 3, 3, border_mode='same')) #14
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Convolution2D(128, 3, 3))  # 13
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(128, 3, 3))  # 11
model.add(MaxPooling2D(pool_size=(2, 2))) #5

model.add(Dropout(0.25))
model.add(Convolution2D(192, 3, 3, border_mode='same')) #5
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(256, 3, 3)) # 3
model.add(Activation('relu'))

# model.add(Convolution2D(512, 3, 3)) # 2
# model.add(MaxPooling2D(pool_size=(2, 2))) #2 

# model.add(Dropout(0.25))


model.add(Convolution2D(10, 2))
model.add(Flatten())
model.add(Activation('softmax'))


from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.007 * 1/(1 + 0.319 * epoch), 10)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.007), metrics=['accuracy'])

~~~

Comparatively  a very good model. Was able to get base accuracy of 82.41%

~~~
Epoch 1/100
390/390 [==============================] - 38s 98ms/step - loss: 2.1759 - acc: 0.2615 - val_loss: 1.8302 - val_acc: 0.3076
Epoch 2/100
390/390 [==============================] - 32s 83ms/step - loss: 1.6033 - acc: 0.4064 - val_loss: 1.6031 - val_acc: 0.4070
Epoch 3/100
390/390 [==============================] - 32s 82ms/step - loss: 1.3550 - acc: 0.5090 - val_loss: 1.4218 - val_acc: 0.4997
Epoch 4/100
390/390 [==============================] - 32s 82ms/step - loss: 1.2163 - acc: 0.5605 - val_loss: 1.2880 - val_acc: 0.5561
Epoch 5/100
390/390 [==============================] - 32s 82ms/step - loss: 1.1040 - acc: 0.6030 - val_loss: 1.1245 - val_acc: 0.5984
Epoch 6/100
390/390 [==============================] - 32s 82ms/step - loss: 0.9856 - acc: 0.6467 - val_loss: 1.1748 - val_acc: 0.6041
Epoch 7/100
390/390 [==============================] - 32s 82ms/step - loss: 0.9320 - acc: 0.6681 - val_loss: 1.1313 - val_acc: 0.6225
Epoch 8/100
390/390 [==============================] - 32s 82ms/step - loss: 0.8308 - acc: 0.7062 - val_loss: 0.8923 - val_acc: 0.6927
Epoch 9/100
390/390 [==============================] - 32s 82ms/step - loss: 0.7756 - acc: 0.7233 - val_loss: 0.9688 - val_acc: 0.6871
Epoch 10/100
390/390 [==============================] - 32s 82ms/step - loss: 0.7503 - acc: 0.7359 - val_loss: 3.6339 - val_acc: 0.4926
Epoch 11/100
390/390 [==============================] - 32s 82ms/step - loss: 1.1285 - acc: 0.6272 - val_loss: 1.4927 - val_acc: 0.4962
Epoch 12/100
390/390 [==============================] - 32s 82ms/step - loss: 1.0373 - acc: 0.6341 - val_loss: 1.0775 - val_acc: 0.6213
Epoch 13/100
390/390 [==============================] - 32s 82ms/step - loss: 1.2488 - acc: 0.5565 - val_loss: 1.1248 - val_acc: 0.6073
Epoch 14/100
390/390 [==============================] - 32s 82ms/step - loss: 1.0151 - acc: 0.6401 - val_loss: 0.9332 - val_acc: 0.6728
Epoch 15/100
390/390 [==============================] - 32s 82ms/step - loss: 0.9089 - acc: 0.6775 - val_loss: 0.8553 - val_acc: 0.7000
Epoch 16/100
390/390 [==============================] - 32s 82ms/step - loss: 0.8469 - acc: 0.7002 - val_loss: 0.8667 - val_acc: 0.7020
Epoch 17/100
390/390 [==============================] - 32s 82ms/step - loss: 0.7954 - acc: 0.7186 - val_loss: 0.8747 - val_acc: 0.7005
Epoch 18/100
390/390 [==============================] - 32s 83ms/step - loss: 0.7591 - acc: 0.7322 - val_loss: 0.8084 - val_acc: 0.7208
Epoch 19/100
390/390 [==============================] - 32s 82ms/step - loss: 0.8642 - acc: 0.6980 - val_loss: 0.9799 - val_acc: 0.6763
Epoch 20/100
390/390 [==============================] - 32s 82ms/step - loss: 0.7555 - acc: 0.7341 - val_loss: 0.9119 - val_acc: 0.6922
Epoch 21/100
390/390 [==============================] - 32s 82ms/step - loss: 0.7048 - acc: 0.7544 - val_loss: 1.1401 - val_acc: 0.6259
Epoch 22/100
390/390 [==============================] - 32s 82ms/step - loss: 0.7076 - acc: 0.7539 - val_loss: 0.7613 - val_acc: 0.7527
Epoch 23/100
390/390 [==============================] - 32s 82ms/step - loss: 0.6651 - acc: 0.7683 - val_loss: 0.7250 - val_acc: 0.7533
Epoch 24/100
390/390 [==============================] - 32s 82ms/step - loss: 0.6341 - acc: 0.7785 - val_loss: 0.7669 - val_acc: 0.7471
Epoch 25/100
390/390 [==============================] - 32s 82ms/step - loss: 0.6563 - acc: 0.7708 - val_loss: 0.7262 - val_acc: 0.7530
Epoch 26/100
390/390 [==============================] - 32s 81ms/step - loss: 0.6091 - acc: 0.7889 - val_loss: 0.8200 - val_acc: 0.7556
Epoch 27/100
390/390 [==============================] - 32s 81ms/step - loss: 0.6018 - acc: 0.7895 - val_loss: 0.6889 - val_acc: 0.7729
Epoch 28/100
390/390 [==============================] - 32s 81ms/step - loss: 0.6731 - acc: 0.7671 - val_loss: 1.0195 - val_acc: 0.6708
Epoch 29/100
390/390 [==============================] - 32s 82ms/step - loss: 0.6336 - acc: 0.7777 - val_loss: 0.7272 - val_acc: 0.7605
Epoch 30/100
390/390 [==============================] - 32s 82ms/step - loss: 0.5714 - acc: 0.8013 - val_loss: 0.7196 - val_acc: 0.7688
Epoch 31/100
390/390 [==============================] - 32s 82ms/step - loss: 0.5447 - acc: 0.8105 - val_loss: 0.7354 - val_acc: 0.7613
Epoch 32/100
390/390 [==============================] - 32s 82ms/step - loss: 0.5334 - acc: 0.8151 - val_loss: 0.6511 - val_acc: 0.7845
Epoch 33/100
390/390 [==============================] - 32s 82ms/step - loss: 0.5681 - acc: 0.8029 - val_loss: 0.9538 - val_acc: 0.7074
Epoch 34/100
390/390 [==============================] - 32s 83ms/step - loss: 0.5236 - acc: 0.8178 - val_loss: 0.6277 - val_acc: 0.7939
Epoch 35/100
390/390 [==============================] - 32s 83ms/step - loss: 0.4941 - acc: 0.8275 - val_loss: 0.6775 - val_acc: 0.7881
Epoch 36/100
390/390 [==============================] - 32s 82ms/step - loss: 0.4847 - acc: 0.8315 - val_loss: 3.3568 - val_acc: 0.6066
Epoch 37/100
390/390 [==============================] - 32s 82ms/step - loss: 0.4748 - acc: 0.8338 - val_loss: 0.6438 - val_acc: 0.7932
Epoch 38/100
390/390 [==============================] - 32s 82ms/step - loss: 0.4922 - acc: 0.8290 - val_loss: 0.6425 - val_acc: 0.7915
Epoch 39/100
390/390 [==============================] - 32s 82ms/step - loss: 0.4559 - acc: 0.8418 - val_loss: 0.6798 - val_acc: 0.7897
Epoch 40/100
390/390 [==============================] - 32s 82ms/step - loss: 0.4444 - acc: 0.8453 - val_loss: 0.7282 - val_acc: 0.7755
Epoch 41/100
390/390 [==============================] - 32s 82ms/step - loss: 0.4361 - acc: 0.8489 - val_loss: 0.6618 - val_acc: 0.7915
Epoch 42/100
390/390 [==============================] - 32s 82ms/step - loss: 0.4932 - acc: 0.8316 - val_loss: 0.7261 - val_acc: 0.7817
Epoch 43/100
390/390 [==============================] - 32s 82ms/step - loss: 0.4220 - acc: 0.8510 - val_loss: 0.6485 - val_acc: 0.8014
Epoch 44/100
390/390 [==============================] - 32s 82ms/step - loss: 0.4120 - acc: 0.8577 - val_loss: 0.6719 - val_acc: 0.7976
Epoch 45/100
390/390 [==============================] - 32s 82ms/step - loss: 0.4290 - acc: 0.8534 - val_loss: 3.3293 - val_acc: 0.3381
Epoch 46/100
390/390 [==============================] - 32s 82ms/step - loss: 0.4722 - acc: 0.8370 - val_loss: 0.6325 - val_acc: 0.7938
Epoch 47/100
390/390 [==============================] - 32s 82ms/step - loss: 0.3926 - acc: 0.8631 - val_loss: 0.7133 - val_acc: 0.7834
Epoch 48/100
390/390 [==============================] - 32s 83ms/step - loss: 0.3848 - acc: 0.8655 - val_loss: 0.6821 - val_acc: 0.7964
Epoch 49/100
390/390 [==============================] - 32s 83ms/step - loss: 0.3659 - acc: 0.8720 - val_loss: 0.6849 - val_acc: 0.7992
Epoch 50/100
390/390 [==============================] - 32s 82ms/step - loss: 0.3727 - acc: 0.8704 - val_loss: 0.6803 - val_acc: 0.8024
Epoch 51/100
390/390 [==============================] - 32s 83ms/step - loss: 0.3468 - acc: 0.8790 - val_loss: 0.6588 - val_acc: 0.8082
Epoch 52/100
390/390 [==============================] - 33s 83ms/step - loss: 0.3441 - acc: 0.8795 - val_loss: 0.7029 - val_acc: 0.8084
Epoch 53/100
390/390 [==============================] - 32s 83ms/step - loss: 0.3377 - acc: 0.8821 - val_loss: 0.6434 - val_acc: 0.8083
Epoch 54/100
390/390 [==============================] - 32s 83ms/step - loss: 0.3296 - acc: 0.8850 - val_loss: 0.7000 - val_acc: 0.8021
Epoch 55/100
390/390 [==============================] - 32s 83ms/step - loss: 0.3209 - acc: 0.8868 - val_loss: 0.7039 - val_acc: 0.8022
Epoch 56/100
390/390 [==============================] - 32s 83ms/step - loss: 0.3159 - acc: 0.8890 - val_loss: 0.8281 - val_acc: 0.7826
Epoch 57/100
390/390 [==============================] - 33s 83ms/step - loss: 0.3088 - acc: 0.8911 - val_loss: 0.6707 - val_acc: 0.8144
Epoch 58/100
390/390 [==============================] - 32s 83ms/step - loss: 0.3080 - acc: 0.8914 - val_loss: 0.7147 - val_acc: 0.8073
Epoch 59/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2977 - acc: 0.8968 - val_loss: 0.8854 - val_acc: 0.7854
Epoch 60/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2912 - acc: 0.8989 - val_loss: 0.6704 - val_acc: 0.8162
Epoch 61/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2810 - acc: 0.9015 - val_loss: 1.0372 - val_acc: 0.7799
Epoch 62/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2782 - acc: 0.9030 - val_loss: 0.7304 - val_acc: 0.8034
Epoch 63/100
390/390 [==============================] - 33s 83ms/step - loss: 0.2821 - acc: 0.9007 - val_loss: 0.8196 - val_acc: 0.7893
Epoch 64/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2734 - acc: 0.9037 - val_loss: 0.6932 - val_acc: 0.8164
Epoch 65/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2681 - acc: 0.9071 - val_loss: 0.7943 - val_acc: 0.7922
Epoch 66/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2614 - acc: 0.9085 - val_loss: 0.7130 - val_acc: 0.8128
Epoch 67/100
390/390 [==============================] - 33s 83ms/step - loss: 0.2544 - acc: 0.9114 - val_loss: 0.7806 - val_acc: 0.8136
Epoch 68/100
390/390 [==============================] - 33s 84ms/step - loss: 0.2462 - acc: 0.9142 - val_loss: 0.6802 - val_acc: 0.8228
Epoch 69/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2386 - acc: 0.9162 - val_loss: 0.7677 - val_acc: 0.8019
Epoch 70/100
390/390 [==============================] - 33s 84ms/step - loss: 0.2457 - acc: 0.9132 - val_loss: 0.8922 - val_acc: 0.7777
Epoch 71/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2359 - acc: 0.9174 - val_loss: 0.7865 - val_acc: 0.8056
Epoch 72/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2382 - acc: 0.9165 - val_loss: 0.7111 - val_acc: 0.8149
Epoch 73/100
390/390 [==============================] - 33s 83ms/step - loss: 0.2265 - acc: 0.9206 - val_loss: 0.7470 - val_acc: 0.8103
Epoch 74/100
390/390 [==============================] - 33s 83ms/step - loss: 0.2392 - acc: 0.9159 - val_loss: 0.7480 - val_acc: 0.8177
Epoch 75/100
390/390 [==============================] - 33s 84ms/step - loss: 0.3675 - acc: 0.8747 - val_loss: 0.8084 - val_acc: 0.7931
Epoch 76/100
390/390 [==============================] - 33s 83ms/step - loss: 0.2643 - acc: 0.9074 - val_loss: 0.6770 - val_acc: 0.8214
Epoch 77/100
390/390 [==============================] - 33s 84ms/step - loss: 0.2259 - acc: 0.9187 - val_loss: 0.7836 - val_acc: 0.8122
Epoch 78/100
390/390 [==============================] - 33s 83ms/step - loss: 0.2121 - acc: 0.9253 - val_loss: 0.7178 - val_acc: 0.8218
Epoch 79/100
390/390 [==============================] - 33s 84ms/step - loss: 0.2103 - acc: 0.9262 - val_loss: 0.7624 - val_acc: 0.8140
Epoch 80/100
390/390 [==============================] - 33s 84ms/step - loss: 0.2086 - acc: 0.9272 - val_loss: 0.7435 - val_acc: 0.8226
Epoch 81/100
390/390 [==============================] - 33s 83ms/step - loss: 0.2065 - acc: 0.9292 - val_loss: 0.7501 - val_acc: 0.8241
Epoch 82/100
390/390 [==============================] - 33s 83ms/step - loss: 0.2000 - acc: 0.9296 - val_loss: 0.7842 - val_acc: 0.8116
Epoch 83/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2038 - acc: 0.9287 - val_loss: 0.8612 - val_acc: 0.8046
Epoch 84/100
390/390 [==============================] - 32s 83ms/step - loss: 0.1949 - acc: 0.9325 - val_loss: 0.7887 - val_acc: 0.8167
Epoch 85/100
390/390 [==============================] - 32s 83ms/step - loss: 0.1928 - acc: 0.9327 - val_loss: 0.9458 - val_acc: 0.7960
Epoch 86/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2009 - acc: 0.9296 - val_loss: 0.7693 - val_acc: 0.8153
Epoch 87/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2943 - acc: 0.9012 - val_loss: 1.4160 - val_acc: 0.6620
Epoch 88/100
390/390 [==============================] - 32s 83ms/step - loss: 0.3190 - acc: 0.8905 - val_loss: 0.7134 - val_acc: 0.8184
Epoch 89/100
390/390 [==============================] - 32s 83ms/step - loss: 0.5221 - acc: 0.8198 - val_loss: 0.6466 - val_acc: 0.7988
Epoch 90/100
390/390 [==============================] - 33s 83ms/step - loss: 0.3274 - acc: 0.8838 - val_loss: 0.6697 - val_acc: 0.8154
Epoch 91/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2552 - acc: 0.9096 - val_loss: 0.6479 - val_acc: 0.8243
Epoch 92/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2188 - acc: 0.9227 - val_loss: 0.7185 - val_acc: 0.8223
Epoch 93/100
390/390 [==============================] - 32s 83ms/step - loss: 0.2028 - acc: 0.9282 - val_loss: 0.7753 - val_acc: 0.8160
Epoch 94/100
390/390 [==============================] - 32s 82ms/step - loss: 0.1997 - acc: 0.9303 - val_loss: 0.8115 - val_acc: 0.8177
Epoch 95/100
390/390 [==============================] - 32s 82ms/step - loss: 0.1847 - acc: 0.9357 - val_loss: 0.8248 - val_acc: 0.8247
Epoch 96/100
390/390 [==============================] - 32s 82ms/step - loss: 0.1855 - acc: 0.9357 - val_loss: 0.8255 - val_acc: 0.8208
Epoch 97/100
390/390 [==============================] - 32s 82ms/step - loss: 0.1764 - acc: 0.9386 - val_loss: 0.7951 - val_acc: 0.8222
Epoch 98/100
390/390 [==============================] - 32s 82ms/step - loss: 0.1726 - acc: 0.9403 - val_loss: 0.7783 - val_acc: 0.8251
Epoch 99/100
390/390 [==============================] - 32s 82ms/step - loss: 0.1772 - acc: 0.9393 - val_loss: 0.7996 - val_acc: 0.8179
Epoch 100/100
390/390 [==============================] - 32s 82ms/step - loss: 0.1783 - acc: 0.9380 - val_loss: 0.8796 - val_acc: 0.8211
Model took 3226.68 seconds to train
~~~

![](C:\Users\level\Documents\GitHub\EVA3\Images\Accuracy vs Loss.png)







---

#### Model 6



~~~~python

~~~~

~~~txt
OUTPUTs
~~~

