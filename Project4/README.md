# Assignment 4 Breakdown

### Though Process.

>  Below is a result of aggressive testing over Assignment 3  without BN, LR  and DO

Over my experiment over vanilla NN architecture(Just conv, max pooling and $1*1$conv) I got pretty much similar result with  greater than 20K and below 20K parameters. 

Result with $ V_{acc} > 20K$ parameter vs $ V_{acc} < 20K$ parameters didn’t see much difference plus it was always overfitting results, which is not desired. 

 Which playing with feature and parameters under 15k I noticed skewness in result. The reason being the learning rate. The skewed bowl image shown in session 5 to explain batch normalization concept was visible in result of the network.   Since the bowl was skewed hence the image data was not properly distributed the result went up and down and in loop and took longer epochs to reach in the local minimum.

> After the use of BN, LR  and DO

Only after using Batch Normalization and Dropout  is saw some improvements over the network. 
Learning rate alteration further improved the network to help receive $V_{acc} > 99.4$ . 

All network on longer epochs over-trained themselves. Vanilla network did that from the start. With BN and DO it happened too before I reached validation accuracy of 99.4.  

With Mnist in mind which is a small dataset with one channel and small images with only 10 class to classify. There is a need of few  but rich features. So dedicating over 32 kernel is an over kill.

So before I built an architecture to help reach $V_{acc} \geq  99.4$ , I had to think about  size of the dataset, no of channel used in image, no of kernels to use, size of the image,  and no of classes to identify. As discussed in class a small dataset like **mnist** does not require a lot of kernels, since all it has are edges and gradient and not a lot of textures.  Image size was small and only one grayscale channel to work with. 



When does kernels receptive field start making sense of  primitive feature?

- It depends on image size, class, size of dataset, etc
- For  primitive dataset with one channel like mnist,  what  receptive field or how many convolution of 3*3  it would take to make sense of primitive features.
- For mnist, a 5*5  view of image should make sense of features.

![img](data:image/png;base64,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)

I decided to got with three layers before max pooling then 4 layers for high level features of till image size of 4.

|              Network               |
| :--------------------------------: |
|            Convolution             |
|            Convolution             |
| 1 * 1 Convolution -  Rich features |
|             MaxPooling             |
|            Convolution             |
|            Convolution             |
|            Convolution             |
|            Convolution             |

## 1. Vanilla Architecture

~~~python
from keras.layers import Activation
model = Sequential()
# Random large parametere generator. 
model.add(Convolution2D(16, 3, 3, activation='relu', input_shape=(28,28,1)))    #26
model.add(Convolution2D(32, 3, 3, activation='relu'))                           #24
model.add(Convolution2D(10, 1, 1, activation='relu'))                           #24
model.add(MaxPooling2D(pool_size=(2, 2)))                                       #12

model.add(Convolution2D(16, 3, 3, activation='relu'))                           #10
model.add(Convolution2D(32, 3, 3, activation='relu'))                           #8
model.add(Convolution2D(32, 3, 3, activation='relu'))                           #6
model.add(Convolution2D(64, 3, 3, activation='relu'))                           #4
model.add(Convolution2D(10, 4))


model.add(Flatten())
model.add(Activation('softmax'))


model.summary()
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=64, epochs=20, verbose=1, validation_data=(X_test, Y_test))
~~~

~~~txt
Model: "sequential_26"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_201 (Conv2D)          (None, 26, 26, 8)         80        
_________________________________________________________________
conv2d_202 (Conv2D)          (None, 24, 24, 16)        1168      
_________________________________________________________________
conv2d_203 (Conv2D)          (None, 24, 24, 10)        170       
_________________________________________________________________
max_pooling2d_26 (MaxPooling (None, 12, 12, 10)        0         
_________________________________________________________________
conv2d_204 (Conv2D)          (None, 10, 10, 8)         728       
_________________________________________________________________
conv2d_205 (Conv2D)          (None, 8, 8, 16)          1168      
_________________________________________________________________
conv2d_206 (Conv2D)          (None, 6, 6, 16)          2320      
_________________________________________________________________
conv2d_207 (Conv2D)          (None, 4, 4, 16)          2320      
_________________________________________________________________
conv2d_208 (Conv2D)          (None, 1, 1, 10)          2570      
_________________________________________________________________
flatten_26 (Flatten)         (None, 10)                0         
_________________________________________________________________
activation_26 (Activation)   (None, 10)                0         
=================================================================
Total params: 10,524
Trainable params: 10,524
Non-trainable params: 0

Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 17s 278us/step - loss: 1.1571 - acc: 0.6180 - val_loss: 0.3037 - val_acc: 0.9096
Epoch 2/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.2759 - acc: 0.9163 - val_loss: 0.1969 - val_acc: 0.9400
Epoch 3/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.1911 - acc: 0.9422 - val_loss: 0.2703 - val_acc: 0.9099
Epoch 4/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.1521 - acc: 0.9533 - val_loss: 0.1242 - val_acc: 0.9597
Epoch 5/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.1290 - acc: 0.9610 - val_loss: 0.1466 - val_acc: 0.9576
Epoch 6/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.1130 - acc: 0.9650 - val_loss: 0.0952 - val_acc: 0.9697
Epoch 7/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.1006 - acc: 0.9690 - val_loss: 0.0962 - val_acc: 0.9710
Epoch 8/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.0919 - acc: 0.9716 - val_loss: 0.0928 - val_acc: 0.9724
Epoch 9/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.0846 - acc: 0.9734 - val_loss: 0.0778 - val_acc: 0.9764
Epoch 10/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0799 - acc: 0.9752 - val_loss: 0.0791 - val_acc: 0.9761
Epoch 11/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.0744 - acc: 0.9770 - val_loss: 0.1024 - val_acc: 0.9700
Epoch 12/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0711 - acc: 0.9777 - val_loss: 0.0656 - val_acc: 0.9796
Epoch 13/20
60000/60000 [==============================] - 11s 183us/step - loss: 0.0679 - acc: 0.9789 - val_loss: 0.0666 - val_acc: 0.9779
Epoch 14/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.0637 - acc: 0.9803 - val_loss: 0.0702 - val_acc: 0.9794
Epoch 15/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0609 - acc: 0.9811 - val_loss: 0.0679 - val_acc: 0.9784
Epoch 16/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0582 - acc: 0.9820 - val_loss: 0.0622 - val_acc: 0.9798
Epoch 17/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0556 - acc: 0.9827 - val_loss: 0.0903 - val_acc: 0.9711
Epoch 18/20
60000/60000 [==============================] - 11s 179us/step - loss: 0.0535 - acc: 0.9832 - val_loss: 0.0598 - val_acc: 0.9823
Epoch 19/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0520 - acc: 0.9838 - val_loss: 0.0612 - val_acc: 0.9810
Epoch 20/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.0490 - acc: 0.9845 - val_loss: 0.0725 - val_acc: 0.9776
<keras.callbacks.History at 0x7faa39311c50>
~~~

## 2. Kernel Optimization to fit trainable parameters under 15K

~~~python
# Set up data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)                           
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_train = X_train.astype('float32')                                            
X_test = X_test.astype('float32') 
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)                                  
Y_test = np_utils.to_categorical(y_test, 10)

from keras.layers import Activation
model = Sequential()
 
model.add(Convolution2D(8, (3, 3), activation='relu', input_shape=(28,28,1)))    #26
model.add(Convolution2D(16, (3, 3), activation='relu'))                           #24
model.add(Convolution2D(10, (1, 1), activation='relu'))                           #24

model.add(MaxPooling2D(pool_size=(2, 2)))                                       #12

model.add(Convolution2D(8, (3, 3), activation='relu'))                           #10
model.add(Convolution2D(16, (3, 3), activation='relu'))                           #8
model.add(Convolution2D(16, (3, 3), activation='relu'))                           #6
model.add(Convolution2D(16, (3, 3), activation='relu') )                          #4
model.add(Convolution2D(10, 4))        
model.add(Flatten())
model.add(Activation('softmax'))


model.summary()
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
~~~

~~~txt
Model: "sequential_26"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_201 (Conv2D)          (None, 26, 26, 8)         80        
_________________________________________________________________
conv2d_202 (Conv2D)          (None, 24, 24, 16)        1168      
_________________________________________________________________
conv2d_203 (Conv2D)          (None, 24, 24, 10)        170       
_________________________________________________________________
max_pooling2d_26 (MaxPooling (None, 12, 12, 10)        0         
_________________________________________________________________
conv2d_204 (Conv2D)          (None, 10, 10, 8)         728       
_________________________________________________________________
conv2d_205 (Conv2D)          (None, 8, 8, 16)          1168      
_________________________________________________________________
conv2d_206 (Conv2D)          (None, 6, 6, 16)          2320      
_________________________________________________________________
conv2d_207 (Conv2D)          (None, 4, 4, 16)          2320      
_________________________________________________________________
conv2d_208 (Conv2D)          (None, 1, 1, 10)          2570      
_________________________________________________________________
flatten_26 (Flatten)         (None, 10)                0         
_________________________________________________________________
activation_26 (Activation)   (None, 10)                0         
=================================================================
Total params: 10,524
Trainable params: 10,524
Non-trainable params: 0

Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 17s 278us/step - loss: 1.1571 - acc: 0.6180 - val_loss: 0.3037 - val_acc: 0.9096
Epoch 2/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.2759 - acc: 0.9163 - val_loss: 0.1969 - val_acc: 0.9400
Epoch 3/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.1911 - acc: 0.9422 - val_loss: 0.2703 - val_acc: 0.9099
Epoch 4/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.1521 - acc: 0.9533 - val_loss: 0.1242 - val_acc: 0.9597
Epoch 5/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.1290 - acc: 0.9610 - val_loss: 0.1466 - val_acc: 0.9576
Epoch 6/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.1130 - acc: 0.9650 - val_loss: 0.0952 - val_acc: 0.9697
Epoch 7/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.1006 - acc: 0.9690 - val_loss: 0.0962 - val_acc: 0.9710
Epoch 8/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.0919 - acc: 0.9716 - val_loss: 0.0928 - val_acc: 0.9724
Epoch 9/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.0846 - acc: 0.9734 - val_loss: 0.0778 - val_acc: 0.9764
Epoch 10/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0799 - acc: 0.9752 - val_loss: 0.0791 - val_acc: 0.9761
Epoch 11/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.0744 - acc: 0.9770 - val_loss: 0.1024 - val_acc: 0.9700
Epoch 12/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0711 - acc: 0.9777 - val_loss: 0.0656 - val_acc: 0.9796
Epoch 13/20
60000/60000 [==============================] - 11s 183us/step - loss: 0.0679 - acc: 0.9789 - val_loss: 0.0666 - val_acc: 0.9779
Epoch 14/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.0637 - acc: 0.9803 - val_loss: 0.0702 - val_acc: 0.9794
Epoch 15/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0609 - acc: 0.9811 - val_loss: 0.0679 - val_acc: 0.9784
Epoch 16/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0582 - acc: 0.9820 - val_loss: 0.0622 - val_acc: 0.9798
Epoch 17/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0556 - acc: 0.9827 - val_loss: 0.0903 - val_acc: 0.9711
Epoch 18/20
60000/60000 [==============================] - 11s 179us/step - loss: 0.0535 - acc: 0.9832 - val_loss: 0.0598 - val_acc: 0.9823
Epoch 19/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0520 - acc: 0.9838 - val_loss: 0.0612 - val_acc: 0.9810
Epoch 20/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.0490 - acc: 0.9845 - val_loss: 0.0725 - val_acc: 0.9776
<keras.callbacks.History at 0x7faa39311c50>
~~~

___

## 3. Go Crazy (Batch Normalization, Drop Out, Learning Rate, Batch Size)

~~~python
# Set up data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)                           
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_train = X_train.astype('float32')                                            
X_test = X_test.astype('float32') 
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)                                  
Y_test = np_utils.to_categorical(y_test, 10)

from keras.layers import Activation
model = Sequential()
 
model.add(Convolution2D(12, 3, 3, activation='relu', input_shape=(28,28,1)))    #26
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(20, 3, 3, activation='relu'))                           #24
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(10, 1, 1, activation='relu'))                           #24
model.add(MaxPooling2D(pool_size=(2, 2)))                                       #12

model.add(Convolution2D(16, 3, 3, activation='relu'))                           #10
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(16, 3, 3, activation='relu'))                           #8
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(16, 3, 3, activation='relu'))                           #6
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(16, 3, 3, activation='relu'))                           #4
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 4))


model.add(Flatten())
model.add(Activation('softmax'))


model.summary()
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.003 * 1/(1 + 0.319 * epoch), 10)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.003), metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=1, validation_data=(X_test, Y_test), callbacks=[LearningRateScheduler(scheduler, verbose=1)])
~~~

~~~txt
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_209 (Conv2D)          (None, 26, 26, 12)        120       
_________________________________________________________________
batch_normalization_55 (Batc (None, 26, 26, 12)        48        
_________________________________________________________________
dropout_55 (Dropout)         (None, 26, 26, 12)        0         
_________________________________________________________________
conv2d_210 (Conv2D)          (None, 24, 24, 20)        2180      
_________________________________________________________________
batch_normalization_56 (Batc (None, 24, 24, 20)        80        
_________________________________________________________________
dropout_56 (Dropout)         (None, 24, 24, 20)        0         
_________________________________________________________________
conv2d_211 (Conv2D)          (None, 24, 24, 10)        210       
_________________________________________________________________
max_pooling2d_27 (MaxPooling (None, 12, 12, 10)        0         
_________________________________________________________________
conv2d_212 (Conv2D)          (None, 10, 10, 16)        1456      
_________________________________________________________________
batch_normalization_57 (Batc (None, 10, 10, 16)        64        
_________________________________________________________________
dropout_57 (Dropout)         (None, 10, 10, 16)        0         
_________________________________________________________________
conv2d_213 (Conv2D)          (None, 8, 8, 16)          2320      
_________________________________________________________________
batch_normalization_58 (Batc (None, 8, 8, 16)          64        
_________________________________________________________________
dropout_58 (Dropout)         (None, 8, 8, 16)          0         
_________________________________________________________________
conv2d_214 (Conv2D)          (None, 6, 6, 16)          2320      
_________________________________________________________________
batch_normalization_59 (Batc (None, 6, 6, 16)          64        
_________________________________________________________________
dropout_59 (Dropout)         (None, 6, 6, 16)          0         
_________________________________________________________________
conv2d_215 (Conv2D)          (None, 4, 4, 16)          2320      
_________________________________________________________________
batch_normalization_60 (Batc (None, 4, 4, 16)          64        
_________________________________________________________________
dropout_60 (Dropout)         (None, 4, 4, 16)          0         
_________________________________________________________________
conv2d_216 (Conv2D)          (None, 1, 1, 10)          2570      
_________________________________________________________________
flatten_27 (Flatten)         (None, 10)                0         
_________________________________________________________________
activation_27 (Activation)   (None, 10)                0         
=================================================================
Total params: 13,880
Trainable params: 13,688
Non-trainable params: 192
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 21s 348us/step - loss: 0.2010 - acc: 0.9376 - val_loss: 0.0608 - val_acc: 0.9802
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0618 - acc: 0.9811 - val_loss: 0.0547 - val_acc: 0.9823
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0507 - acc: 0.9843 - val_loss: 0.0382 - val_acc: 0.9887
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 10s 174us/step - loss: 0.0416 - acc: 0.9874 - val_loss: 0.0278 - val_acc: 0.9908
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 10s 174us/step - loss: 0.0377 - acc: 0.9882 - val_loss: 0.0379 - val_acc: 0.9883
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 10s 174us/step - loss: 0.0339 - acc: 0.9890 - val_loss: 0.0249 - val_acc: 0.9931
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0305 - acc: 0.9904 - val_loss: 0.0279 - val_acc: 0.9912
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0288 - acc: 0.9907 - val_loss: 0.0242 - val_acc: 0.9916
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 10s 172us/step - loss: 0.0280 - acc: 0.9911 - val_loss: 0.0206 - val_acc: 0.9936
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0258 - acc: 0.9917 - val_loss: 0.0210 - val_acc: 0.9936
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0247 - acc: 0.9919 - val_loss: 0.0206 - val_acc: 0.9930
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0223 - acc: 0.9927 - val_loss: 0.0211 - val_acc: 0.9942
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0210 - acc: 0.9930 - val_loss: 0.0217 - val_acc: 0.9928
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0208 - acc: 0.9931 - val_loss: 0.0202 - val_acc: 0.9945
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0200 - acc: 0.9937 - val_loss: 0.0199 - val_acc: 0.9941
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 10s 175us/step - loss: 0.0193 - acc: 0.9939 - val_loss: 0.0206 - val_acc: 0.9934
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 10s 172us/step - loss: 0.0182 - acc: 0.9943 - val_loss: 0.0227 - val_acc: 0.9936
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 10s 174us/step - loss: 0.0170 - acc: 0.9945 - val_loss: 0.0223 - val_acc: 0.9928
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0177 - acc: 0.9944 - val_loss: 0.0195 - val_acc: 0.9934
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 10s 172us/step - loss: 0.0175 - acc: 0.9942 - val_loss: 0.0250 - val_acc: 0.9919
<keras.callbacks.History at 0x7faa34bbbf98>
~~~

___

END OF ASSIGNMENT 4 DOCUMENTATION

Below are the architectures used to play around with features and accuracy over Assignment 3 and Assignment 4

---

# Other Architecture and their results

### 1

```python
from keras.layers import Activation
model = Sequential()


model.add(Convolution2D(8, (3, 3), activation='relu', input_shape=(28,28,1)))#26  
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(16, (3, 3), activation='relu'))  #24
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(32, (3, 3), activation='relu'))  #22
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #11

model.add(Convolution2D(16, (3, 3), activation='relu'))  #9
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(32, (3, 3), activation='relu'))  #7
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(20, (3, 3), activation='relu'))  #5
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 5))
model.add(Flatten())
model.add(Activation('softmax'))
```

~~~text
Total params: 20,010
Trainable params: 19,742
Non-trainable params: 268
~~~

~~~txt
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:1: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
  """Entry point for launching an IPython kernel.
Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 25s 414us/step - loss: 0.2640 - acc: 0.9169 - val_loss: 0.0545 - val_acc: 0.9819
Epoch 2/20
60000/60000 [==============================] - 22s 373us/step - loss: 0.0747 - acc: 0.9768 - val_loss: 0.0418 - val_acc: 0.9853
Epoch 3/20
60000/60000 [==============================] - 22s 370us/step - loss: 0.0587 - acc: 0.9817 - val_loss: 0.0356 - val_acc: 0.9889
Epoch 4/20
60000/60000 [==============================] - 22s 372us/step - loss: 0.0481 - acc: 0.9853 - val_loss: 0.0253 - val_acc: 0.9922
Epoch 5/20
60000/60000 [==============================] - 22s 374us/step - loss: 0.0441 - acc: 0.9867 - val_loss: 0.0305 - val_acc: 0.9911
Epoch 6/20
60000/60000 [==============================] - 22s 370us/step - loss: 0.0394 - acc: 0.9875 - val_loss: 0.0280 - val_acc: 0.9916
Epoch 7/20
60000/60000 [==============================] - 22s 369us/step - loss: 0.0374 - acc: 0.9881 - val_loss: 0.0304 - val_acc: 0.9901
Epoch 8/20
60000/60000 [==============================] - 22s 370us/step - loss: 0.0364 - acc: 0.9885 - val_loss: 0.0292 - val_acc: 0.9908
Epoch 9/20
60000/60000 [==============================] - 22s 368us/step - loss: 0.0330 - acc: 0.9892 - val_loss: 0.0241 - val_acc: 0.9922
Epoch 10/20
60000/60000 [==============================] - 22s 371us/step - loss: 0.0298 - acc: 0.9902 - val_loss: 0.0341 - val_acc: 0.9895
Epoch 11/20
60000/60000 [==============================] - 23s 384us/step - loss: 0.0306 - acc: 0.9904 - val_loss: 0.0217 - val_acc: 0.9931
Epoch 12/20
60000/60000 [==============================] - 23s 380us/step - loss: 0.0276 - acc: 0.9912 - val_loss: 0.0239 - val_acc: 0.9915
Epoch 13/20
60000/60000 [==============================] - 23s 380us/step - loss: 0.0271 - acc: 0.9909 - val_loss: 0.0235 - val_acc: 0.9925
Epoch 14/20
60000/60000 [==============================] - 23s 390us/step - loss: 0.0253 - acc: 0.9916 - val_loss: 0.0254 - val_acc: 0.9921
Epoch 15/20
60000/60000 [==============================] - 23s 386us/step - loss: 0.0266 - acc: 0.9915 - val_loss: 0.0209 - val_acc: 0.9941
Epoch 16/20
60000/60000 [==============================] - 23s 384us/step - loss: 0.0236 - acc: 0.9924 - val_loss: 0.0195 - val_acc: 0.9935
Epoch 17/20
60000/60000 [==============================] - 23s 388us/step - loss: 0.0216 - acc: 0.9929 - val_loss: 0.0257 - val_acc: 0.9916
Epoch 18/20
60000/60000 [==============================] - 23s 384us/step - loss: 0.0226 - acc: 0.9925 - val_loss: 0.0227 - val_acc: 0.9924
Epoch 19/20
60000/60000 [==============================] - 23s 379us/step - loss: 0.0220 - acc: 0.9927 - val_loss: 0.0244 - val_acc: 0.9927
Epoch 20/20
60000/60000 [==============================] - 23s 381us/step - loss: 0.0206 - acc: 0.9934 - val_loss: 0.0213 - val_acc: 0.9943
<keras.callbacks.History at 0x7f2557612630>

~~~

---

### 2 

~~~python
from keras.layers import Activation
model = Sequential()


model.add(Convolution2D(10, (3, 3), activation='relu', input_shape=(28,28,1)))#26  
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(16, (3, 3), activation='relu'))  #24
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 1, activation='relu'))

model.add(Convolution2D(20, (3, 3), activation='relu'))  #22
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(MaxPooling2D(pool_size=(2, 2))) #11

model.add(Convolution2D(10, (3, 3), activation='relu'))  #9
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(16, (3, 3), activation='relu'))  #7
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(16, (3, 3), activation='relu'))  #5
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(16, (3, 3), activation='relu'))  #3
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 3))
model.add(Flatten())
model.add(Activation('softmax'))		

~~~

~~~txt
Total params: 12,664
Trainable params: 12,436
Non-trainable params: 228

~~~



Not worth it.

---

### 3

~~~python
from keras.layers import Activation
model = Sequential()


model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(28,28,1)))#26  
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 1, activation='relu'))#22

model.add(Convolution2D(16, (3, 3), activation='relu'))  #24
model.add(BatchNormalization())
model.add(Dropout(0.1))



model.add(Convolution2D(16, (3, 3), activation='relu'))  #22
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(MaxPooling2D(pool_size=(2, 2))) #11


model.add(Convolution2D(16, (3, 3), activation='relu'))  #9
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(16, (3, 3), activation='relu'))  #7
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(10, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(16, (3, 3), activation='relu'))  #5
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 5))
model.add(Flatten())
model.add(Activation('softmax'))	

~~~



~~~
Total params: 15,958
Trainable params: 15,650
Non-trainable params: 308

~~~

~~~txt
Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 26s 438us/step - loss: 0.2816 - acc: 0.9117 - val_loss: 0.0564 - val_acc: 0.9816
Epoch 2/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0755 - acc: 0.9759 - val_loss: 0.0378 - val_acc: 0.9867
Epoch 3/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0588 - acc: 0.9814 - val_loss: 0.0565 - val_acc: 0.9832
Epoch 4/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0498 - acc: 0.9846 - val_loss: 0.0305 - val_acc: 0.9904
Epoch 5/20
60000/60000 [==============================] - 23s 383us/step - loss: 0.0463 - acc: 0.9854 - val_loss: 0.0336 - val_acc: 0.9893
Epoch 6/20
60000/60000 [==============================] - 23s 381us/step - loss: 0.0412 - acc: 0.9872 - val_loss: 0.0250 - val_acc: 0.9926
Epoch 7/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0378 - acc: 0.9882 - val_loss: 0.0239 - val_acc: 0.9927
Epoch 8/20
60000/60000 [==============================] - 23s 381us/step - loss: 0.0363 - acc: 0.9885 - val_loss: 0.0287 - val_acc: 0.9911
Epoch 9/20
60000/60000 [==============================] - 23s 384us/step - loss: 0.0341 - acc: 0.9893 - val_loss: 0.0224 - val_acc: 0.9938
Epoch 10/20
60000/60000 [==============================] - 23s 383us/step - loss: 0.0341 - acc: 0.9894 - val_loss: 0.0265 - val_acc: 0.9916
Epoch 11/20
60000/60000 [==============================] - 23s 381us/step - loss: 0.0315 - acc: 0.9897 - val_loss: 0.0228 - val_acc: 0.9923
Epoch 12/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0292 - acc: 0.9906 - val_loss: 0.0243 - val_acc: 0.9924
Epoch 13/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0283 - acc: 0.9910 - val_loss: 0.0208 - val_acc: 0.9931
Epoch 14/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0274 - acc: 0.9912 - val_loss: 0.0316 - val_acc: 0.9909
Epoch 15/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0270 - acc: 0.9911 - val_loss: 0.0276 - val_acc: 0.9917
Epoch 16/20
60000/60000 [==============================] - 23s 383us/step - loss: 0.0253 - acc: 0.9916 - val_loss: 0.0217 - val_acc: 0.9930
Epoch 17/20
60000/60000 [==============================] - 23s 384us/step - loss: 0.0255 - acc: 0.9915 - val_loss: 0.0223 - val_acc: 0.9929
Epoch 18/20
60000/60000 [==============================] - 23s 384us/step - loss: 0.0236 - acc: 0.9923 - val_loss: 0.0213 - val_acc: 0.9940
Epoch 19/20
60000/60000 [==============================] - 23s 383us/step - loss: 0.0225 - acc: 0.9925 - val_loss: 0.0220 - val_acc: 0.9929
Epoch 20/20
60000/60000 [==============================] - 23s 384us/step - loss: 0.0232 - acc: 0.9926 - val_loss: 0.0221 - val_acc: 0.9939
<keras.callbacks.History at 0x7fce5b5ee6d8>

~~~





---

### 4

~~~txt
# Set up data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)                           
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_train = X_train.astype('float32')                                            
X_test = X_test.astype('float32') 
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)                                  
Y_test = np_utils.to_categorical(y_test, 10)

from keras.layers import Activation
model = Sequential()
 
model.add(Convolution2D(8, (3, 3), activation='relu', input_shape=(28,28,1)))    #26
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(16, (3, 3), activation='relu'))                           #24
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(10, (1, 1), activation='relu'))                           #24

model.add(MaxPooling2D(pool_size=(2, 2)))                                       #12

model.add(Convolution2D(8, (3, 3), activation='relu'))                           #10
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(16, (3, 3), activation='relu'))                           #8
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(16, (3, 3), activation='relu'))                           #6
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(16, (3, 3), activation='relu') )                          #4
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 4))        
model.add(Flatten())
model.add(Activation('softmax'))


model.summary()
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

~~~

> Was able to achieve 99.42 but after overfitting, but result was not overfitted.

___



## Kernel Optimization ( Parameters < 15k )

## Go Crazy 

## Accuracy - Result

