# Assignment 5

1. Change the code 8 or your own 4th Code from Assignment 4 to include:
   - image normalization
   - L2 regularization
   - ReLU after BN
2. Run your new code for 40 epochs and save the model with highest validation accuracy
3. Find out 25 misclassified images from the validation dataset and create an image gallery
4. Submit



---



## Image Normalization 

~~~python
# Image Normalization 

datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen.fit(X_train)
iterator = datagen.flow(X_train, y_train, batch_size=len(X_train), shuffle=False)
X_train, y_train = iterator.next()
~~~

~~~python
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
~~~

> ImageDataGenerator is all that is needed to do image normalization 

Which is then applied over the batches and new augmented data set is generated

~~~python
iterator = datagen.flow(X_train, y_train, batch_size=len(X_train), shuffle=False)
X_train, y_train = iterator.next()
~~~

---

## L2 Regularization 

 L2 work for well for very large number of kernels but we don't have large number of kernel here, might backfire.

~~~Python

# Define custom loss
def custom_loss(actual,predicted):
    sqr_w = 0
    lamda = 0.01 #1e-4

    for layer in model.layers:
       print(layer.get_weights())
       sqr_w = sqr_w + np.sum(np.sum(np.sum(np.square(layer.get_weights()))))

    l2_regularization = (lamda*sqr_w)/2*(bs)
    loss = keras.losses.categorical_crossentropy(actual,predicted) + l2_regularization

    # Return a function
    return loss
~~~

```python
# Pass to metric system in model.compile
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.003 * 1/(1 + 0.319 * epoch), 10)
bs =128
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.003), metrics=['accuracy',custom_loss])
```

---

## BN before Relu

```python
model.add(BatchNormalization())#BN before RELU

model.add(Convolution2D(20, (3, 3), activation='relu'))                         #24
model.add(Dropout(0.1))
```

---

Ran for 40 Epochs

```
model.fit(X_train, Y_train, batch_size=bs, epochs=40, verbose=1, validation_data=(X_test, Y_test), callbacks=[LearningRateScheduler(scheduler, verbose=1)])
```

Highest Validation Accuracy saved

```txt
Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 6s 93us/step - loss: 0.0395 - acc: 0.9872 - custom_loss: 156.7916 - val_loss: 2.2461 - val_acc: 0.8472 - val_custom_loss: 158.9982
```

There was a significant drop as compared to not using tweaks done in this Network.

> With so many Regularization added, we see an increase in overfitting.

~~~
Epoch 00040: LearningRateScheduler setting learning rate to 0.0002231977.
60000/60000 [==============================] - 6s 92us/step - loss: 0.0119 - acc: 0.9962 - custom_loss: 156.7640 - val_loss: 4.1784 - val_acc: 0.7254 - val_custom_loss: 160.9306
~~~

