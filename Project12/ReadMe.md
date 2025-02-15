# Session 12

~~~python
tf.enable_eager_execution()
~~~

---

### init_pytorch

Initialization of kernel values 

~~~python
def init_pytorch(shape, dtype=tf.float32, partition_info=None):
  fan = np.prod(shape[:-1])
  bound = 1 / math.sqrt(fan)
  return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)

~~~

Output -  When shape is  [3,3].

~~~txt
tf.Tensor(
[[-0.54611534  0.23626256 -0.48132807]
 [ 0.28005707 -0.14628437 -0.38467693]
 [-0.572527    0.3208421  -0.31219405]], shape=(3, 3), dtype=float32)
~~~

A random 3x3 Tensor is generated. 

---

### ConvBN

~~~python
class ConvBN(tf.keras.Model):
  def __init__(self, c_out):
    super().__init__()
    self.conv = tf.keras.layers.Conv2D(filters=c_out, kernel_size=3, padding="SAME", kernel_initializer=init_pytorch, use_bias=False)
    self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
    self.drop = tf.keras.layers.Dropout(0.05)

    
  def call(self, inputs):
    return tf.nn.relu(self.bn(self.drop(self.conv(inputs))))
~~~



ConvBN class contains an `__init__` function which initializes 

- Conv2D 
  - Number  of filters =  `c_out`.
  - Filter/Kernel size of 3
  - `padding = "SAME"`, input and output image stay the same
  - kernel values initialized using `init_pytorch`
- Batch Normalization with values `momentum=0.9, epsilon=1e-5`
- Drop Out   = 0.05 

These layers are initialized and used in call function transforming to a ConvBN block

 ConvBN class contain a **call** function 

`tf.nn.relu(self.bn(self.drop(self.conv(inputs))))` 

	1. Takes in Input  performs convolution followed by 
 	2. 5% pixel Dropout 
 	3. Applying batch normalization.
 	4.  Activation Relu

> A small Block with 2DConv, BN, DO and Activation



---

### ResBlk

~~~python
class ResBlk(tf.keras.Model):
  def __init__(self, c_out, pool, res = False):
    super().__init__()
    self.conv_bn = ConvBN(c_out)
    self.pool = pool
    self.res = res
    if self.res:
      self.res1 = ConvBN(c_out)
      self.res2 = ConvBN(c_out)

  def call(self, inputs):
    h = self.pool(self.conv_bn(inputs))
    if self.res:
      h = h + self.res2(self.res1(h))
    return h
~~~

If `res` argument is set to False its same as ConvBN with MAX pooling layer of size 2 added 

If `res` argument is set to True. 2 more Conv BN blocks are added after Max pooling.

A residual block. Yellow block is added after max pooling if `res` is set to TRUE.

![r](r.png)



---

### DavidNet

~~~python
class DavidNet(tf.keras.Model):
  def __init__(self, c=64, weight=0.125):
    super().__init__()
    pool = tf.keras.layers.MaxPooling2D()
    self.init_conv_bn = ConvBN(c)
    self.blk1 = ResBlk(c*2, pool, res = True)
    self.blk2 = ResBlk(c*4, pool)
    self.blk3 = ResBlk(c*8, pool, res = True)
    self.pool = tf.keras.layers.GlobalMaxPool2D()
    self.linear = tf.keras.layers.Dense(10, kernel_initializer=init_pytorch, use_bias=False)
    self.weight = weight

  def call(self, x, y):
    h = self.pool(self.blk3(self.blk2(self.blk1(self.init_conv_bn(x)))))
    h = self.linear(h) * self.weight # Reasons Unknown
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h, labels=y)
    loss = tf.reduce_sum(ce)
    correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(h, axis = 1), y), tf.float32))
    return loss, correct
~~~



Finally, let’s build DavidNet itself, which contains

1. A Conv-BN-Relu block
2.  Three Residual blocks, two with residual components and one without.
3. A global max pool,
4. A fully connected layer that outputs logits.
5. A mysterious “Multiply by 0.125” operation.
6.  It outputs two values: cross-entropy loss and accuracy, in terms of the number of correct predictions in a batch. This sounds like a lot of things, but it’s actually not particularly hard to implement in Eager Keras:

---

### Data Filter

~~~python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
len_train, len_test = len(x_train), len(x_test)
y_train = y_train.astype('int64').reshape(len_train)
y_test = y_test.astype('int64').reshape(len_test)

train_mean = np.mean(x_train, axis=(0,1,2))
train_std = np.std(x_train, axis=(0,1,2))

normalize = lambda x: ((x - train_mean) / train_std).astype('float32') # todo: check here
pad4 = lambda x: np.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)], mode='reflect')

x_train = normalize(pad4(x_train))
x_test = normalize(x_test)
~~~

- Data Set Splitting

- Convert to type 64 bit int  

- Calculate mean and std for Manual Image Normalization and padding. 



---

### Pre Model

~~~python
model = DavidNet()
batches_per_epoch = len_train//BATCH_SIZE + 1

lr_schedule = lambda t: np.interp([t], [0, (EPOCHS+1)//5, EPOCHS], [0, LEARNING_RATE, 0])[0]
global_step = tf.train.get_or_create_global_step()
lr_func = lambda: lr_schedule(global_step/batches_per_epoch)/BATCH_SIZE
opt = tf.train.MomentumOptimizer(lr_func, momentum=MOMENTUM, use_nesterov=True)
data_aug = lambda x, y: (tf.image.random_flip_left_right(tf.random_crop(x, [32, 32, 3])), y)

import matplotlib.pyplot as plt
plt.plot([0, (EPOCHS+1)//5, EPOCHS], [0, LEARNING_RATE, 0], 'o')
~~~

- Learning Rate Scheduler   function, creating a cyclic scheduler as shown below.

![cyclic](cyclic.png)

- Momentum Scheduler (using TF's Nesterov Momentum)
- Data Aug - Mirror Images created horizontal , adding to dataset as vertical flit will create useless data.

---

### Model Run

~~~python
t = time.time()
test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

for epoch in range(EPOCHS):
  train_loss = test_loss = train_acc = test_acc = 0.0
  train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(data_aug).shuffle(len_train).batch(BATCH_SIZE).prefetch(1)

  tf.keras.backend.set_learning_phase(1)
  for (x, y) in tqdm(train_set):
    with tf.GradientTape() as tape:
      loss, correct = model(x, y)

    var = model.trainable_variables
    grads = tape.gradient(loss, var)
    for g, v in zip(grads, var):
      g += v * WEIGHT_DECAY * BATCH_SIZE
    opt.apply_gradients(zip(grads, var), global_step=global_step)

    train_loss += loss.numpy()
    train_acc += correct.numpy()

  tf.keras.backend.set_learning_phase(0)
  for (x, y) in test_set:
    loss, correct = model(x, y)
    test_loss += loss.numpy()
    test_acc += correct.numpy()
    
  print('epoch:', epoch+1, 'lr:', lr_schedule(epoch+1), 'train loss:', train_loss / len_train, 'train acc:', train_acc / len_train, 'val loss:', test_loss / len_test, 'val acc:', test_acc / len_test, 'time:', time.time() - t)
~~~

- Record Time
- Slice test set (BatchSize)
- map(dataaug) applied to training set.

This loop is the most important loop 

~~~python
for (x, y) in tqdm(train_set):
    with tf.GradientTape() as tape:
      loss, correct = model(x, y)

    var = model.trainable_variables
    grads = tape.gradient(loss, var)
    for g, v in zip(grads, var):
      g += v * WEIGHT_DECAY * BATCH_SIZE
    opt.apply_gradients(zip(grads, var), global_step=global_step)

    train_loss += loss.numpy()
    train_acc += correct.numpy()
~~~

It takes in training set and uses gradient Tape to keep track to gradient computation and train  gradient wrt to `loss` as done here `grads = tape.gradient(loss, var)`. 

Gradient are then update and changes are made and the model is run for one batch `opt.apply_gradients(zip(grads, var), global_step=global_step)`