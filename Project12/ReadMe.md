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









---

### ResBlk

