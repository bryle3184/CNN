import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self, batch_size=128, learning_rate=1e-4, num_labels=10):
        super().__init__()
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_labels = num_labels

        self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(num_labels)

    def call(self, images, training=False):
        x = self.conv1(images)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        return self.fc2(x)

    def compute_loss(self, logits, labels):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cost = tf.reduce_mean(cross_entropy)
        return cost

    def accuracy(self, logits, labels):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), dtype=tf.float32))
        return accuracy