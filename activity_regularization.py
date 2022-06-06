import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Layers & models recursively track any losses created during the forward pass
# by layers that call self.add_loss(value). The resulting list of scalar loss
# values are available via the property model.losses at the end of the forward
# pass. If you want to be using these loss components, you should sum them and
# add them to the main loss in your training step. Consider this layer, that
# creates an activity regularization loss:

class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(1e-2 * tf.reduce_sum(inputs))
        return inputs
