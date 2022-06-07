# Basic-MNIST image-classification

This repo contains a pytorch MNIST image classification script that uses `@tf.function` decorators and `tf.GradientTape`
for low-level training and evaluation loops. 
The script is a python-ized adaptation of jupyter notebook,
[Writing a training loop from scratch](https://keras.io/guides/writing_a_training_loop_from_scratch)
by fchollet, last modified: 2020/04/15

This repo also includes a python-ized version of jupyter notebook [Install TensorFlow on Apple M1 (M1, Pro, Max) with GPU (Metal)
] (https://sudhanva.me/install-tensorflow-on-apple-m1-pro-max) by Sudhanva Narayana, posted in Dec, 2021.
It is used to verify that Apple's tensorflow-metal plugin is using the 
on-board GPU of the Apple M1 chip.

Quote from ["Getting Started with tensorflow-metal PluggableDevice"](https://developer.apple.com/metal/tensorflow-plugin/):  

Accelerate training of machine learning models with TensorFlow 
right on your Mac. Install TensorFlow and the tensorflow-metal 
PluggableDevice to accelerate training with Metal on Mac GPUs.


