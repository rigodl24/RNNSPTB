#Import Packages
import tensorflow as tf

#Confirm TS is using GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))