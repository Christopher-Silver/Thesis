# -*- coding: utf-8 -*-
"""Cache - TF-66 Data.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WpOQVxWp36I0AEOeHyiaKUP1Kn72MZRd

# Imports
"""

#!pip install openpyxl

import random
import numpy as np
import pandas as pd
import os
import re
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, LeakyReLU, Input, Layer
from tensorflow.keras.metrics import AUC, Precision, Recall, F1Score
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)

batch_size = 16 # change this
target_size = (256,256)
#target_size = (520,364)
numFrames = 10 #change this

"""# Initialization"""

# Load the spreadsheet
#spreadsheet_path = '/content/drive/MyDrive/Final Dataset.xlsx'
spreadsheet_path = '/home/crsilver/scratch/TF66/Final Dataset.xlsx'
spreadsheet = pd.read_excel(spreadsheet_path)


# Now, create a dictionary that maps video names to their respective framesBeforeFall and framesAfterFall
video_info = {}
for i, row in spreadsheet.iterrows():
    video_name = row['Recording Name']
    video_info[video_name] = {
        'framesBeforeFall': row['framesBeforeFall'],
        'framesAfterFall': row['framesAfterFall'],
        'firstFallFrameOfVideo': row['First Fall Frame of Video']
    }

#train_cache_file = '/content/drive/MyDrive/colab_full_train_cache.npz'
#val_cache_file = '/content/drive/MyDrive/colab_full_val_cache.npz'
train_cache_file = '/home/crsilver/scratch/train_cache.npz'
val_cache_file = '/home/crsilver/scratch/val_cache.npz'
def load_cached_images(cache_file):
    return np.load(cache_file, allow_pickle=True)

# Load the cached images
train_cache = load_cached_images(train_cache_file)
val_cache = load_cached_images(val_cache_file)

# Create filepaths list from the cached data
train_filepaths = list(train_cache.keys())
val_filepaths = list(val_cache.keys())


def sort_frames_numerically(filenames):
    # This function extracts the number from the filename and sorts by the number
    def extract_number(f):
        s = re.findall("\d+", f)
        return (int(s[0]) if s else -1, f)
    return sorted(filenames, key=extract_number)
    
    
    
# Predefined arrays
all_data_array = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66']
eight_feet_array = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66']
nine_feet_array = ['15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37']
ten_feet_array = ['38','39','40','41','42','43','44']
senior_array= ['13','38','43','44','53','54','65']
hospital_array = ['45','46','50','51','52','53','54','61','62']
exposed_arms_array = ['02','03','06','08','20','23','25','28','31','32','34','35','38','39','40','43','47','48','51','53','55','56','57','58','62','63','64','65']
covered_arms_array = ['01','04','05','07','09','10','11','12','13','14','15','16','17','18','19','21','22','24','26','27','29','30','33','36','37','42','44','52']
inconsistent_arms_array = ['41','45','46','49','50','54','59','60','61','66']

# Toggles
all_data = True
eight_feet = False
nine_feet = False
ten_feet = False
senior = False
hospital = False
exposed_arms = False
covered_arms = False
inconsistent_arms = False
 


class CachedNumpyDataGenerator:
    def __init__(self, cache, filepaths, batch_size, num_frames=10, target_size=(256, 256)):
        self.cache = cache
        self.filepaths = filepaths
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.target_size = target_size

        self.active_values = []
        if all_data:
            self.active_values.extend(all_data_array)
        if eight_feet:
            self.active_values.extend(eight_feet_array)
        if nine_feet:
            self.active_values.extend(nine_feet_array)
        if ten_feet:
            self.active_values.extend(ten_feet_array)
        if senior:
            self.active_values.extend(senior_array)
        if hospital:
            self.active_values.extend(hospital_array)
        if exposed_arms:
            self.active_values.extend(exposed_arms_array)
        if covered_arms:
            self.active_values.extend(covered_arms_array)
        if inconsistent_arms:
            self.active_values.extend(inconsistent_arms_array)

    def __iter__(self):
        return self

    def __next__(self):
        X_batch = []
        y_batch = []
        paths_batch = []

        while len(X_batch) < self.batch_size:
            random_file_path = random.choice(self.filepaths)
            path_parts = random_file_path.split(os.sep)
            category_folder = path_parts[-3]
            video_name = path_parts[-2]
            #print("Category folder is " + category_folder)
            #print("Video name is " + video_name)

            # Extract the first two characters of the video name
            video_name_key = video_name.replace('.avi', '')
            first_two_chars = video_name_key[:2]

            # Check if the first two characters match any value in the active_values list
            #if first_two_chars not in self.active_values:
                #print("first_two_chars NOT IN active values. first_two_chars is " + str(first_two_chars))
                #continue
            #else:
                #print("first_two_chars IN active values. first_two_chars is " + str(first_two_chars))

            if video_name_key not in video_info:
                continue

            framesBeforeFall = video_info[video_name_key]['framesBeforeFall']
            framesAfterFall = video_info[video_name_key]['framesAfterFall']
            firstFallFrameOfVideo = video_info[video_name_key]['firstFallFrameOfVideo']

            frames_dir = os.path.join(os.path.dirname(random_file_path), '')
            if not os.path.isdir(frames_dir):
                continue

            frames = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
            frames = sort_frames_numerically(frames)

            totalFrames = len(frames)

            if '-NonFall-' in video_name:
                start_min = 0
                start_max = max(0, totalFrames - self.num_frames)
            else:
                if self.num_frames > framesAfterFall:
                    raise ValueError("numFrames must be less than framesAfterFall")

                start_min = max(0, firstFallFrameOfVideo - self.num_frames)
                start_max = min(firstFallFrameOfVideo - (self.num_frames // 2), totalFrames - self.num_frames)
                #print("firstFallFrameOfVideo is " + firstFallFrameOfVideo)
                #print("self.num_Frames " + self.num_frames)
                #print("start_min is " + start_min)
                #print("start_max is " + start_max)
                #print ("totalFrames is " + totalFrames)
                #print("random_file_path is " + random_file_path)
                #print("video_name_key is " + video_name_key)
                #print("first_two_chars is " + first_two_chars)
                #print("frames is " + frames)
                if start_min > start_max:
                    raise ValueError("Invalid constraints: No valid starting frame can be found.")

            start_frame = random.randint(start_min, start_max)
            frames_array = [self.cache[os.path.join(frames_dir, frames[j])] for j in range(start_frame, start_frame + self.num_frames)]

            X_batch.append(np.array(frames_array))
            paths_batch.append([os.path.join(frames_dir, frames[j]) for j in range(start_frame, start_frame + self.num_frames)])

            if '-NonFall-' in video_name:
                y_batch.append(0)
            else:
                y_batch.append(1)

        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch).astype('float32')
        return X_batch, y_batch#, paths_batch


train_generator = CachedNumpyDataGenerator(train_cache, train_filepaths, batch_size=batch_size)
validation_generator = CachedNumpyDataGenerator(val_cache, val_filepaths, batch_size=batch_size)


import tensorflow as tf
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, numFrames, 256, 256, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        #tf.TensorSpec(shape=(None, numFrames), dtype=tf.string)
    )
).prefetch(tf.data.experimental.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, numFrames, 256, 256, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        #tf.TensorSpec(shape=(None, numFrames), dtype=tf.string)
    )
).prefetch(tf.data.experimental.AUTOTUNE)



from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K

class Sensitivity(Metric):
    def __init__(self, name='sensitivity', **kwargs):
        super(Sensitivity, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.possible_positives = self.add_weight(name='pp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.reshape(y_pred, [-1])
        y_pred = K.round(y_pred)

        # Calculate true positives
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        # Calculate possible positives
        pp = K.sum(K.round(K.clip(y_true, 0, 1)))

        # Print intermediate values for debugging
        # tf.print("Batch TP:", tp)
        # tf.print("Batch Possible Positives:", pp)
        # tf.print("Accumulated TP before update:", self.true_positives)
        # tf.print("Accumulated Possible Positives before update:", self.possible_positives)

        # Update accumulated values
        self.true_positives.assign_add(tp)
        self.possible_positives.assign_add(pp)

        # Print accumulated values after update
        # tf.print("Accumulated TP after update:", self.true_positives)
        # tf.print("Accumulated Possible Positives after update:", self.possible_positives)

    def result(self):
        sensitivity_value = self.true_positives / (self.possible_positives + K.epsilon())
         # Print the final values for debugging
        # tf.print("Final TP (across all batches):", self.true_positives)
        # tf.print("Final Possible Positives (across all batches):", self.possible_positives)
        # tf.print("Final Sensitivity Value:", sensitivity_value)
        return sensitivity_value

    def reset_state(self):
        self.true_positives.assign(0)
        self.possible_positives.assign(0)

import tensorflow as tf
from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K

class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.reshape(y_pred, [-1])
        y_pred = K.round(y_pred)

        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
        fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

class MCC(Metric):
    def __init__(self, name='mcc', **kwargs):
        super(MCC, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.reshape(y_pred, [-1])
        y_pred = K.round(K.clip(y_pred, 0, 1))

        tp = K.sum(y_true * y_pred)
        tn = K.sum((1 - y_true) * (1 - y_pred))
        fp = K.sum((1 - y_true) * y_pred)
        fn = K.sum(y_true * (1 - y_pred))

        self.true_positives.assign_add(tp)
        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        numerator = (self.true_positives * self.true_negatives) - (self.false_positives * self.false_negatives)
        denominator = K.sqrt(
            (self.true_positives + self.false_positives) *
            (self.true_positives + self.false_negatives) *
            (self.true_negatives + self.false_positives) *
            (self.true_negatives + self.false_negatives)
        )
        return numerator / (denominator + K.epsilon())

    def reset_state(self):
        self.true_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

class Specificity(Metric):
    def __init__(self, name='specificity', **kwargs):
        super(Specificity, self).__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.reshape(y_pred, [-1])
        y_pred = K.round(K.clip(y_pred, 0, 1))

        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))

        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)

    def result(self):
        return self.true_negatives / (self.true_negatives + self.false_positives + K.epsilon())

    def reset_state(self):
        self.true_negatives.assign(0)
        self.false_positives.assign(0)



from tensorflow.keras.callbacks import EarlyStopping







class SpatialAttention(Layer):
    def __init__(self, activation='sigmoid'):
        super(SpatialAttention, self).__init__()
        self.activation = activation

    def call(self, inputs):
        # Reduce across temporal and channel dimensions
        # inputs: [batch_size, num_frames, height, width, channels]
        spatial_features = tf.reduce_mean(inputs, axis=[1, 4])  # Shape: [batch_size, height, width]

        # Apply the chosen activation function to generate attention weights
        if self.activation == 'sigmoid':
            attention_weights = tf.nn.sigmoid(spatial_features)  # Shape: [batch_size, height, width]
        elif self.activation == 'softmax':
            attention_weights = tf.nn.softmax(spatial_features, axis=-1)  # Shape: [batch_size, height, width]
        else:
            raise ValueError("Activation must be 'sigmoid' or 'softmax'")

        # Expand dimensions to match the input shape for broadcasting
        attention_weights = tf.expand_dims(tf.expand_dims(attention_weights, axis=1), axis=-1)  # Shape: [batch_size, 1, height, width, 1]

        # Apply attention weights to the input
        weighted_inputs = inputs * attention_weights  # Broadcasting happens here

        return weighted_inputs



# Define Temporal Attention Layer
class TemporalAttention(Layer):
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def call(self, inputs):
        # Inputs shape: (batch, num_frames, height, width, channels)
        # Apply temporal mean across spatial dimensions
        temporal_features = tf.reduce_mean(inputs, axis=[2, 3])  # Shape: (batch, num_frames, channels)
        attention_weights = tf.nn.softmax(temporal_features, axis=1)  # Shape: (batch, num_frames, channels)
        # Expand dimensions to match inputs
        attention_weights = tf.expand_dims(tf.expand_dims(attention_weights, axis=2), axis=3)  # Shape: (batch, num_frames, 1, 1, channels)
        # Apply attention
        weighted_inputs = inputs * attention_weights
        return weighted_inputs

    def compute_output_shape(self, input_shape):
        # Output shape is the same as input shape
        return input_shape

# Feature-Based Attention (Squeeze-and-Excitation style)
class FeatureBasedAttention(Layer):
    def __init__(self, reduction_ratio=16):
        super(FeatureBasedAttention, self).__init__()
        self.reduction_ratio = reduction_ratio
        # Define Dense layers during initialization
        self.reduce_dense = Dense(
            units=None,  # Set dynamically based on input channels
            activation='relu',
            use_bias=True
        )
        self.restore_dense = Dense(
            units=None,  # Set dynamically based on input channels
            activation='sigmoid',
            use_bias=True
        )

    def build(self, input_shape):
        # Infer channels dynamically and configure Dense layers
        channels = input_shape[-1]
        self.reduce_dense.units = channels // self.reduction_ratio
        self.restore_dense.units = channels

    def call(self, inputs):
        # Squeeze: Compute global channel descriptors
        channel_weights = tf.reduce_mean(inputs, axis=[1, 2, 3])  # Shape: (batch, channels)

        # Reduce dimensionality
        reduced = self.reduce_dense(channel_weights)

        # Restore dimensionality
        restored = self.restore_dense(reduced)

        # Apply attention
        attention_weights = tf.expand_dims(tf.expand_dims(tf.expand_dims(restored, axis=1), axis=1), axis=1)
        return inputs * attention_weights

# Self-Attention
class SelfAttention(Layer):
    def __init__(self, num_heads=4, key_dim=64):
        super(SelfAttention, self).__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def build(self, input_shape):
        # Define any additional layers or variables here if needed
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        # Get the input shape dynamically
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        num_frames = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        channels = input_shape[4]

        # Reshape input for multi-head attention (combine spatial and temporal dimensions)
        reshaped_inputs = tf.reshape(inputs, (batch_size, num_frames * height * width, channels))
        attended = self.attention(reshaped_inputs, reshaped_inputs)
        
        # Reshape back to original 5D shape
        output = tf.reshape(attended, (batch_size, num_frames, height, width, channels))
        return output




















batch_size = 16 # change this
target_size = (256,256)
#target_size = (520,364)
numFrames = 10 #change this

# Model definition
#model = Sequential()
#model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(numFrames, target_size[0], target_size[1], 1)))
#model.add(LeakyReLU(alpha=0.1))
#model.add(MaxPooling3D(pool_size=(1, 2, 2)))
#model.add(Dropout(0.25))
#model.add(Conv3D(64, kernel_size=(3, 3, 3)))
#model.add(LeakyReLU(alpha=0.1))
#model.add(MaxPooling3D(pool_size=(1, 2, 2)))
#model.add(Dropout(0.25))
#model.add(Conv3D(128, kernel_size=(3, 3, 3)))
#model.add(LeakyReLU(alpha=0.1))
#model.add(MaxPooling3D(pool_size=(1, 2, 2)))
#model.add(Dropout(0.25))
#conv_output_shape = model.layers[-1].output_shape
#model.add(Flatten(input_shape=conv_output_shape[1:]))  # Start flattening from the second element (exclude batch size)
#model.add(Flatten())
#flattened_size = model.layers[-1].output_shape[1]
#model.add(Dense(flattened_size, activation='relu'))  # Adjust the number of neurons in the Dense layer
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1, activation='sigmoid'))
from tensorflow.keras.layers import Reshape

from tensorflow.keras.layers import (
    Input, Conv3D, LeakyReLU, MaxPooling3D, Dropout, Reshape, Flatten, Dense, Concatenate, ConvLSTM2D
)
from tensorflow.keras.models import Model

# Model definition
inputs = Input(shape=(numFrames, target_size[0], target_size[1], 1))

# Add first conv block
x = Conv3D(32, kernel_size=(3, 3, 3), padding="same")(inputs)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling3D(pool_size=(1, 2, 2))(x)
x = Dropout(0.25)(x)

# Apply Spatial Attention
#x = SpatialAttention(activation='sigmoid')(x)

# Add second conv block
x = Conv3D(64, kernel_size=(3, 3, 3), padding="same")(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling3D(pool_size=(1, 2, 2))(x)
x = Dropout(0.25)(x)

# Apply Temporal Attention
#x = TemporalAttention()(x)

# Add third conv block
x = Conv3D(128, kernel_size=(3, 3, 3), padding="same")(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling3D(pool_size=(1, 2, 2))(x)
x = Dropout(0.25)(x)

# Reshape for Bi-ConvLSTM
x = Reshape((numFrames, target_size[0] // 8, target_size[1] // 8, 128))(x)

# Forward ConvLSTM
forward_lstm = ConvLSTM2D(64, kernel_size=(3, 3), padding="same", return_sequences=True)(x)

# Backward ConvLSTM
backward_lstm = ConvLSTM2D(64, kernel_size=(3, 3), padding="same", return_sequences=True, go_backwards=True)(x)

# Concatenate forward and backward outputs
x = Concatenate()([forward_lstm, backward_lstm])

# Reshape for Attention
x = Reshape((-1, x.shape[-1]))(x)  # Flatten temporal dimension

# Add Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(1,),
                                 initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)

# Add Attention
x = Attention()(x)

# Fully connected layers
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layer
outputs = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs, outputs)
model.summary()





learning_rate = 0.0001  # Example learning rate
optimizer = Adam(learning_rate=learning_rate)


model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=[
                  'accuracy',
                  AUC(name='auc'),
                  AUC(name='auc_pr', curve='PR'),
                  Recall(name='recall'),
                  Precision(name='precision'),
                  Sensitivity(),
                  Specificity(),
                  F1Score(),
                  MCC()])



initial_steps_per_epoch = max(1, len(train_filepaths) // (batch_size * 50))  # 10% of the dataset
initial_validation_steps = max(1, len(val_filepaths) // (batch_size * 50))  # 10% of the validation dataset

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=10,          # Number of epochs to wait for improvement
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
    mode='min'           # Mode can be 'min', 'max', or 'auto'
)
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint(f'Bi-ConvLSTMAttentionOriginal.keras', save_best_only=True, monitor='val_loss', mode='min')
# Train the model with EarlyStopping callback
model.fit(train_dataset,
          epochs=100,
          steps_per_epoch=initial_steps_per_epoch,
          validation_data=validation_dataset,
          validation_steps=initial_validation_steps,
          verbose=2,
          callbacks=[early_stopping, checkpoint])  # Add the early stopping callback


#Evaluate the model
#train_loss, train_accuracy, train_auc = model.evaluate(train_gen, steps=10)
#print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Training AUC: {train_auc:.4f}")
val_metrics = model.evaluate(validation_dataset, steps=(len(val_filepaths) + batch_size - 1) // batch_size, verbose=2)


val_loss = val_metrics[0]
val_accuracy = val_metrics[1]
val_auc = val_metrics[2]
val_auc_pr = val_metrics[3]
val_recall = val_metrics[4]
val_precision = val_metrics[5]
val_sensitivity = val_metrics[6]
val_specificity = val_metrics[7]
val_f1_score = val_metrics[8]
val_mcc = val_metrics[9]

print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation AUC: {val_auc:.4f}, "
      f"Validation AUC PR: {val_auc_pr:.4f}, Validation Recall: {val_recall:.4f}, Validation Precision: {val_precision:.4f}, "
      f"Validation Sensitivity: {val_sensitivity:.4f}, Validation Specificity: {val_specificity:.4f}, "
      f"Validation F1 Score: {val_f1_score:.4f}, Validation MCC: {val_mcc:.4f}")



import time
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Function to calculate GFLOPS
def calculate_flops(model):
    # Convert Keras model to a concrete function
    concrete_func = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete_func.get_concrete_function(tf.TensorSpec([1, numFrames, target_size[0], target_size[1], 1], tf.float32))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    frozen_func.graph.as_graph_def()

    # Calculate FLOPS
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    
    # Use the profiler to measure FLOPs
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph,
        run_meta=run_meta,
        options=opts
    )
    
    return flops.total_float_ops if flops else 0

# Calculate GFLOPS
total_flops = calculate_flops(model)
gflops = total_flops / 1e9
print(f"GFLOPS: {gflops:.2f}")

# Measure inference time
dummy_input = tf.random.normal([1, numFrames, target_size[0], target_size[1], 1])  # Single sample input
start_time = time.time()
model.predict(dummy_input)
end_time = time.time()

inference_time_per_sample = (end_time - start_time) * 1000  # Convert to milliseconds
print(f"Inference Time per Sample: {inference_time_per_sample:.2f} ms")
