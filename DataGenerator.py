import random
import numpy as np
import pandas as pd
import os
import re
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.metrics import AUC, Precision, Recall, F1Score
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)

batch_size = 16 # change this
target_size = (256,256)

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
eight_feet_array = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66']
nine_feet_array = ['15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37']
ten_feet_array = ['38','39','40','41','42','43','44']
senior_array= ['13','38','43','44','53','54','65']
hospital_array = ['45','46','50','51','52','53','54','61','62']
exposed_arms_array = ['02','03','06','08','20','23','25','28','31','32','34','35','38','39','40','43','47','48','51','53','55','56','57','58','62','63','64','65']
covered_arms_array = ['01','04','05','07','09','10','11','12','13','14','15','16','17','18','19','21','22','24','26','27','29','30','33','36','37','42','44','52']
inconsistent_arms_array = ['41','45','46','49','50','54','59','60','61','66']

# Toggles
eight_feet = False
nine_feet = False
ten_feet = False
senior = True
hospital = False
exposed_arms = False
covered_arms = False
inconsistent_arms = False
 


class DataGenerator:
    def __init__(self, cache, filepaths, batch_size, num_frames=10, target_size=(256, 256)):
        self.cache = cache
        self.filepaths = filepaths
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.target_size = target_size

        self.active_values = []
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
            if first_two_chars not in self.active_values:
                print("first_two_chars NOT IN active values. first_two_chars is " + str(first_two_chars))
                continue
            else:
                print("first_two_chars IN active values. first_two_chars is " + str(first_two_chars))

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





train_generator = DataGenerator(train_cache, train_filepaths, batch_size=batch_size)
validation_generator = DataGenerator(val_cache, val_filepaths, batch_size=batch_size)



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

