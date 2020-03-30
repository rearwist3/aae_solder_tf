import os
import sys
import pandas as pd
sys.path.append(os.getcwd())
from tensorflow.python.keras.preprocessing.image import Iterator
import os
import numpy as np
from PIL import Image
from image_sampler import normalize, denormalize, get_image_paths
from tqdm import tqdm


class ImageSampler:
    def __init__(self, target_size=(128, 128), channel=8, normalize_mode='tanh', is_training=True):
        self.target_size = target_size
        self.channel = channel
        self.normalize_mode = normalize_mode
        self.is_training = is_training

    def flow_from_csv(self, ok_csv_path, ng_csv_path, ok_image_dir, ng_image_dir, batch_size=32, shuffle=True, seed=None, ok_nb_sample=None, ng_nb_sample=None):
        print('LOADING CSV ... ', end='')
        df_ok = pd.read_csv(ok_csv_path)
        if  ok_nb_sample is not None:
            df_ok = df_ok[:ok_nb_sample]
        df_ng = pd.read_csv(ng_csv_path)
        if  ng_nb_sample is not None:
            df_ng = df_ng[:ng_nb_sample]
        print('[COMPLETE]')
        return CSVIterator(df_ok=df_ok, df_ng=df_ng, ok_image_dir=ok_image_dir, ng_image_dir=ng_image_dir, target_size=self.target_size,
                           channel=self.channel, batch_size=batch_size,
                           normalize_mode=self.normalize_mode, shuffle=shuffle, seed=seed,
                           is_training=self.is_training)


class CSVIterator(Iterator):
    def __init__(self, df_ok, df_ng, ok_image_dir, ng_image_dir, target_size, channel, batch_size, normalize_mode, shuffle, seed, is_training):
        self.df_ok = df_ok
        self.df_ng = df_ng
        self.df = pd.concat([self.df_ok, self.df_ng])
        self.ok_image_dir = ok_image_dir
        self.ng_image_dir = ng_image_dir    
        self.target_size = target_size
        self.channel = channel
        self.nb_sample = len(self.df)
        self.normalize_mode = normalize_mode
        super().__init__(self.nb_sample, batch_size, shuffle, seed)
        self.current_path = None
        self.current_df = None
        self.is_training = is_training

    def __call__(self, *args, **kwargs):
        return self.flow_on_training()

    def flow_on_training(self):
        with self.lock:
            index_array = next(self.index_generator)
        df_batch = self.df.iloc[index_array]
        image_path_batch = []
        for i, row in df_batch.iterrows():
            name = '_'.join([row['name'], row['location']]) + '.png'
            if row['label'] == 'inlier':
                image_path_batch.append(os.path.join(self.ok_image_dir, name))
            elif row['label'] == 'outlier':
                image_path_batch.append(os.path.join(self.ng_image_dir, name))
        #print('create image_batch:')
        image_batch = np.array([load_image(path, channel=self.channel)
                                for path in image_path_batch])
        self.current_path = image_path_batch
        return image_batch

    def data_to_image(self, x):
        if x.shape[-1] == 1:
            x = x.reshape(x.shape[:-1])
        return denormalize(x, self.normalize_mode)


def load_image(path, channel=None):
    """
    :param paths: (channel, ) string list
    :param target_size:
    :return:
    """

    image_array = np.asarray(Image.open(path))
    image_array = normalize(image_array)
    split_array = np.array(np.split(image_array, channel, axis=1))
    image_array = split_array.transpose((1, 2, 0))
    return image_array