import os
import sys
import pandas as pd
sys.path.append(os.getcwd())
from tensorflow.python.keras.preprocessing.image import Iterator
import os
import numpy as np
from PIL import Image
from image_sampler import normalize, denormalize, get_image_paths
import tqdm


class ImageSampler:
    def __init__(self, target_size=(128, 128), channel=8, normalize_mode='tanh', is_training=True):
        self.target_size = target_size
        self.channel = channel
        self.normalize_mode = normalize_mode
        self.is_training = is_training

    def flow_from_csv(self, csv_path, src_dir, batch_size=32, shuffle=True, seed=None, nb_sample=None):
        print('LOADING CSV ... ', end='')
        df = pd.read_csv(csv_path)
        if nb_sample is not None:
            df = df[:nb_sample]
        print('[COMPLETE]')
        return CSVIterator(df=df, src_dir=src_dir, target_size=self.target_size,
                           channel=self.channel, batch_size=batch_size,
                           normalize_mode=self.normalize_mode, shuffle=shuffle, seed=seed,
                           is_training=self.is_training)


class CSVIterator(Iterator):
    def __init__(self, df, src_dir, target_size, channel, batch_size, normalize_mode, shuffle, seed, is_training):
        self.df = df
        self.src_dir = src_dir
        self.target_size = target_size
        self.channel = channel
        self.nb_sample = len(self.df)
        self.normalize_mode = normalize_mode
        super().__init__(self.nb_sample, batch_size, shuffle, seed)
        self.current_path = None
        self.current_df = None
        self.is_training = is_training

    def __call__(self, *args, **kwargs):
            return self.flow_on_test()

    def flow_on_test(self):
        indexes = list(np.arange(self.nb_sample))
        if self.shuffle:
            print('Now you set is_training = False. \nBut shuffle = True')
            np.random.shuffle(indexes)

        steps = self.nb_sample // self.batch_size
        if self.nb_sample % self.batch_size != 0:
            steps += 1

        for step in range(steps):
            df_batch = self.df.iloc[step*self.batch_size: (step+1)*self.batch_size]
            image_path_batch = []

            for i, row in df_batch.iterrows():
                name = '_'.join([row['name'], row['location']]) + '.png'
                image_path_batch.append(os.path.join(self.src_dir, name))
            image_batch = np.array([load_image(path, channel=self.channel)
                                    for path in image_path_batch])
            self.current_df = df_batch
            yield image_batch

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


def preprocessing(x, target_size=None):
    image = x.convert('L')

    if target_size is not None and target_size != image.size:
        image = image.resize(target_size, Image.BILINEAR)

    image_array = np.asarray(image)
    image_array = normalize(image_array)

    return image_array