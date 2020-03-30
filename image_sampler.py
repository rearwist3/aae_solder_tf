from tensorflow.python.keras.preprocessing.image import Iterator
import os
import numpy as np
from PIL import Image


class ImageSampler:
    def __init__(self, target_size=None, color_mode='rgb', normalize_mode='tanh', is_training=True):
        self.target_size = target_size
        self.color_mode = color_mode
        self.normalize_mode = normalize_mode
        self.is_training = is_training

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None):
        return ArrayIterator(x, y, target_size=self.target_size,
                             color_mode=self.color_mode, batch_size=batch_size,
                             normalize_mode=self.normalize_mode, shuffle=shuffle, seed=seed,
                             is_training=self.is_training)

    def flow_from_directory(self, image_dir, batch_size=32, shuffle=True, seed=None, nb_sample=None):
        image_paths = np.array([path for path in get_image_paths(image_dir)])
        if nb_sample is not None:
            image_paths = image_paths[:nb_sample]

        return DirectoryIterator(paths=image_paths, target_size=self.target_size,
                                 color_mode=self.color_mode, batch_size=batch_size,
                                 normalize_mode=self.normalize_mode, shuffle=shuffle, seed=seed,
                                 is_training=self.is_training)

    def flow_from_path(self, paths, y=None, batch_size=32, shuffle=True, seed=None, nb_sample=None):
        if nb_sample is not None:
            image_paths = paths[:nb_sample]
            labels = y[:nb_sample]
        else:
            image_paths = paths
            labels = y
        raise NotImplementedError


class DirectoryIterator(Iterator):
    def __init__(self, paths, target_size, color_mode, batch_size, normalize_mode, shuffle, seed, is_training):
        self.paths = paths
        self.target_size = target_size
        self.color_mode = color_mode
        self.nb_sample = len(self.paths)
        self.batch_size = batch_size
        self.normalize_mode = normalize_mode
        super().__init__(self.nb_sample, batch_size, shuffle, seed)
        self.current_paths = None
        self.is_trianing = is_training

    def __call__(self, *args, **kwargs):
        if self.is_trianing:
            return self.flow_on_training()
        else:
            return self.flow_on_test()

    def flow_on_training(self):
        with self.lock:
            index_array = next(self.index_generator)
        image_path_batch = self.paths[index_array]
        image_batch = np.array([load_image(path, self.target_size, self.color_mode)
                                for path in image_path_batch])
        self.current_paths = image_path_batch
        return image_batch

    def flow_on_test(self):
        indexes = np.arange(self.nb_sample)
        if self.shuffle:
            print('Now you set is_training = False. \nBut shuffle = True')
            np.random.shuffle(indexes)

        steps = self.nb_sample // self.batch_size
        if self.nb_sample % self.batch_size != 0:
            steps += 1
        for i in range(steps):
            index_array = indexes[i * self.batch_size: (i + 1) * self.batch_size]
            image_path_batch = self.paths[index_array]

            image_batch = np.array([load_image(path, self.target_size, self.color_mode)
                                    for path in image_path_batch])

            self.current_paths = image_path_batch
            yield image_batch

    def data_to_image(self, x):
        if x.shape[-1] == 1:
            x = x.reshape(x.shape[:-1])
        return denormalize(x, self.normalize_mode)


class ArrayIterator(Iterator):
    def __init__(self, x, y, target_size, color_mode, batch_size, normalize_mode, shuffle, seed, is_training):
        self.x = x
        self.y = y
        self.target_size = target_size
        self.color_mode = color_mode
        self.nb_sample = len(self.x)
        self.batch_size = batch_size
        self.normalize_mode = normalize_mode
        super().__init__(self.nb_sample, batch_size, shuffle, seed)
        self.is_training = is_training

        if len(self.x.shape) == 4:
            if x.shape[3] == 1:
                self.x = self.x.reshape(self.x.shape[:3])

    def __call__(self, *args, **kwargs):
        if self.is_training:
            return self.flow_on_training()
        else:
            return self.flow_on_test()

    def flow_on_training(self):
        with self.lock:
            index_array = next(self.index_generator)
        image_batch = np.array([preprocessing(Image.fromarray(x),
                                              color_mode=self.color_mode,
                                              target_size=self.target_size)
                                for x in self.x[index_array]])
        if self.y is not None:
            label_batch = self.y[index_array]
            return image_batch, label_batch
        else:
            return image_batch

    def flow_on_test(self):
        indexes = np.arange(self.nb_sample)
        if self.shuffle:
            print('Now you set is_training = False. \nBut shuffle = True')
            np.random.shuffle(indexes)

        steps = self.nb_sample // self.batch_size
        if self.nb_sample % self.batch_size != 0:
            steps += 1
        for i in range(steps):
            index_array = indexes[i * self.batch_size: (i + 1) * self.batch_size]
            image_batch = np.array([preprocessing(Image.fromarray(x),
                                                  color_mode=self.color_mode,
                                                  target_size=self.target_size)
                                    for x in self.x[index_array]])
            if self.y is not None:
                label_batch = self.y[index_array]
                yield image_batch, label_batch
            else:
                yield image_batch

    def data_to_image(self, x):
        if x.shape[-1] == 1:
            x = x.reshape(x.shape[:-1])
        return denormalize(x, self.normalize_mode)


def preprocessing(x, target_size=None, color_mode='rgb'):
    assert color_mode in ['grayscale', 'gray', 'rgb']
    if color_mode in ['grayscale', 'gray']:
        image = x.convert('L')
    else:
        image = x

    if target_size is not None and target_size != image.size:
        image = image.resize(target_size, Image.BILINEAR)

    image_array = np.asarray(image)

    image_array = normalize(image_array)

    if color_mode in ['grayscale', 'gray']:
        image_array = image_array.reshape(image.size[1], image.size[0], 1)
    return image_array


def load_image(path, target_size=None, color_mode='rgb'):
    image = Image.open(path)
    try:
        return preprocessing(image, target_size, color_mode)
    except:
        print(path)
        exit()


def normalize(x, mode='tanh'):
    if mode == 'tanh':
        return (x.astype('float32') / 255 - 0.5) / 0.5
    elif mode == 'sigmoid':
        return x.astype('float32') / 255
    else:
        raise NotImplementedError


def denormalize(x, mode='tanh'):
    if mode == 'tanh':
        return ((x + 1.) / 2 * 255).astype('uint8')
    elif mode == 'sigmoid':
        return (x * 255).astype('uint8')
    else:
        raise NotImplementedError


def get_image_paths(src_dir):
    def get_all_paths():
        for root, dirs, files in os.walk(src_dir):
            yield root
            for file in files:
                yield os.path.join(root, file)

    def is_image(path):
        if 'png' in path or 'jpg' in path or 'bmp' in path:
            return True
        else:
            return False

    return [path for path in get_all_paths() if is_image(path)]
