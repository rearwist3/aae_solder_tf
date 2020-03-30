import os
import sys
import argparse
import pickle
import gzip
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('agg')
sys.path.append(os.getcwd())
import seaborn as sns
import matplotlib.pyplot as plt
from aae import AdversarialAutoEncoder as AAE
from solder.custom_image_sampler_generate import ImageSampler
from solder.autoencoder import AutoEncoder
from solder.discriminator import Discriminator
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ok_csv_path', type=str)
    parser.add_argument('ng_csv_path', type=str)
    parser.add_argument('ok_image_dir', type=str)
    parser.add_argument('ng_image_dir', type=str)
    parser.add_argument('--ok_test_nb_sample', '-otns', type=int, default=None)
    parser.add_argument('--ng_test_nb_sample', '-ntns', type=int, default=None)
    parser.add_argument('--batch_size', '-bs', type=int, default=234)
    parser.add_argument('--latent_dim', '-ld', type=int, default=16)
    parser.add_argument('--height', '-ht', type=int, default=64)
    parser.add_argument('--width', '-wd', type=int, default=64)
    parser.add_argument('--channel', '-ch', type=int, default=8)
    parser.add_argument('--model_path', '-mp', type=str, default="./params/epoch_200/model.ckpt")
    parser.add_argument('--result_dir', '-rd', type=str, default="./result")
    parser.add_argument('--nb_visualize_batch', '-nvb', type=int, default=1)
    parser.add_argument('--select_gpu', '-sg', type=str, default="0")

    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list=args.select_gpu, # specify GPU number
        allow_growth=True)
    )

    input_shape = (args.height, args.width, args.channel)

    autoencoder = AutoEncoder(input_shape, args.latent_dim,
                              is_training=False,
                              channel=args.channel)
    discriminator = Discriminator(is_training=False)

    aae = AAE(autoencoder, discriminator, is_training=False)
    aae.restore(args.model_path)

    result_dir_inlier = os.path.join(args.result_dir, "decoded/inlier")
    result_dir_outlier = os.path.join(args.result_dir, "decoded/outlier")

    image_sampler = ImageSampler(target_size=(args.width, args.height),
                                 channel=args.channel,
                                 is_training=False)

    data_generator_inlier = image_sampler.flow_from_csv(args.ok_csv_path, args.ok_image_dir, args.batch_size, shuffle=False, nb_sample=args.ok_test_nb_sample)
    df_inlier = get_encoded_save_decoded(aae,
                                         data_generator_inlier,
                                         args.latent_dim,
                                         result_dir_inlier,
                                         label='inlier',
                                         nb_visualize=args.nb_visualize_batch)

    data_generator_outlier = image_sampler.flow_from_csv(args.ng_csv_path, args.ng_image_dir, args.batch_size, shuffle=False, nb_sample=args.ng_test_nb_sample)
    df_outlier = get_encoded_save_decoded(aae,
                                          data_generator_outlier,
                                          args.latent_dim,
                                          result_dir_outlier,
                                          label='outlier',
                                          nb_visualize=args.nb_visualize_batch)

    df = pd.concat([df_inlier, df_outlier], ignore_index=True)
    os.makedirs(args.result_dir, exist_ok=True)
    df.to_csv(os.path.join(args.result_dir, "output.csv"), index=False)


def get_encoded_save_decoded(model, data_generator, latent_dim, result_dir, label, nb_visualize):
    os.makedirs(result_dir, exist_ok=True)
    df = None

    for index, image_batch in enumerate(data_generator()):
        print('Processing ... [{} / {}]'.format(index*data_generator.batch_size, data_generator.n), end='\r')
        current_df = data_generator.current_df
        encoded_batch = model.predict_latent_vectors_on_batch(image_batch)
        decoded_batch = model.predict_on_batch(image_batch)

        # store dataframe
        dict_ = dict(zip(['z_{}'.format(ld) for ld in range(latent_dim)],
                         [encoded_batch[:, ld] for ld in range(latent_dim)]))
        current_df = current_df.assign(**dict_)
        zeros = np.zeros_like(encoded_batch)
        distances = np.array([get_distance(e, z) for e, z in zip(encoded_batch, zeros)])
        current_df = current_df.assign(
            distance=distances,
            label=label
        )

        if index == 0:
            df = current_df
        else:
            df = pd.concat([df, current_df])
    return df


def get_distance(v1, v2):
    return np.sqrt(np.sum([(x - y) ** 2 for (x, y) in zip(v1, v2)]))


if __name__ == '__main__':
    main()
