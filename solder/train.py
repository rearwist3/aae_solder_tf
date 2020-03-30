import os
import sys
import argparse
sys.path.append(os.getcwd())
from aae import AdversarialAutoEncoder as AAE
from solder.custom_image_sampler_train import ImageSampler
from noise_sampler import NoiseSampler
from utils.config import dump_config
from solder.autoencoder import AutoEncoder
from solder.discriminator import Discriminator
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ok_csv_path', type=str)
    parser.add_argument('ng_csv_path', type=str)
    parser.add_argument('ok_image_dir', type=str)
    parser.add_argument('ng_image_dir', type=str)
    parser.add_argument('--ok_nb_sample', '-ons', type=int, default=47736)
    parser.add_argument('--ng_nb_sample', '-nns', type=int, default=450)
    parser.add_argument('--batch_size', '-bs', type=int, default=234)
    parser.add_argument('--nb_epoch', '-e', type=int, default=200)
    parser.add_argument('--latent_dim', '-ld', type=int, default=16)
    parser.add_argument('--height', '-ht', type=int, default=64)
    parser.add_argument('--width', '-wd', type=int, default=64)
    parser.add_argument('--channel', '-ch', type=int, default=8)
    parser.add_argument('--save_steps', '-ss', type=int, default=1)
    parser.add_argument('--visualize_steps', '-vs', type=int, default=1)
    parser.add_argument('--model_dir', '-md', type=str, default="./params")
    parser.add_argument('--result_dir', '-rd', type=str, default="./result")
    parser.add_argument('--noise_mode', '-nm', type=str, default="normal")
    parser.add_argument('--select_gpu', '-sg', type=str, default="0")

    args = parser.parse_args()

    config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list=args.select_gpu, # specify GPU number
        allow_growth=True)
    )

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    dump_config(os.path.join(args.result_dir, 'config.csv'), args)
    input_shape = (args.height, args.width, args.channel)

    image_sampler = ImageSampler(target_size=(args.width, args.height),
                                 channel=args.channel,
                                 is_training=True)
    noise_sampler = NoiseSampler(args.noise_mode)

    autoencoder = AutoEncoder(input_shape, args.latent_dim,
                              is_training=True,
                              channel=args.channel)
    discriminator = Discriminator(is_training=True)

    aae = AAE(autoencoder, discriminator, is_training=True, kld_weight=args.kld_weight, config=config)

    aae.fit_generator(image_sampler.flow_from_csv(args.ok_csv_path,
                                                  args.ng_csv_path,
                                                  args.ok_image_dir,
                                                  args.ng_image_dir,
                                                  args.batch_size,
                                                  ok_nb_sample=args.ok_nb_sample,
                                                  ng_nb_sample=args.ng_nb_sample),
                      noise_sampler, nb_epoch=args.nb_epoch,
                      save_steps=args.save_steps, visualize_steps=args.visualize_steps,
                      result_dir=args.result_dir, model_dir=args.model_dir)


if __name__ == '__main__':


    main()
