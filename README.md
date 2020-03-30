# README

## Enviroment

* python 3.6
* tensorflow 1.8

## How to use

* In our experiment, we created CSV file in advance for reading dataset. When you exectute any dataset you want to use, please use `./image_sampler.py` instead of `solder/custom_image_sampler_*.py`.

* Example: OKsample.csv

| location | name                    | depth_0 | depth_1 | depth_2 | depth_3 | depth_4 | depth_5 | depth_6 | depth_7 |
|----------|-------------------------|---------|---------|---------|---------|---------|---------|---------|---------|
| 1-1     | 100000_Voxel_OK_N1 | L0     | L1     | L2     | L3     | L4     | L5     | L6     | L7     |
| 1-14    | 100000_Voxel_OK_N1 | L0     | L1     | L2     | L3     | L4     | L5     | L6     | L7     |
| 2-3    | 100000_Voxel_OK_N1 | L0     | L1     | L2     | L3     | L4     | L5     | L6     | L7     |
| 4-27   | 100000_Voxel_OK_N1 | L0     | L1     | L2     | L3     | L4     | L5     | L6     | L7     |
| 1-11   | 100000_Voxel_OK_N1 | L0     | L1     | L2     | L3     | L4     | L5     | L6     | L7     |
| 12-14    | 100000_Voxel_OK_N1 | L0     | L1     | L2     | L3     | L4     | L5     | L6     | L7     |
|...|...|...|...|...|...|...|...|...|...|

* Example: NGsample.csv

| location | name                    | depth_0 | depth_1 | depth_2 | depth_3 | depth_4 | depth_5 | depth_6 | depth_7 |
|----------|-------------------------|---------|---------|---------|---------|---------|---------|---------|---------|
| 1-1     | 100000_Voxel_NG_N1 | L0     | L1     | L2     | L3     | L4     | L5     | L6     | L7     |
| 1-14    | 100000_Voxel_NG_N1 | L0     | L1     | L2     | L3     | L4     | L5     | L6     | L7     |
| 2-3    | 100000_Voxel_NG_N1 | L0     | L1     | L2     | L3     | L4     | L5     | L6     | L7     |
| 4-27   | 100000_Voxel_NG_N1 | L0     | L1     | L2     | L3     | L4     | L5     | L6     | L7     |
| 1-11   | 100000_Voxel_NG_N1 | L0     | L1     | L2     | L3     | L4     | L5     | L6     | L7     |
| 12-14    | 100000_Voxel_NG_N1 | L0     | L1     | L2     | L3     | L4     | L5     | L6     | L7     |
|...|...|...|...|...|...|...|...|...|...|

### train (our experiment)

```python
python solder/train.py
train.py [-h] [--ok_nb_sample OK_NB_SAMPLE] [--ng_nb_sample NG_NB_SAMPLE]
                [--batch_size BATCH_SIZE] [--nb_epoch NB_EPOCH]
                [--latent_dim LATENT_DIM] [--height HEIGHT] [--width WIDTH]
                [--channel CHANNEL] [--save_steps SAVE_STEPS]
                [--visualize_steps VISUALIZE_STEPS] [--model_dir MODEL_DIR]
                [--result_dir RESULT_DIR] [--noise_mode NOISE_MODE]
                [--select_gpu SELECT_GPU]
                ok_csv_path ng_csv_path ok_image_dir ng_image_dir

```

### generate (our experiment)

```python
python solder/generate.py
generate.py [-h] [--ok_test_nb_sample OK_TEST_NB_SAMPLE] [--ng_test_nb_sample NG_TEST_NB_SAMPLE]
                   [--batch_size BATCH_SIZE] [--latent_dim LATENT_DIM]
                   [--height HEIGHT] [--width WIDTH] [--channel CHANNEL]
                   [--model_path MODEL_PATH] [--result_dir RESULT_DIR]
                   [--nb_visualize_batch NB_VISUALIZE_BATCH] [--select_gpu SELECT_GPU]
                   ok_csv_path ng_csv_path ok_image_dir ng_image_dir
```

### train (any image dataset)

```python
python train.py
train.py [-h] [--ok_nb_sample OK_NB_SAMPLE] [--ng_nb_sample NG_NB_SAMPLE]
                [--batch_size BATCH_SIZE] [--nb_epoch NB_EPOCH]
                [--latent_dim LATENT_DIM] [--height HEIGHT] [--width WIDTH]
                [--channel CHANNEL] [--save_steps SAVE_STEPS]
                [--visualize_steps VISUALIZE_STEPS] [--model_dir MODEL_DIR]
                [--result_dir RESULT_DIR] [--noise_mode NOISE_MODE]
                [--select_gpu SELECT_GPU]
                ok_image_dir ng_image_dir

```

### generate (any image dataset)

```python
python generate.py
generate.py [-h] [--batch_size BATCH_SIZE] [--latent_dim LATENT_DIM]
                   [--height HEIGHT] [--width WIDTH] [--channel CHANNEL]
                   [--model_path MODEL_PATH] [--result_dir RESULT_DIR]
                   [--nb_visualize_batch NB_VISUALIZE_BATCH] [--select_gpu SELECT_GPU]
                   ok_image_dir ng_image_dir
```