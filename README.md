# PCB-EESRGAN
## Dataset
Download the dataset from [COWC](https://gdo152.llnl.gov/cowc/) and [HRIPCB](http://robotics.pkusz.edu.cn/resources/dataset/)

Then symlink your dataset root to the folder `dataset` with the command `ln -s xxx yyy`.

For example: 

```bash
ln -s /Users/jlwei/workspace/datasets/DetectionPatches_256x256/Potsdam_ISPRS ./dataset/
```

## Dataset preprocessing

### Only for HRIPCB

The image of HRIPCB dataset is too large, so you have to split it into tiles to match the input size of GAN.

This file `scripts/hripcb_split_image.py` split the file into tiles of 256x256, it will only keep the tiles with defects, the tiles without defect will not be saved. And it will use convert the annotations from xml to txt for easy input.

Don't forget to change the `dir` inside `scripts/hripcb_split_image.py`.

```bash
python scripts/hripcb_split_image.py
```

### For COWC and HRIPCB

The script `scripts/image_prepare.py` is used to generate a low resolution images based on the `up_scale` and it will split the dataset into train and test dataset based on the `valid_percent`.

Update the source and dir folder in `scripts/image_prepare.py`, then execute this script.

```bash
python scripts/image_prepare.py
```

## Edit config file

For the model EESRGAN, edit the `config_hripcb.json` and `config.json` file for the configuration.

For the model FRCNN without EESRGAN, you may have to edit `trainers/frcnn_trainer.py` directly.

## Train

### Config for different loss function

#### Fix pixel and fix feature weight

$$
Loss = W1 * L1(W) + W2 * L2(W) + others
$$

```python
config['train']['pixel_weight'] = 1
config['train']['feature_weight'] = 0.1
config['train']['learned_weight'] = False
```

#### Learned pixel and feature weight

$$
Loss = \frac{1}{2(\sigma1)^2}L1(W) + \frac{1}{2(\sigma2)^2}L2(W) + \log\sigma1\sigma2 + others
$$

```python
config['train']['pixel_sigma'] = 0.5
config['train']['feature_sigma'] = 0.5
config['train']['learned_weight'] = True
```

#### Learned pixel and feature weight, fixed intermediate_weight

$$
Loss = \frac{1}{2(\sigma1)^2}L1(W) + \frac{1}{2(\sigma2)^2}L2(W) + W3 * L3(W) + \log\sigma1\sigma2 + others
$$

```python
config['train']['pixel_sigma'] = 0.5
config['train']['feature_sigma'] = 0.5
config['train']['learned_weight'] = True
config['train']['intermediate_weight'] = 1
config['train']['intermediate_loss'] = True
```

#### Learned pixel, feature weight and intermediate_weight

$$
Loss = \frac{1}{2(\sigma1)^2}L1(W) + \frac{1}{2(\sigma2)^2}L2(W) + \frac{1}{2(\sigma3)^2}L3(W) + \log\sigma1\sigma2 + others
$$

```python
config['train']['pixel_sigma'] = 0.5
config['train']['feature_sigma'] = 0.5
config['train']['learned_weight'] = True
config['train']['intermediate_sigma'] = 0.5
config['train']['intermediate_loss'] = True
config['train']['intermediate_learned'] = True
```

### Train

Use `nohup *** &` to run Python script in background

Use `stdbuf` to write Python stdout to file immediately

Use `> log.log` to print the Python stdout to file

```bash
# Train EESRGAN with COWC dataset
nohup stdbuf -o0 python train.py > ./saved/logs/log.log &

# Train EESRGAN with HRIPCB dataset
nohup stdbuf -o0 python train_hripcb.py > ./saved_hripcb/logs/log.log &

# Train FRCNN with HRICPB dataset
nohup stdbuf -o0 python train_frcnn.py > ./saved_hripcb/logs/frcnn_log.log &
```













