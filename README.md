# Why Do Self-Supervised Models Transfer? Investigating the Impact of Invariance on Downstream Tasks
This repository contains the official code for the paper [Why Do Self-Supervised Models Transfer? Investigating the Impact of Invariance on Downstream Tasks](https://arxiv.org/abs/2111.11398).

## Requirements
This codebase has been tested with the following package versions:
```
python=3.8.8
torch=1.9.0+cu102
torchvision=0.10.0+cu102
PIL=8.1.0
numpy=1.19.2
scipy=1.6.1
tqdm=4.57.0
sklearn=0.24.1
albumentations=1.0.3
```

## Prepare data
There are several classes defined in the `datasets` directory. The data is expected in a directory name `data`, located on the same level as this repository. Below is an outline of the expected file structure:
```
data/
    imagenet/
    CIFAR10/
    300W/
    ...
ssl-invariances/
    datasets/
    models/
    readme.md
    ...
```

For synthetic invariance evaluation, get the ILSVRC2012 validation data from https://image-net.org/ and store in `../data/imagenet/val/`.

For real-world invariances, download the following datasets: [Flickr1024](https://yingqianwang.github.io/Flickr1024/), [COIL-100](https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php), [ALOI](https://aloi.science.uva.nl/), [ALOT](https://aloi.science.uva.nl/public_alot/#:~:text=ALOT%20is%20a%20color%20image,illumination%20color%20for%20each%20material.), [DaLI](https://esslab.jp/~ess/en/data/dali_data/), [ExposureErrors](https://github.com/mahmoudnafifi/Exposure_Correction), [RealBlur](http://cg.postech.ac.kr/research/RealBlur/).

For extrinsic invariances, get [Causal3DIdent](https://zenodo.org/record/4784282#.YZ9vsvz7Tr4).

Finally, our downstream datasets are [CIFAR10](https://pytorch.org/vision/stable/datasets.html), [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/), [Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), [300W](https://ibug.doc.ic.ac.uk/resources/300-W/), [CelebA](https://pytorch.org/vision/stable/datasets.html), [LSPose](http://sam.johnson.io/research/lsp.html).

## Pre-training models
We pre-train several models based on the [MoCo codebase](https://github.com/facebookresearch/moco).

To set up a version of the codebase that can pre-train our models, first clone the MoCo repo onto the same level as this repo:

```bash
git clone https://github.com/facebookresearch/moco
```
This should be the resulting file structure:
```
data/
ssl-invariances/
moco/
```
Then copy the files from `ssl-invariances/pretraining/` into the cloned repo:
```bash
cp ssl-invariances/pretraining/* moco/
```
Finally, to run our models, enter the cloned repo by `cd moco` and run one of the following:
```bash
# train the Default model
python main_moco.py -a resnet50 --model default --lr 0.03 --batch-size 256 --mlp --moco-t 0.2 --cos --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ../data/imagenet

# train the Ventral model
python main_moco.py -a resnet50 --model ventral --lr 0.03 --batch-size 256 --mlp --moco-t 0.2 --cos --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ../data/imagenet

# train the Dorsal model
python main_moco.py -a resnet50 --model dorsal --lr 0.03 --batch-size 256 --mlp --moco-t 0.2 --cos --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ../data/imagenet

# train the Default(x3) model
python main_moco.py -a resnet50w3 --model default --moco-dim 384 --lr 0.03 --batch-size 256 --mlp --moco-t 0.2 --cos --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ../data/imagenet
```

This will train the models for 200 epochs and save checkpoints. When training has completed, the final model checkpoint, e.g. `default_00199.pth.tar`, should be moved to `ssl-invariances/models/default.pth.tar`for use in evaluation in the below code.

The rest of this codebase assumes these final model checkpoints are located in a directory called `ssl-invariances/models/` as shown below.
```
ssl-invariances/
    models/
        default.pth.tar
        default_w3.pth.tar
        dorsal.pth.tar
        ventral.pth.tar
```

## Synthetic invariance
To evaluate the Default model on grayscale invariance, run: 
```bash
python eval_synthetic_invariance.py --model default --transform grayscale ../data/imagenet
```
This will compute the mean and covariance of the model's feature space and save these statistics in the `results/` directory. These are then used to speed up future invariance computations for the same model.

## Real-world invariance
To evaluate the Ventral model on COIL100 viewpoint invariance, run: 
```bash
python eval_realworld_invariance.py --model ventral --dataset COIL100
```

## Extrinsic invariance on Causal3DIdent
To evaluate the Dorsal model on Causal3DIdent object x position prediction, run: 
```bash
python eval_causal3dident.py --model dorsal --target 0
```

## Downstream performance
To evaluate the combined Def+Ven+Dor model on 300W facial landmark regression, run: 
```bash
python eval_downstream.py --model default+ventral+dorsal --dataset 300w
```

## Citation
If you find our work useful for your research, please consider citing our paper:
```bibtex
@misc{ericsson2021selfsupervised,
      title={Why Do Self-Supervised Models Transfer? Investigating the Impact of Invariance on Downstream Tasks}, 
      author={Linus Ericsson and Henry Gouk and Timothy M. Hospedales},
      year={2021},
      eprint={2111.11398},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
If you have any questions, feel welcome to create an issue or contact Linus Ericsson (linus.ericsson@ed.ac.uk).
