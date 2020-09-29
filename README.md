# Semantic Segmentation with MobileNetV3

This is the training code associated with [FastSeg](https://github.com/ekzhang/fastseg). It is based on a fork of Nvidia's [semantic-segmentation](https://github.com/NVIDIA/semantic-segmentation) monorepository.

See the original repository for full details about their code. This README only includes relevant information about training **MobileNetV3 + LR-ASPP** on Cityscapes data.

## Installation

- The code is tested with PyTorch 1.5-1.6 and Python 3.7 or later.
- You can use ./Dockerfile to build an image.

## Download/Prepare Data

First, update `config.py` to include an absolute path to a location to keep some large files, such as precomputed centroids:

```python
__C.ASSETS_PATH=<path_to_assets_dir>
```

If using Cityscapes, download Cityscapes data, then update `config.py` to set the path:

```python
__C.DATASET.CITYSCAPES_DIR=<path_to_cityscapes>
```

## Running the code

The instructions below make use of a tool called `runx`, which we find useful to help automate experiment running and summarization. For more information about this tool, please see [runx](https://github.com/NVIDIA/runx).
In general, you can either use the runx-style commandlines shown below. Or you can call `python train.py <args ...>` directly if you like.

## Train a model

Train cityscapes, using MobileNetV3-Large + LR-ASPP with fine annotations data.

```bash
> python -m runx.runx scripts/train_mobilenet_large.yml -i
```

The first time this command is run, a centroid file has to be built for the dataset. It'll take about 10 minutes. The centroid file is used during training to know how to sample from the dataset in a class-uniform way.

This training run should deliver a model that achieves 72.3 mIoU. If you download the resulting checkpoint `.pth` file from the logging directory, this can be loaded into `fastseg` for inference with the following code:

```python
from fastseg import MobileV3Large

model = MobileV3Large.from_pretrained('checkpoint.pth')
```

Under the default training configuration, this model should have 3.6M parameters and F=256 filters in the segmentation head. You can experiment with modifying the configuration in `scripts/train_mobilenet_large.yml` to train other models, such as those based on MobileNetV3-Small.

## Notes from Eric

Unfortunately, I am not able to take requests to train new models, as I do not currently have access to Nvidia DGX-1 compute resources. This training code is provided "as-is" for your benefit and research use.

Thanks to Andrew Tao (@ajtao) and Karan Sapra (@karansapra) for their support. They currently maintain the upstream repository.
