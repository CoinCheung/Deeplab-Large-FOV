# Deeplab-LFOV


This is my implementation of [Deeplab-LargeFOV](https://arxiv.org/pdf/1412.7062.pdf).


## Get Pascal VOC2012 and VOG_AUG dataset
Run the script to download the dataset:
```
    $ sh get_dataset.sh
```
This will download and extract the VOC2012 dataset together with the augumented VOC dataset. In the paper, the author used the aug version dataset to train the model and then test on the standard VOC2012 val set. 

## Train and evaluate the model
Just run th training script and evaluating script:
```
    $ python3 train.py
    $ python3 eval.py
```
The super-parameters are written in the code to make it looks simple and neat. Feel free to modify them as you like if you are using this project on other datasets.

## Inference
The script `infer.py` is an example how we could use the trained model to implement segmentation on the picture of `example.jpg`. Just run it to see the performance of this model: 
```
    $ python3 infer.py
```
And you will see the result picture.


## Notes:
1. The authors claimed to use the weight pretrained on imagenet to initialize the model, and they provided the [initial model](http://www.cs.jhu.edu/~alanlab/ccvl/init_models) which is exported from the framework of caffe. So far, I found no easy way to convert caffe pretrained weights to that of pytorch, so I employ the following tricks as a remedy:   

* Initialize weights of the common parts of the 'deeplab vgg' and 'pytorch standard vgg' with the weights from pytorch model zoo.

* Initialize the weights of the extraly appended layers with the [msra](https://arxiv.org/abs/1502.01852) normal distribution random numbers.

* Implement warmup with the strategy mentioned in this [paper](https://arxiv.org/abs/1706.02677)

* upsample the output logits of the model instead of downsample the ground truth as does in [deeplabv3](https://arxiv.org/abs/1706.05587).

* Enlarge the total training iter number to be 16000, and let lr jump by a factor of 0.1 at 10000th and 14000th iteration.

With these trick, my naive model trained on pure pascal VOC2012 training set(1467 images) achieves the performance quite close to the performance in the paper.
