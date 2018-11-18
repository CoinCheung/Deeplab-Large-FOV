# Deeplab-LFOV


This is my implementation of [Deeplab-LargeFOV](https://arxiv.org/pdf/1412.7062.pdf).


## Get Pascal VOC2012 dataset
Run the script to download the dataset:
```
    $ sh get_dataset.sh
```

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
1. The authors claimed to use the weight pretrained on imagenet to initialize the model, and they provided the [initial model](http://www.cs.jhu.edu/~alanlab/ccvl/init_models) which is exported from the framework of caffe. So far, I found no easy way to convert caffe pretrained weights to that of pytorch, so I initialize the model with the following methods as a remedy:   

* Initialize weights of the common parts of the 'deeplab vgg' and 'pytorch standard vgg' with the weights from pytorch model zoo.

* Initialize the weights of the extraly appended layers with the [msra](https://arxiv.org/abs/1502.01852) normal distribution random numbers.

* Implement warmup with the strategy mentioned in this [paper](https://arxiv.org/abs/1706.02677)

With these methods, my naive model achieves the performance quite close to  the performance in the paper.
