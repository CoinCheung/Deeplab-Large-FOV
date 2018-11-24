# Deeplab-LFOV


This is my implementation of [Deeplab-LargeFOV](https://arxiv.org/pdf/1412.7062.pdf).


## Get Pascal VOC2012 and VOG_AUG dataset
Run the script to download the dataset:
```
    $ sh get_dataset.sh
```
This will download and extract the VOC2012 dataset together with the augumented VOC dataset. In the paper, the author used the aug version dataset to train the model and then test on the standard VOC2012 val set. 


## Train and evaluate the model
Just run the training script and evaluating script:
```
    $ python3 train.py --cfg config/pascal_voc_2012_multi_scale.py
    $ python3 eval.py --cfg config/pascal_voc_2012_multi_scale.py
```
Then you will see the mIOU result as reported by the author.

If you want to see a better result, use the configuration file `pascal_voc_aug_multi_scale.py`:
```
    $ python3 train.py --cfg config/pascal_voc_aug_multi_scale.py
    $ python3 eval.py --cfg config/pascal_voc_aug_multi_scale.py
```
This includes some other tricks beyond the paper which I find helpful to boost the performance.


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

* Implement warmup with the linear strategy mentioned in this [paper](https://arxiv.org/abs/1706.02677)

* Upsample the output logits of the model instead of downsample the ground truth as does in [deeplabv3](https://arxiv.org/abs/1706.05587).

* Enlarge the total training iter number to be 20000, and let lr jump by a factor of 0.1 at the 12500th and 17500th iteration.

* Do multi-scale training with image pyramid, with the images cropped to be (497, 497) and scaled with ratios [0.5, 0.75, 1].

With these tricks, my model achieves **71.84** mIOU(without crf), far better than the result reported by the authors and even suppressing the [vgg version deeplabv2](https://arxiv.org/pdf/1606.00915.pdf]).

## By the way
I also tried to use [mixup](https://arxiv.org/abs/1710.09412) in hopes of some further boost to the performance, but the result just gave a big slap on my face. I left the mixup code in the program and if you have special insights on this trick, please feel free to add a pull request or open an issue. Many thanks !
