# How to deblock JPEG images in VGGFACE2 dataset

1. [Download](https://github.com/alexjc/neural-enhance/releases/download/v0.3/ne1x-photo-repair-0.3.pkl.bz2) the pretrained model for JPEG artifact removal
2. [Download](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) VGGFACE2 training set & test set and store in './dataset/VGGFACE2/train/' and './dataset/VGGFACE2/test/' respectively.
3. [Download](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) bb_landmark and store in './dataset/VGGFACE2/bb_landmark/'
4. Execute `deblock_jpeg.py' as follows:
```python deblock_jpeg.py```

*Note* that you should split `identity_info.csv` into multiple files if you want to run parallel on multi-processes:
```
python deblock_jpeg.py --identity-info=./identity_info1.csv # for process 1
python deblock_jpeg.py --identity-info=./identity_info2.csv # for process 2
...
```

**Output images will be stored in './deblocked' directory.**
