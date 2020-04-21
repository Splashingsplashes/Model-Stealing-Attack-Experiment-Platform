Modified from https://github.com/tribhuvanesh/knockoffnets (**Tribhuvanesh Orekondy, Bernt Schiele Mario Fritz**)

# Model Stealing Attack Experiment Platform



## Installation

### Environment
  * Python 3.6
  * Pytorch 1.1

Can be set up as:
```bash
$ conda env create -f environment.yml   # anaconda; or
$ pip install -r requirements.txt       # pip
```

### Set up

1. Create a data/ directory
2. Download Caltech256 dataset into this directory
   * Caltech256 ([Link](http://www.vision.caltech.edu/Image_Datasets/Caltech256/). Images in `data/256_ObjectCategories/<classname>/*.jpg`)
3. Create a models/victim/caktech-256/ directory
4. Download victim model into this directory
   * [Caltech256](https://datasets.d2.mpi-inf.mpg.de/orekondy19cvpr/victim_models/caltech256-resnet34.zip) (Accuracy = 78.4%)
5. Set path (might need to set path each time the terminal is restarted)
```bash
export PYTHONPATH="${PYTHONPATH}:[path to the directory]"
```

### Synthetic samples
* -d = -1 for running on CPU based environment
* -d >= 0 for GPU based environment

1. Adaptively generate samples

format: 
$ python knockoff/adversary/daptive_transfer.py models/victim/VIC_DIR \
        --out_dir models/adversary/ADV_DIR --budget BUDGET \
        model_arch testdataset --queryset QUERY_SET --batch_size 8 -d DEV_ID (-1 for cpu) --pretrained imagenet

```bash
python ./knockoff/adversary/adaptive_transfer.py models/victim/caltech256-resnet34 --out_dir models/adversary/caltech256-resnet34 --budget 1000 resnet34 Caltech256 --queryset Caltech256 --batch_size 8 -d -1 --pretrained imagenet
```

2. Generate random samples

format: 
$ python knockoff/adversary/transfer.py models/victim/VIC_DIR \
        --out_dir models/adversary/ADV_DIR --budget BUDGET \
        model_arch testdataset --queryset QUERY_SET --batch_size 8 -d DEV_ID
  example: 
```bash
python ./knockoff/adversary/transfer.py models/victim/caltech256-resnet34 --out_dir models/adversary/caltech256-resnet34 --budget 1000 resnet34 --queryset Caltech256 --batch_size 8 -d -1 --pretrained imagenet
```
 
3. Generate adversarial samples(fgsm or jsma)
format: 
$ python knockoff/adversary/transfer.py models/victim/VIC_DIR \
        --out_dir models/adversary/ADV_DIR --budget BUDGET \
        model_arch --algo ALGO --eps (for fgsm/jsma) 
        testdataset --queryset QUERY_SET --batch_size 8 -d DEV_ID --pretrained
```bash
python ./knockoff/adversary/jacobian_transfer.py models/victim/caltech256-resnet34 --out_dir models/adversary/caltech256-resnet34 --budget 1000 resnet34 --algo fgsm --eps 0.5 Caltech256 --queryset Caltech256 --batch_size 8 -d -1 --pretrained imagenet
```

### Train Symthetic Samples
1. Train adaptively generated samples
```bash
python ./knockoff/adversary/train.py adaptive models/adversary/caltech256-resnet34 resnet34 Caltech256 --budgets 1000 -d -1 --pretrained imagenet --log-interval 1000 --epochs 20 --lr 0.1 
```

2. Train randomly generated samples
```bash
python ./knockoff/adversary/train.py random models/adversary/caltech256-resnet34 resnet34 Caltech256 --budgets 1000 -d -1 --pretrained imagenet --log-interval 1000 --epochs 20 --lr 0.1 
```
3. Train adversarial sample (fgsm)
```bash
python ./knockoff/adversary/jacobian_train.py models/adversary/caltech256-resnet34 resnet34 Caltech256 --budgets 1000 --algo fgsm --eps 0.5 -d -1 --pretrained imagenet --log-interval 1000 --epochs 20 --lr 0.1 -w 4
```
