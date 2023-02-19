# Probing Predictions on OOD Images via Nearest Categories

This repository contains the code of the experiments in the paper

[Probing Predictions on OOD Images via Nearest Categories (TMLR)](https://openreview.net/forum?id=fTNorIvVXG)

Authors: [Yao-Yuan Yang](http://yyyang.me), [Cyrus Rashtchian](http://www.cyrusrashtchian.com), [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/), [Kamalika Chaudhuri](http://cseweb.ucsd.edu/~kamalika/)

## Abstract

We study out-of-distribution (OOD) prediction behavior of neural networks when they classify images from unseen classes or corrupted images. To probe the OOD behavior, we introduce a new measure, _nearest category generalization_ (NCG), where we compute the fraction of OOD inputs that are classified with the same label as their nearest neighbor in the training set. Our motivation stems from understanding the prediction patterns of adversarially robust networks, since previous work has identified unexpected consequences of training to be robust to norm-bounded perturbations. We find that robust networks have consistently higher NCG score than natural training, even when the OOD data is much farther away than the robustness radius. This implies that the local regularization of robust training has a significant impact on the network's decision regions. We replicate our findings using many datasets, comparing new and existing training methods. Overall, adversarially robust networks resemble a nearest neighbor classifier when it comes to OOD data.


## Setup

### Install required libraries
```bash
pip install -r ./requirements.txt
```

### Install FAISS

https://github.com/facebookresearch/faiss

### Install cleverhans from its github repository
```
pip install --upgrade git+https://github.com/tensorflow/cleverhans.git#egg=cleverhans
```

## Scripts

Notebooks
- Precompute
  - [notebooks/out_of_sample.ipynb](notebooks/out_of_sample.ipynb): Precalculate the nearest neighbors and save to file. This notebook needs to be run first.
- Table figure generation
  - [notebooks/ncg_on_all.ipynb](notebooks/ncg_on_all.ipynb): Chi-square test in Table 1
  - [notebooks/robustness.ipynb](notebooks/robustness.ipynb): Figure 3.
  - [notebooks/feature_space_ncg.ipynb](notebooks/feature_space_ncg.ipynb): Tables 1 and 2; Appendix D.4.2 and D.6.2 and Tables 17 and 19
  - [notebooks/pixel_space_ncg.ipynb](notebooks/pixel_space_ncg.ipynb): Tables 1 and 2; Appendix D.5 and D.6.2 and Tables 18 and 20
  - [notebooks/ncg_dists.ipynb](notebooks/ncg_dists.ipynb): Figure 5 and Appendix D.6.3
  - [notebooks/imgnet-c.ipynb](notebooks/imgnet-c.ipynb): Imgnet100 and pixel/feature space of Table 4 and Figure 6, and Appendix D.3, D.6.1 and D.7
  - [notebooks/cifar10-c.ipynb](notebooks/cifar10-c.ipynb): CIFAR10/100 and pixel/feature space of Table 4 and Figure 6, and Appendix D.3, D.6.1 and D.7
  - [notebooks/in-distribution-ncg.ipynb](notebooks/in-distribution-ncg.ipynb): Appendix B.1
  - [notebooks/ncg_on_pretrained.ipynb](notebooks/ncg_on_pretrained.ipynb): Appendix D.4.1 and D.4.2
  - [notebooks/visualize-Mnist.ipynb](notebooks/visualize-Mnist.ipynb): Appendix D.6.2

Other scripts
- [params.py](params.py): listed all parameter setups to run
- [scripts/get_preds_on_corrupted.py](scripts/get_preds_on_corrupted.py)
- [scripts/get_preds_on_corrupted_feature.py](scripts/get_preds_on_corrupted_feature.py)
- [scripts/calc_pretrained_reprs.py](scripts/calc_pretrained_reprs.py)
- [scripts/calc_ood_reprs.py](scripts/calc_ood_reprs.py)


### Parameters

The network architectures are defined in [lolip/models/torch_utils/archs/](lolip/models/torch_utils/archs/)

### Algorithm implementations

#### Training algorithms

- [TRADES](lolip/models/torch_utils/losses/trades.py)
- [adversarial training (AT)](lolip/models/losses/__init__.py)

### Example options for model parameter

arch: ("CNN002", "WRN_40_10", "ResNet50)
attack: "cwl2" (just set it to "cwl2", it won't effect the result)

- ce-vtor2-{arch}
- TRADES (beta=6): trades6ce-vtor2-{arch}
  - robust radius 2: set eps to 2.0
- AT: advce-vtor2-{arch}

### Dataset options

- "mnistwo9": MNIST with the digit 9 as the unseen category
- "cifar10wo0": CIFAR10 with airplane as the unseen category
- "cifar100coarsewo0": CIFAR100 coarse labeling with aquatic mammal as the unseen category
- "aug10-imgnet100wo0": ImgNet100 with Americal robin as the unseen category,
  the implementation of data augmentation can be found [here](lolip/models/torch_utils/data_augs.py#L117)

## Examples

Run natural training with CNN002 on the MNIST dataset and digit 9 as unseen category.
Running with L2 distance, batch size is $128$, and using the SGD optimizer (default parameters).
The learned model will be saved to
``models/out_of_sample/pgd-128-mnistwo9-70-1.0-0.01-ce-vtor2-CNN002-0.9-2-sgd-0-0.0-ep0070.pt``, and the result will be saved to
``results/oos_repr/cwl2-128-mnistwo9-70-1.0-0.01-ce-vtor2-CNN002-0.9-2-sgd-0-0.0.pkl``.

```bash
python ./main.py --experiment oos_repr \
  --norm 2 --eps 1.0 --attack cwl2 --epochs 70 --random_seed 0 \
  --optimizer sgd --momentum 0.9 --batch_size 128 --learning_rate 0.01 --weight_decay 0. \
  --dataset mnistwo9 \
  --model ce-vtor2-CNN002
```

Compute the closest adversarial example for TRADES(beta=6, r=2) trained with
ADAM, initialal learning rate = 0.01, and batch size = 128 on CIFAR10 with
airplane (the first class) as the unseen category.
```bash
python ./main.py --experiment ood_robustness_correct \
  --norm 2 --eps 2.0 --attack cwl2 --epochs 70 --random_seed 0 \
  --optimizer adam --momentum 0. --batch_size 128 --learning_rate 0.01 --weight_decay 0. \
  --dataset cifar10wo0 \
  --model trades6ce-vtor2-WRN_40_10
```
