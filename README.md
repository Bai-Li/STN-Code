# Stability Training with Noise (STN)

This repository contains code to reproduce results from the paper:

**Certified Adversarial Robustness with Additive Gaussian Noise** <br>
*Bai Li, Changyou Chen, Wenlin Wang, Lawrence Carin* <br>
ArXiv report: https://arxiv.org/abs/1809.03113

###### REQUIREMENTS

The code was tested with Python 3.6.8, Tensorflow 1.8.0 and Keras 2.2.4.
To compute certified bound, install R, rpy2 and R package *MultinomialCI*.

To evaluate robustness to various attacks, we use

```
python -m simple_eval [dataset] [attack] [source_model] [target_model] [norm] [size]
```

For Example, to run attacks on CIFAR10 with l2 attacks bounded by 4

```
python -m simple_eval cifar10 so models/model_CIFAR10 models/modelA_CIFAR10 l2 4

```

It reports the classification accuracy of model_CIFAR10 against white-box attack and transferred attack from modelA_CIFAR10.
When l2 norm is selected, we use the Carlini and Wagner attack. When linf norm is selected, we use the PGD attack.
