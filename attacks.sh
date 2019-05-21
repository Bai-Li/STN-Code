#!/bin/sh
for l in 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0
do
    python -m simple_eval mnist so models/model_mnist models/modelA_mnist l2 $l
done

for l in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
do
    python -m simple_eval mnist so models/model_mnist models/modelA_mnist linf $l
done

for l in 0.2 0.4 0.6 0.8 1.0 1.2 1.4
do
    python -m simple_eval cifar10 so models/model_cifar10 models/modelA_cifar10 l2 $l
done

for l in 2.0/255.0 4.0/255.0 6.0/255.0 8.0/255.0 10.0/255.0 12.0/255.0 14.0/255.0 16.0/255.0 18.0/255.0 20.0/255.0
do
    python -m simple_eval cifar10 so models/model_cifar10 models/modelA_cifar10 linf $l
done
