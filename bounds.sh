#!/bin/sh
for l in 1.4 1.6 1.8 2.0 2.2
do
    python -m bound models/model_mnist $l
done
