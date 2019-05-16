import numpy as np
import tensorflow as tf
import keras.backend as K
from mnist import *
from cifar10 import load_data, set_flags, load_model
from fgs import symbolic_fgs, iter_fgs, momentum_fgs, so
from attack_utils import gen_grad
from tf_utils_adv import tf_test_acc, batch_eval
from os.path import basename
import pdb

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def main(attack, src_model_name, target_model_names):
    np.random.seed(0)
    tf.set_random_seed(0)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    if args.dataset == "mnist":
        K.set_image_data_format('channels_last')
        set_mnist_flags()
        x = K.placeholder((None,
                       FLAGS.IMAGE_ROWS,
                       FLAGS.IMAGE_COLS,
                       FLAGS.NUM_CHANNELS
                       ))
        y = K.placeholder((None, FLAGS.NUM_CLASSES))
        _, _, X_test, Y_test = data_mnist()
        # source model for crafting adversarial examples
        src_model = load_model_mnist(src_model_name)
        sd = 0.7
    
    elif args.dataset == "cifar10":
        set_flags(20)
        K.set_image_data_format('channels_first')
        x = K.placeholder((None,
                       FLAGS.NUM_CHANNELS,
                       FLAGS.IMAGE_ROWS,
                       FLAGS.IMAGE_COLS
                       ))
        y = K.placeholder((None, FLAGS.NUM_CLASSES))
        _, _, X_test, Y_test = load_data()
        # source model for crafting adversarial examples
        src_model = load_model(src_model_name)
        sd = 100./255.

    # model(s) to target
    target_models = [None] * len(target_model_names)
    for i in range(len(target_model_names)):
        target_models[i] = load_model(target_model_names[i])

    # simply compute test error
    if attack == "test":
        acc = tf_test_acc(src_model, x, X_test, Y_test)
        print('{}: {:.1f}'.format(basename(src_model_name), acc))

        for (name, target_model) in zip(target_model_names, target_models):
            acc = tf_test_acc(target_model, x, X_test, Y_test)
            print('{}: {:.1f}'.format(basename(name), acc))
        return

    eps = args.eps

    # take the random step in the RAND+FGSM
    if attack == "rfgs":
        X_test = np.clip(
            X_test + args.alpha * np.sign(np.random.randn(*X_test.shape)),
            0.0, 1.0)
        eps -= args.alpha

    logits = src_model(x)
    grad = gen_grad(x, logits, y)

    # FGSM and RAND+FGSM one-shot attack
    if attack in ["fgs", "rfgs"]:
        adv_x = symbolic_fgs(x, grad, eps=eps)

    # iterative FGSM
    if attack == "pgd":
        adv_x = iter_fgs(src_model, x, y, steps=args.steps, eps=eps, alpha=eps/10.0)
    
    if attack == 'mim':
        adv_x = momentum_fgs(src_model, x, y, eps=eps)

    if attack == 'so':
        adv_x = so(src_model, x, y, steps=args.steps, eps=eps, alpha=eps/10.0, norm="l2", sd=sd)

    print('start')
    # compute the adversarial examples and evaluate
    X_adv = batch_eval([x, y], [adv_x], [X_test, Y_test])[0]
    # pdb.set_trace()
    print('-----done----')
    # white-box attack
    acc = tf_test_acc(src_model, x, X_adv, Y_test, sd=sd, num_iter=10)
    with open('attacks.txt', 'a') as log:
            log.write('{}->{}: {:.1f}, size = {:.4f}\n'.format(basename(src_model_name), basename(src_model_name), acc, eps))

    # black-box attack
    for (name, target_model) in zip(target_model_names, target_models):
        acc = tf_test_acc(target_model, x, X_adv, Y_test, sd=sd, num_iter=10)
        with open('attacks.txt', 'a') as log:
            log.write('{}->{}: {:.1f}, size = {:.4f}\n'.format(basename(src_model_name), basename(name), acc, eps))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="name of data sets",
                        choices=["mnist", "cifar10"])
    parser.add_argument("attack", help="name of attack",
                        choices=["test", "fgs", "pgd", "rfgs", "mim","so"])
    parser.add_argument("src_model", help="source model for attack")
    parser.add_argument('target_models', nargs='*',
                        help='path to target model(s)')
    parser.add_argument("norm", help="type of norms",
                        choices=["linf", "l2"])
    parser.add_argument("eps", type=float, default=2.0,
                        help="attack scale")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="RAND+FGSM random perturbation scale")
    parser.add_argument("--steps", type=int, default=20,
                        help="Iterated FGS steps for PGD")

    args = parser.parse_args()
    main(args.attack, args.src_model, args.target_models)
