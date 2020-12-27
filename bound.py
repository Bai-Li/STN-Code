import numpy as np
import tensorflow as tf
import keras.backend as K
from mnist import *
from cifar10 import load_data, set_flags, load_model
from attack_utils import gen_grad
from tf_utils_adv import batch_eval
from os.path import basename
import numpy as np

from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
multici = importr("MultinomialCI")

def isRobust(prob, sd, epsilon):
    fv = FloatVector(sorted(prob)[::-1])
    ci = np.array(multici.multinomialCI(fv, 0.05))
    qi = ci[0,0]
    qj = ci[1,1]
    alpha = np.linspace(1.01,2.0, 100)
    # pdb.set_trace()
    bound = (-np.log(1-qi-qj+2*((qi**(1-alpha)+qj**(1-alpha))/2)**(1/(1-alpha)))/alpha).max()
    # return np.sqrt(bound*2.*sd**2)
    if bound > epsilon**(2.)/2./sd**(2.):
        return np.array([True, np.sqrt(bound*2.)*sd])
    else:
        return np.array([False, np.sqrt(bound*2.)*sd])

def main(src_model_name, eps):
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
    logits = src_model(x)

    for sd in [0.4, 0.5, 0.6]:
        one_hot = np.zeros(shape=(len(Y_test), 10))
        for i in range(100):
            logits_np = batch_eval([x, y], [logits], [X_test + np.random.normal(scale=sd, size = X_test.shape), Y_test])[0]
            one_hot[np.arange(len(Y_test)), logits_np.argmax(axis=1)] += 1
        robust = np.apply_along_axis(func1d=isRobust, axis=1, arr=one_hot, sd=sd, epsilon=eps)
        total_robust = np.sum(np.logical_and(robust[:,0]==True, one_hot.argmax(axis=1)==Y_test.argmax(axis=1)))/100.
        accuracy = np.sum(one_hot.argmax(axis=1)==Y_test.argmax(axis=1))/100.
        with open('bound_' + src_model_name + '_bound.txt', 'a') as log:
            log.write("Ave bound is {} at sigma = {}\n".format(np.mean(robust[:,1]), sd))
            log.write("Accuracy: {}, Robust accuracy: {}, l={}\n".format(accuracy, total_robust, eps))
        # print("Ave bound is {} at sigma = {}".format(np.mean(robust[:,1]), sd))
        # print("Accuracy: {}, Robust accuracy: {}, l={}".format(accuracy, total_robust, eps))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="name of data sets",
                        choices=["mnist", "cifar10"])
    parser.add_argument("src_model", help="source model for attack")
    parser.add_argument("eps", type=float, default=0.1,
                        help="attack scale")

    args = parser.parse_args()
    main(args.src_model, args.eps)
