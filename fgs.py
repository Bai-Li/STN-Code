import tensorflow as tf
import keras.backend as K
from attack_utils import gen_grad
import numpy as np
import pdb


def symbolic_fgs(x, grad, eps=0.3, clipping=True, reverse=False):
    """
    FGSM attack.
    """

    # signed gradient
    normed_grad = K.sign(grad)

    # Multiply by constant epsilon
    scaled_grad = eps * normed_grad

    # Add perturbation to original example to obtain adversarial example
    if not reverse:
        adv_x = K.stop_gradient(x + scaled_grad)
    else:
        adv_x = K.stop_gradient(x - scaled_grad)

    if clipping:
        adv_x = K.clip(adv_x, 0, 1)
    return adv_x


def symbolic_alpha_fgs(x, grad, eps, alpha, clipping=True):
    """
    R+FGSM attack.
    """

    # signed gradient
    normed_grad = K.sign(grad)

    # Multiply by constant epsilon
    scaled_grad = (eps-alpha) * normed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = K.stop_gradient(x + scaled_grad)

    if clipping:
        adv_x = K.clip(adv_x, 0, 1)
    return adv_x

def iter_fgs(model, x, y, steps, eps, alpha):
    """
    PGD / I-FGSM attack.
    """

    adv_x = x
   
    # iteratively apply the FGSM with small step size
    for i in range(steps):
        logits = model(adv_x)
        grad = gen_grad(adv_x, logits, y)

        adv_x = symbolic_fgs(adv_x, grad, alpha, True)
        adv_x = tf.clip_by_value(adv_x, x-eps, x+eps)
    return adv_x

def l2_norm(x):
      return K.sqrt(tf.reduce_sum(K.square(x), axis=(1,2,3), keepdims=True))

def so(model, x, y, steps, eps, alpha=0, norm="l2", sd=0.0):
    adv_x = x 
    for i in range(steps):
        total_grad = 0
        for j in range(10):
            temp_adv_x = adv_x + tf.random_normal(stddev=sd, shape = tf.shape(x), seed=42)
            logits = model(temp_adv_x)
            if norm == "linf":
                grad = gen_grad(temp_adv_x,
                                logits, y, loss='logloss')
            elif norm == "l2":
                grad = gen_grad(temp_adv_x,
                                logits, y, loss='cw')
            total_grad += grad
        
        if norm == "linf":
            normed_grad = K.sign(total_grad)
            adv_x += alpha * normed_grad
            adv_x = tf.clip_by_value(adv_x, x-eps, x+eps)
        if norm == "l2":
            grad_norm = tf.clip_by_value(l2_norm(total_grad), 1e-8, np.inf)
            adv_x += 2.5 * eps / steps * total_grad/grad_norm
            dx = adv_x - x
            dx_norm = tf.clip_by_value(l2_norm(dx), 1e-8, np.inf)
            dx_final_norm = tf.clip_by_value(dx_norm, 0, eps)
            adv_x = x + dx_final_norm * dx / dx_norm
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    return adv_x

