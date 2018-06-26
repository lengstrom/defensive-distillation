import robustml

from robustml_model import MODEL_PATH

import sys
import argparse
import tensorflow as tf
import numpy as np

import l0_attack

class CarliniAttack(robustml.attack.Attack):
    def __init__(self, sess, epsilon):
        self._model = make_model(MODEL_PATH)
        self._eps = epsilon
        self._sess = sess

    # this is super hacky
    def run(self, x, y):
        return modified_papernot_attack(x, y, 100, self._sess, self._model, self._eps)

