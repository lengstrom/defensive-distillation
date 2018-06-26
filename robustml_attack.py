import robustml


import sys
import argparse
import tensorflow as tf
from model import make_model
import numpy as np

from l0_attack import modified_papernot_attack

class CarliniAttack(robustml.attack.Attack):
    def __init__(self, sess, epsilon):
        self._eps = epsilon
        self._sess = sess

    def run(self, x, y):
        return modified_papernot_attack(x, y, self._sess, self._eps)


    
