import robustml
import tensorflow as tf

TEMPERATURE = 100
MODEL_PATH = 'models/distilled'

class DefensiveDistillation(robustml.model.Model):
    def __init__(self, sess):
        self._sess = sess
        self._input = tf.placeholder(tf.float32, (1, 28, 28, 1))

        model = make_model(MODEL_PATH)
        randomized = defend(input_expanded)

        self._logits = tf.nn.softmax(model(img+delta)/TEMPERATURE)
        self._predictions = tf.reduce_max(self._logits, axis=1)
        
        self._dataset = robustml.dataset.MNIST((1, 28, 28, 1))
        self._threat_model = robustml.threat_model.L0(epsilon=112)

    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model

    def classify(self, x):
        return self._sess.run(self._predictions, {self._input: x})[0]

    # exposne internals for white box attacks
    @property
    def input(self):
        return self._input

    @property
    def logits(self):
        return self._logits

    @property
    def predictions(self):
        return self._predictions
