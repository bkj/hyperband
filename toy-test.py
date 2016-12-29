"""
    toy-test.py
"""

import numpy as np
from hyperband import HyperBand

class TestModel:
    
    def rand_config(self):
        """ Returns random configuration """
        return np.random.normal(0.5, 0.1, 1)[0]
        
    def config2loss(self, iters, config):
        """ Evaluates model w/ given configuration on validation data """
        return config / iters

# --
# Run

model = TestModel()
hb = HyperBand(model)
hb.run()

hb.total_iters
hb.history[0]