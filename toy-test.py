"""
    toy-test.py
"""

import numpy as np

class TestModel:
    
    def rand_config(self):
        """ Returns random configuration """
        return np.random.normal(0.5, 0.1, 1)[0]
        
    def eval_config(self, config, iters):
        """ Evaluates model w/ given configuration on validation data """
        return {
            "config" : config,
            "obj" : config / iters
        }

# --
# Run

if __name__ == "__main__":
    from hyperband import HyperBand
    model = TestModel()
    HyperBand(model).run()
