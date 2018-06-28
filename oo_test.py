import numpy as np
import pandas as pd
from hashlib import md5
from hyperband import HyperBand

class Worker:
    def __init__(self, param):
        self.param = param
        self.epochs = 1
    
    def train_until(self, epochs):
        print(epochs)
        while self.epochs < epochs:
            self.epochs += 1
    
    def eval(self):
        return self.param
    
    @classmethod
    def init_random(cls):
        """ Returns random configuration """
        c = np.random.normal(0.5, 0.1, 1)[0]
        return cls(c)


class Trainer:
    def __init__(self, rand_worker_fn):
        self.configs = {}
        self.rand_worker_fn = rand_worker_fn
    
    def random_configs(self, s, n):
        for nn in range(n):
            self.configs[(s, nn)] = self.rand_worker_fn()
        
        return [(s, nn) for nn in range(n)]
    
    def eval_config(self, config, iters):
        """ Evaluates model w/ given configuration on validation data """
        model = self.configs[config]
        model.train_until(iters)
        obj = model.eval()
        return {
            "config" : config,
            "obj"    : obj,
        }


if __name__ == "__main__":
    model = Trainer(rand_worker_fn=Worker.init_random)
    h = HyperBand(model, max_iter=81, eta=3)
    h.run()
    
    print(pd.value_counts([m.epochs for m in model.configs.values()]))
    
    # check that this looks right vs. 
    #   https://people.eecs.berkeley.edu/~kjamieson/hyperband.html

