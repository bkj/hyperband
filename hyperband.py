"""
    hyperband.py
    
    `hyperband` algorithm for hyper-parameter optimization
    
    Copied/adapted from https://people.eecs.berkeley.edu/~kjamieson/hyperband.html
    
    The algorithm is motivated by the idea that random search is pretty 
    good as far as black-box optimization goes, so let's try to do it faster.
    
"""

import sys
import numpy as np
import ujson as json
from math import log, ceil

class HyperBand:
    
    def __init__(self, model, max_iter=81, eta=3):
        
        self.model = model
        
        self.max_iter = max_iter
        self.eta = eta
        self.s_max = int(log(max_iter) / log(eta))
        self.B = (self.s_max + 1) * max_iter  
        
        self.best_obj = np.inf
    
    def run(self):
        for s in reversed(range(self.s_max + 1)):
            
            # initial number of configs
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s)) 
            
            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s) 
            
            # initial configs
            configs = [self.model.rand_config() for _ in range(n)] 
            
            for i in range(s + 1):
                r_i = r * self.eta ** i
                
                print >> sys.stderr, "\n -- %d configs @ %d iterations -- \n" % (len(configs), int(round(r_i)))
                
                results = []
                for config in configs:
                    print >> sys.stderr, "Config: %s" % config
                    
                    res = self.model.config2loss(iters=r_i, config=config)
                    results.append(res)
                    
                    self.best_obj = min(res['obj'], self.best_obj)
                    print >> sys.stderr, \
                        "Current: %f | Best: %f" % (float(res['obj']), self.best_obj)
                    
                    print json.dumps(res)
                    sys.stdout.flush()
                
                results = sorted(results, key=lambda x: x['obj'])
                n_keep = int(n * self.eta ** (-i - 1))
                configs = [result['config'] for result in results[:n_keep]]

