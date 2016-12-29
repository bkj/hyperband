"""
    hyperband.py
    
    `hyperband` algorithm for hyper-parameter optimization
    
    Copied/adapted from https://people.eecs.berkeley.edu/~kjamieson/hyperband.html
    
    The algorithm is motivated by the idea that random search is pretty 
    good as far as black-box optimization goes, so let's try to do it faster.
"""

from math import log, ceil

class HyperBand:
    
    def __init__(self, model, max_iter=81, eta=3):
        
        self.model = model
        
        self.max_iter = max_iter
        self.eta = eta
        self.s_max = int(log(max_iter) / log(eta))
        self.B = (self.s_max + 1) * max_iter  
        
        self.history = []
        self.total_iters = 0
    
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
                
                val_losses = []
                for config in configs:
                    self.total_iters += r_i
                    print config
                    val_loss = self.model.config2loss(iters=r_i, config=config)
                    val_losses.append(val_loss)
                
                these_results = zip(configs, val_losses, [r_i] * len(configs))
                these_results = sorted(these_results, key=lambda x: x[1])
                self.history += these_results
                
                n_keep = int(n * self.eta ** (-i - 1))
                configs = [config for config,loss,iters in these_results[:n_keep]]

