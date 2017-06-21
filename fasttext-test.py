"""
    fasttext-test.py
"""

import json
import numpy as np
import fasttext as ft
from hashlib import md5

class FasttextModel:
    
    def __init__(self, train_path, dev_path):
        self.train_path = train_path
        self.dev_path = dev_path
    
    def rand_config(self):
        return {
            # Fixed
            "input_file"         : self.train_path,
            "lr_freeze"          : 1,
            "save_vectors"       : 0,
            "save_label_vectors" : 0,
            
            # Random
            "loss"        : np.random.choice(["ns", "hs"]),
            "neg"         : int(2 ** np.random.uniform(2, 5.5)),
            "lr"          : float(2 ** np.random.uniform(-10, -1)),
            "dim"         : int(2 ** np.random.uniform(2, 8)),
            "min_count"   : int(2 ** np.random.choice(range(10))),
            "word_ngrams" : int(np.random.choice(range(1, 7)))
        }
    
    def eval_config(self, config, iters):
        config['epoch']  = iters
        config['output'] = './models/%s' % md5(json.dumps(config)).hexdigest()
        model = ft.supervised(**config)
        perf = model.test(self.dev_path)
        return {
            "obj" : -perf.precision,
            "config" : config,
            "iters" : iters
        }

# --
# Run

if __name__ == "__main__":
    from hyperband import HyperBand
    train_path = './data/emojis-train.txt'
    dev_path = './data/emojis-dev.txt'
    model = FasttextModel(train_path, dev_path)
    HyperBand(model).run()

