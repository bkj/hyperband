"""
    fasttext-test.py
"""

import json
import numpy as np
import fasttext as ft
from hashlib import md5
from hyperband import HyperBand

class FasttextModel:
    
    def __init__(self, train_path, dev_path):
        self.train_path = train_path
        self.dev_path = dev_path
    
    def rand_config(self):
        config = {
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
        
        return config
        
    def config2loss(self, iters, config):
        config['epoch']  = iters
        config['output'] = './models/%s' % md5(json.dumps(config)).hexdigest()
        model = ft.supervised(**config)
        perf = model.test(self.dev_path)
        return -perf.precision

# --
# Run

train_path = './data/emojis-train.txt'
dev_path = './data/emojis-dev.txt'

ft_model = FasttextModel(train_path, dev_path)

ft_hb = HyperBand(ft_model)
ft_hb.run()

