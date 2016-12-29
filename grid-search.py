"""
    fasttext-grid.py
    
    Grid search on `fastText` parameter settings
"""

"""
# Run this first:

time fasttext init-dictionary -input yelp-2013-train-ft.txt \
    -output ./.dict \
    -minCount 10
"""


import sys
import json
import itertools
from copy import copy
from hashlib import md5

import fasttext as ft

# --
# Helpers

def run_id(params):
    keys = map(str, params.keys())
    values = map(str, params.values())
    return 'ft-%s' % md5(' '.join(keys + values)).hexdigest()

def run_model(params, dev_path):
    model = ft.supervised(**params)
    perf = model.test(dev_path)
    
    results = {
        "precision" : perf.precision, 
        "recall" : perf.recall
    }
    print results
    
    params.update({"results" : results})
    return params

def expand_grid(all_params):
    all_params = copy(all_params)
    # Make everything a list
    for k,v in all_params.items():
        if not isinstance(v, list):
            all_params[k] = [v]
    
    grid_points = list(itertools.product(*all_params.values()))
    print >> sys.stderr, "Fitting model on %d grid points" % len(grid_points)
    for grid_point in grid_points:
        yield dict(zip(all_params.keys(), grid_point))

# --
# Run

train_path = 'yelp-2013-train-ft.txt'
dev_path = 'yelp-2013-dev-ft.txt'

all_params = {
    "input_file" : train_path,
    "output" : ".ft-model",
    "lr" : [0.05, 0.1, 0.25, 0.5],
    "dim" : [1, 5, 10, 20],
    "epoch" : 5,
    # "min_count" : [1, 10, 50, 100],
    "word_ngrams" : [1, 2, 3, 4, 5],
    "loss" : "softmax",
    "dictionary" : ".dict.dict"
}

all_results = []
for grid_point in expand_grid(all_params):
    try:
        print grid_point
        all_results.append(run_model(grid_point, dev_path))
    except KeyabordError:
        raise
    except:
        print "error"
