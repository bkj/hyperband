### hyperband
Toy hyperband optimization

See http://www.argmin.net/2016/06/23/hyperband/ for details

#### Getting Started

To use, you'll need to wrap the function of interest in a class w/ two methods:
    
```

class TestModel:

    def rand_config(self):
        """ 
            Takes:
                Nothing
            Returns:
                Random parameter configuration for the model
        """
        pass

    def eval_config(self, config, iters):
        """ 
            Takes:
                Random parameter configuration for the model
                Number of iterations to run the model
            Returns:
                Dictionary like:
                    {
                        "obj" : ... value of objective function (smaller = better)
                        "config" : config,
                        "iters" : iters
                    }
        """
        pass
```

The run the optimization like:

```
from hyperband import HyperBand
model = TestModel()
hb = HyperBand(model)
hb.run()
print(hb.history)
```

`hb.history` will contain records of all the experiments that were run.  By default, `hb.run()` dumps the results of experiments in JSON to `sys.stdout` as it runs.
