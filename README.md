### hyperband
Toy hyperband optimization

See http://www.argmin.net/2016/06/23/hyperband/ for details

#### Getting Started

Need to wrap the function of interest in a class w/ two methods, as followers
    
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

# --
# Run

if __name__ == "__main__":
    from hyperband import HyperBand
    model = TestModel()
    HyperBand(model).run()


#### To Do

- Cache models