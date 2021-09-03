# efx charity algorithm

This implements the u0 and (regular) u2 algorithm in the charity paper.

## potential issues and future

I've removed all randomness from the algorithm (I think..) This is to help debug issues in the algorithm.

There aren't any unit tests, but based on my ad hoc testing, I think it's working ok. We should write some.

There's not a ton of logging into debugging the actual rules. Hopefully the `assignments` variable has what you need when the algorithm gets stuck.

I hope it's bug-free, but if you find any issues please let me know.

Based on random inputs, it definitely gets stuck. And it could be my algorithm's fault -- in places where random choices were made, we could enumerate all options and try from there.

Enjoy!

## using it

1. Python

You need to install `numpy` and `networkx` packages.

Then get the code:
```
curl https://raw.githubusercontent.com/marwahaha/EFX/main/efx_charity.py > efx_charity.py
```

Then run it.
```
python efx_charity.py run
```


2. Jupyter notebook

You need to install Jupyter notebook (it's pretty straightforward)

Then clone this repository.

Then use `efx_charity.ipynb` notebook.

There are examples of randomly generated inputs and saving the input data as a python object.