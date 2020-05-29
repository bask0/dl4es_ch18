#!/bin/bash

# ipython -- hp_tune.py -c n_sm.n_perm -O && ipython -- model_tune.py -c n_sm.n_perm -O && ipython -- inference.py -c n_sm.n_perm -O

# ipython -- hp_tune.py -c w_sm.n_perm -O && ipython -- model_tune.py -c w_sm.n_perm -O && ipython -- inference.py -c w_sm.n_perm -O

ipython -- hp_tune.py -c n_sm.w_perm -O && ipython -- model_tune.py -c n_sm.w_perm -O && ipython -- inference.py -c n_sm.w_perm -O

ipython -- hp_tune.py -c w_sm.w_perm -O && ipython -- model_tune.py -c w_sm.w_perm -O && ipython -- inference.py -c w_sm.w_perm -O


