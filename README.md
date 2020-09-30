Chapter 18 of "Deep learning for the Earth Sciences: A comprehensive approach to remote sensing, climate science and geosciences".

The repository contains the code for the experiments conducted for this book chapter.

<<<<<<< HEAD
If you need asistance with the code, contace bkraft@bgc-jena.mpg.de
=======
Structure:

* `src` code
    * `data` data processing
    * `models` pytorch models
    * `visualizations` code to create visualizations
    * `experiments` the experiments (model trainers etc.)
        * `hydrology`
        * `vegetation`
* `data` the data
* `docs` text documents
    * `chapter` the book chapter

Run:

To run hyperparameter turing, model training and inference, use the following command from the `src` directory:

`ipython -- hp_tune.py -c [config name] -O && ipython -- model_tune.py -c [config name] -O && ipython -- inference.py -c [config name] -O`
replace `[config name]` with one of:

* n_sm.n_perm: do not use soil moisture as predictor / do not permute time-series
* w_sm.n_perm: use soil moisture as predictor / do not permute time-series
* w_sm.w_perm: use soil moisture as predictor / permute time-series
* n_sm.w_perm: do not use soil moisture as predictor / permute time-series

ipython -- hp_tune.py -c n_sm.n_perm -O && ipython -- model_tune.py -c n_sm.n_perm -O && ipython -- inference.py -c n_sm.n_perm -O
>>>>>>> dev_bk
