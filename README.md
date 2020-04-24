# chess-imitation-learing

### Generating Data
* `python data_generator_multiprocess.py`

### Training a model
* State to action: `python train_cp.py`
* State to value(buckets): `python train_disc.py`
* State to value(L1): `python train_disc.py`
* simese: `python train_siam.py`

### uci wrapper for model
* `python uci_wrapper.py --model_type=reg --model=model.nn`
