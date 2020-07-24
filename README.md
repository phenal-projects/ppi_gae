# PPI GAE
An implementation of graph autoencoder for PPI graph embedding.
- run.py -- GAE training code
- finetuning.py -- finetuning Encoder with the specified class

## Input details
- `edges` -- weighted tab-separated edge list.
- `feats` -- initial nodes' features table (csv).
- `ids` -- a file with some protein ids and class labels. 
