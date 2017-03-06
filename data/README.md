The training, validation, and test files for Penn Treebank (ptb.train.txt, ptb.valid.txt, and ptb.test.txt) were downloaded from
https://github.com/wojzaremba/lstm/tree/master/data

The enwik8 data reader was adapted from the Tensorflow data reader for the Penn Treebank dataset.

Raw data for the Hutter Prize dataset (enwik8) should be downloaded from http://www.mattmahoney.net/dc/textdata

## Instructions
----

To use the enwik8/text8 dataset with Tensorflow, Torch7 or Brainstorm, please run the create_enwik8.py/create_text8.py script to download and prepare the enwik8/text8 dataset in HDF5 format.
This will require the Python ```h5py``` package.

Note that the batch size and sequence length are used to preprocess the data into an appropriate format. To use a different batch size (currently 128) and sequence length (currently 50) for enwik8/text8 please change the corresponding numbers in the create_enwik8.py/create_text8.py file.

To train on enwik8/text8 with Tensorflow, please unzip the downloaded enwik8.zip/text8.zip file into this directory.
