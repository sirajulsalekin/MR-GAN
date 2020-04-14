# Genomics-Feature-Learning

The algorithms takes the gene sequences as input in fasta format (*.fa).

Requirement:

2. Python3
3. Tensorflow

To run code:

1. At first, run onehot_encode_geneSeq.py for one-hot encoding of transcriptomic sequences and save the embedded matrix as HDF5 file (*.h5)

python3 onehot_encode_geneSeq.py --genome_file_path your/saved/fasta/file/directory

*** Note that, this code generates ~30 GB data for whole gene sequence and save to your local disk.

2. Feed the saved HDF5 files into the GFL_main_wgan.py to train the MR-GAN model.

python3 GFL_main_wgan.py --genome_file_path your/saved/HDF5/file/directory --log_dir directory/where/trained/model/will/be/saved 
