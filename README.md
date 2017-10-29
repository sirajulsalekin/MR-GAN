# Genomics-Feature-Learning

Genomics feature learning is an unsupervised feature extraction algorithm. The algorithm is inspired by the 'Adversarilly Learned Inference' method and extracts features from human gene sequences.

The algorithms takes the gene sequences as input in fasta format (*.fa).

Requirement:

1. dna2vec (https://github.com/pnpnpn/dna2vec).
2. Python3
3. Tensorflow

To run code:

1. At first, run embed_geneSeq.py to embed gene sequences and save the embedded matrix as HDF5 file (*.h5)

python3 embed_geneSeq.py --genome_file_path your/saved/fasta/file/directory --dna2vec_file_path directory/of/.w2v(dna2vec)/file 

*** Note that, this code generates >250 GB data for whole gene sequence and save to your local disk.

2. Feed the saved HDF5 files into the GFL_main.py to train the GFL algorithm.

python3 GFL_main.py --genome_file_path your/saved/HDF5/file/directory --log_dir directory/where/trained/model/will/be/saved 
