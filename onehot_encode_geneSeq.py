import argparse
from joblib import Parallel, delayed
import multiprocessing
from generate_hdf5 import ReadWrite
import glob
import sys

"""
python3 embed_geneSeq.py --genome_file_path your/saved/fasta/file/directory
"""


def embedding(seq):
    encoding = {'A': [0.475, 0.175, 0.175, 0.175],
                'C': [0.175, 0.475, 0.175, 0.175],
                'G': [0.175, 0.175, 0.475, 0.175],
                'T': [0.175, 0.175, 0.175, 0.475]}

    embed_len = len(seq)-1
    gene_vec = [[0] * 4] * embed_len

    for i in range(embed_len):
        gene_vec[i] = encoding[seq[i]]

    return gene_vec, embed_len


def main(args):
    num_cores = multiprocessing.cpu_count()
    read_write = ReadWrite()
    fasta_files = args.genome_file_path + "*.fa"

    for genome_filename in glob.glob(fasta_files):
        gene_seq = read_write.read_input_file(genome_filename)
        print("Encoding", genome_filename)
        args = Parallel(n_jobs=num_cores, verbose=1)(delayed(embedding)(gene_seq[gene]) for gene in range(len(gene_seq)))
        read_write.create_output_array(args, genome_filename)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome_file_path", help="Directory of FASTA file")
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
