import argparse
from joblib import Parallel, delayed
import multiprocessing
from generate_hdf5 import ReadWrite
import time
import glob
from dna2vec.multi_k_model import MultiKModel

parser = argparse.ArgumentParser()
parser.add_argument("--genome_file_path", help="Directory of FASTA file")
parser.add_argument("--dna2vec_file_path", help="Directory where .w2v file is saved (from dna2vec)")
args = parser.parse_args()

num_cores = multiprocessing.cpu_count()
read_write = ReadWrite()
mk_model = MultiKModel(args.dna2vec_file_path)
fasta_files = args.genome_file_path + "*.fa"


def embedding(seq):

    # seq = gene_seq[gene]
    embed_len = (len(seq)-1)//3
    gene_vec = [[0] * 100] * embed_len

    for i in range(embed_len):
        gene_vec[i] = mk_model.vector(seq[i*3:i*3+3])

    return (gene_vec, embed_len)


for genome_filename in glob.glob(fasta_files):
    gene_seq = read_write.read_input_file(genome_filename)
    start_time = time.time()
    args = Parallel(n_jobs=num_cores, verbose=1)(delayed(embedding)(gene_seq[gene]) for gene in range(len(gene_seq)))
    print(time.time()-start_time)
    read_write.create_output_array(args, genome_filename)

