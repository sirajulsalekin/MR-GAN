import tables as tb


class ReadWrite:

    def read_input_file(self, genome_filename):

        with open(genome_filename, 'r') as fa:
            gene_seq_ = fa.readlines()[1::2]

        for i in range(len(gene_seq_)):
            if 'N' in gene_seq_[i] or 'n' in gene_seq_[i]:
                gene_seq_[i] = gene_seq_[i].replace("N", "").replace("n", "").upper()
            else:
                gene_seq_[i] = gene_seq_[i].upper()

        return gene_seq_

    def create_output_array(self, args_, genome_filename):

        gene_vec_, embed_len = zip(*args_)
        total_len = sum(embed_len)
        h5f = tb.open_file(genome_filename[:-3] + ".h5", 'w')
        out = h5f.create_carray(h5f.root, 'data', tb.Float32Atom(), shape=(total_len, 100), filters=tb.Filters(complevel=0))

        start_pos = 0
        for i in range(len(embed_len)):
            end_pos = start_pos + embed_len[i]
            out[start_pos:end_pos] = gene_vec_[i]
            start_pos = end_pos
        h5f.close()

        return out
