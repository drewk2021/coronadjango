from Bio import SeqIO

coronaseq_dict = SeqIO.to_dict(SeqIO.parse("coronawgs/sequences.fasta", "fasta"))
coronaref_record = SeqIO.read("coronawgs/refseq.fasta","fasta")
