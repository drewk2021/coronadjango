from Bio.Blast.NCBIWWW import qblast

def coronablast(coronaseqrec):

    spikeblast = qblast("blastn","nt",coronaseqrec.seq)

    with open("results.xml", "w") as save_file:
        blast_results = spikeblast.read()
        save_file.write(blast_results)

    return
