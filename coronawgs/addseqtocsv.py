from Bio import SeqIO

coronaseq_dict = SeqIO.to_dict(SeqIO.parse("sequences.fasta", "fasta")) # sequenced complete genomes


file = open("sequences.csv", "r")
lines = file.readlines()
new_lines = [lines.pop(0)]


for i in range(len(coronaseq_dict.keys())):
    new_lines += [lines[i].strip("\n")+","+str(coronaseq_dict[list(coronaseq_dict.keys())[i]].seq)+"\n"]

file.close()
file = open("sequences.csv","w")
file.writelines(new_lines)
file.close()
