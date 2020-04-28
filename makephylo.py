from Bio import Phylo

tree = Phylo.read("phylodata/simple.dnd", "newick")
Phylo.draw(tree)
