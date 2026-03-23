from Bio import SeqIO
from collections import Counter
import sys

with open('proteins.fasta', 'r') as handle:
    for record in SeqIO.parse(handle, 'fasta'):
        print(f"UniProt ID: {record.id}")
        print(f"Molecular weight: {len(record.seq)}")
        print(f"Amino acid composition: {Counter(record.seq)}")
        print("\n")
