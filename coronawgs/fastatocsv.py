import sys, os


if len(sys.argv) <= 1:
    print('\nPlease provide input FASTA file.\n')
    print('Usage:\nfastatocsv.py input.fasta output.csv\n')
    sys.exit()

if sys.argv[1] == '-h' or sys.argv[1] == 'help'or sys.argv[1] == '-help':
    print('\nUsage:\npython fastatocsv.py  input.fasta  output.csv\n')
    sys.exit()

input = sys.argv[1]
if not os.path.exists(input):
    print('\nError: File "%s" is not exist!\n' % input)
    sys.exit()

output = 'output.csv'
if len(sys.argv) > 2:
    output = sys.argv[2]

# Read in FASTA
file = open(input, 'r')
lines_i = file.readlines()
seq = {}
seqs = []


for l in lines_i:

    if l[0] == '>':
        'Fasta head line, getting type and location'
        seqs += [seq] # adding dictionary to broader list
        seq_info = l.split("|")
        seq_local = {}
        seq_local["seq_type"] = seq_info[1]
        seq_local["seq_location"] = seq_info[3]
        seq_local["seq"] = ""
        seq = seq_local
    else:
        'Sequence line'
        seq["seq"] += l.strip()



file.close()

print('The Input file is: %s' %input)


# Convert FASTA to CSV
seqs.pop(0) # removing first (empty) item in seqs list
l = []
lines = ["Type,Sequence,Location\n"]
for seqi in seqs:
    line = seqi["seq_type"]+","+seqi["seq"]+","+seqi["seq_location"]+"\n"
    lines += line



# Output CSV file
file = open(output, 'w')
file.writelines(lines)
file.close()
print('The Output file is: %s' %output)
