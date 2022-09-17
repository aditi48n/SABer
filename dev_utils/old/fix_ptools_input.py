import sys

file = sys.argv[1]
new_file = file + '.bak'
print(file)
print(new_file)
with open(file, 'r') as infile:
    lines = [line for line in infile]
new_lines = []
for line in lines:
    if (('NCBI-TAXON-ID	155864' in line) | ('NCBI-TAXON-ID	12908' in line)):
        line = 'NCBI-TAXON-ID	199310\n'
    new_lines.append(line)

with open(new_file, 'w') as outbak:
    outbak.write(''.join(lines))

with open(file, 'w') as outfile:
    outfile.write(''.join(new_lines))
