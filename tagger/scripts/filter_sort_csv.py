import sys
import csv
import time
from csvsort import csvsort

csv.field_size_limit(sys.maxint)

def filter_starting_with(csvinput, outputname, startchars):
    with open(csvinput) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        f = open(outputname, 'w')
        for row in reader:
            if row[2].startswith(startchars):
                print row[2]
                f.write('\t'.join(row) + '\n')
        f.close()

def sort_image_data(csvinput):
    csvsort(csvinput, [2], has_header=False, delimiter='\t')

# filter_starting_with("filename", "0")
# sort_images('filename')