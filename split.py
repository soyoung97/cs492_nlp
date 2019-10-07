import pandas as pd


f= open("termDocMatrix.txt", 'r')
whole_file = f.read()
lines = whole_file.split('\n')
def make_float(val):
    if val == '':
        return
    return float(val)
string_values = [[make_float(val) for val in line.split('  ')] for line in lines][:-1]

#print(string_values) # should print out 4423 * 500 float matrix
