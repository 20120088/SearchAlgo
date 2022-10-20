table = open('table.csv', 'r').readlines()
header = table[0][:-1].split(',')

def print_line():
    print(' ', end = '')
    for i in range(37 + 23 * len(header[2:])): print('-', end='')
    print()

def print_row(row):
    print('|  {:<12}|| {:<20}'.format(row[0], row[1]),end='')
    for h in row[2:]:
        print('|  {:<20}'.format(h), end='')
    print('|')

print_line()
print_row(header)
print_line()
for row in table[1:]:
    row = row[:-1].split(',')
    print_row(row)
print_line()