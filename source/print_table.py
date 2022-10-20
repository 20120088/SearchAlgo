table = open('table.csv', 'r').readlines()
header = table[0][:-1].split(',')

print(' ', end = '')
for i in range(37 + 20 * len(header[2:])): print('-', end='')
print()

print('|  {:<12}|| {:<20}'.format(header[0], header[1]),end='')
for h in header[2:]:
    print('|  {:<17}'.format(h), end='')
print('|')

print(' ', end = '')
for i in range(37 + 20 * len(header[2:])): print('-', end='')
print('')

for row in table[1:]:
    row = row[:-1].split(',')
    print('|  {:<12}|| {:<20}'.format(row[0], row[1]), end='')
    for item in row[2:]:
        print('|  {:<17}'.format(item), end='')
    print('|')

print(' ', end = '')
for i in range(37 + 20 * len(header[2:])): print('-', end='')
print('')
