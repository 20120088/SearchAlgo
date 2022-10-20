import os

input_list = []

pwd = os.path.dirname(os.getcwd())
input_folder = os.path.join(pwd, 'input')

for level in os.listdir(input_folder):
    level_folder = os.path.join(input_folder, level)
    if len(os.listdir(level_folder)) > 0:
        for maze_file in os.listdir(level_folder):
            input_list.append('{}.{}'.format(level, maze_file.split('.')[0]))

input_list = ['Algorithm', 'Heuristic'] + input_list

with open('table.csv', 'w') as f:
    f.write(','.join(input_list))
    f.write('\n')