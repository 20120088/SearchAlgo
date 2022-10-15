import copy

def read_map(file_name):
    with open(file_name, "r") as f:
        n = int(f.readline())

        plus_point = [] #read list of plus point
        for i in range(n):
            point = [int(x) for x in f.readline().split(' ')]
            point[2] = -point[2]
            plus_point.append(point)

        maze = [] #read maze
        for line in f:
            if line[-1] == '\n': line = line[:-1]
            maze.append([x for x in line])

        for point in plus_point: #embed plus point to maze
            maze[point[0]][point[1]] = point[2]

        for i,x in enumerate(maze[0]): 
            if x == ' ': maze[0][i] = 'G'
        for i,x in enumerate(maze[-1]): 
            if x == ' ': maze[-1][i] = 'G'
        for line in maze[1:-1]:
            if line[0] == ' ': line[0] = 'G'
            if line[-1] == ' ': line[-1] = 'G'
        
        return maze
            
def encode_char(char):
    if char == 'x': return 0
    if char == 'S': return 255
    if char == ' ': return 170
    if char == 'P': return 235
    if char == 'V': return 200
    if char == 'O': return 220
    if char == 'G': return 240
    return 120 - int(char) * 9