### PATHFINDING

def min_pos(M, path):
    D = {}
    max_x = len(M[0])
    max_y = len(M)
    x,y = path[-1]
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            if ((0 <= x+j < max_x) and (0 <= y+i < max_y) and (abs(j) != abs(i))):
                if ((M[y+i][x+j] != "#") and (type(M[y+i][x+j]) == type(1)) and ((x+j, y+i) not in path)):
                    if (D.get(M[y+i][x+j], None)):
                        D[M[y+i][x+j]].append((x+j, y+i))
                    else:
                        D[M[y+i][x+j]] = [(x+j, y+i)]

    return D[min(D.keys())]


def pathfinding_djikstra(M, start, end):
    Heuristic_M = heuristic_queue_djikstra(M, start, end)
    x_start, y_start = start
    max_x = len(M[0])
    max_y = len(M)

    queue = [[end], None]
    
    while queue:
        path = queue.pop(0)
        if path == None:
            continue
        
        if (path[-1] == start):
            return path
          
        min_neighbors = min_pos(M, path)
        
        for m in min_neighbors:
            queue.append(path + [m])

        if (queue[0] == None):
            queue.append(None)
            

def heuristic_queue_djikstra(M, start, end):
    x_end, y_end = end
    max_x = len(M[0])
    max_y = len(M)

    queue = [start, None]
    c = 0
    early_stop = False
    
    while queue:
        pos = queue.pop(0)
        if pos == end:
            early_stop = True

        if pos == None:
            if early_stop:
                break
            c += 1
            continue
        
        x,y = pos   
        M[y][x] = c
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                if ((0 <= x+j < max_x) and (0 <= y+i < max_y) and (abs(j) != abs(i))):
                    if ((M[y+i][x+j] != "#") and (type(M[y+i][x+j]) == type(""))):
                        queue.append((x+j, y+i))
                        
        
        if (queue[0] == None):
            queue.append(None)

    return M


def pretty_print(M, path):
    for coord in path:
        M[coord[1]][coord[0]] = "X"

    for row in M:
        for x in row:
            if len(str(x)) == 1:
                print(str(x)+" ", end=" ")
            else:
                print(x, end=" ")
        print()
        

Mat = [['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
       ['.', '.', '.', '#', '#', '#', '#', '.', '.', '.', '.', '.', '.', '.'],
       ['.', '.', '.', '.', '.', '.', '#', '.', '.', '#', '#', '#', '.', '.'],
       ['#', '#', '#', '#', '.', '.', '#', '.', '.', '#', '#', '#', '.', '.'],
       ['.', '.', '.', '.', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.'],
       ['.', '.', '.', '#', '#', '#', '#', '.', '.', '#', '#', '#', '#', '#'],
       ['.', '.', '.', '#', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.'],
       ['.', '.', '.', '#', '.', '.', '#', '#', '#', '#', '#', '#', '.', '.'],
       ['.', '.', '.', '.', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.']]

p = pathfinding_djikstra(Mat, (0,0), (13,8))
pretty_print(Mat, p)
