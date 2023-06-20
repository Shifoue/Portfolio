def min_pos(M, path):
    D = {}
    max_x = len(M[0])
    max_y = len(M)
    x,y = path[-1]

    for i in [-1,0,1]:

        for j in [-1,0,1]:
            if ((abs(j) != abs(i)) and (0 <= x+j < max_x) and (0 <= y+i < max_y)):
                if ((M[y+i][x+j] != "#") and (type(M[y+i][x+j]) == type(1)) and ((x+j, y+i) not in path)):
                    if (D.get(M[y+i][x+j], None)):
                        D[M[y+i][x+j]].append((x+j, y+i))
                    else:
                        D[M[y+i][x+j]] = [(x+j, y+i)]

    if not D:
        return None
    
    return D[min(D.keys())]


def pathfinding_djikstra(M, start, end):
    Heuristic_M = heuristic_queue_djikstra(M, start, end)
    print(Heuristic_M)
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
        
        if min_neighbors == None:
            continue
        
        for m in min_neighbors:
            queue.append(path + [m])

        if (queue[0] == None):
            queue.append(None)

    return -1

def heuristic_queue_djikstra(M, start, end):
    x_end, y_end = end
    max_x = len(M[0])
    max_y = len(M)

    queue = [(start, 0)]
    early_stop = False
    
    while queue:
        pos = queue.pop(0)
        
        if pos[0] == end:
            break

        x,y = pos[0]
        cost = pos[1]
        
        M[y][x] = cost + abs(y - y_end) + abs(x - x_end)
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                if ((0 <= x+j < max_x) and (0 <= y+i < max_y) and (abs(j) != abs(i))):
                    if ((M[y+i][x+j] != "#") and (type(M[y+i][x+j]) == type(""))):
                        weight = (cost + 1) + abs((y + i) - y_end) + abs((x + j) - x_end)
                        
                        if not queue:
                            queue.insert(0, ((x+j, y+i), cost+1))
                            continue

                        length = len(queue)
                        last_elem = True
                        for k in range(length):
                            x_k, y_k = queue[k][0]
                            temp_weight = (queue[k][1] + 1) + abs(y_k - y_end) + abs(x_k - x_end)
                            if weight < temp_weight and (((x+j, y+i)) not in [p[0] for p in queue if p != None]):
                                queue.insert(k, ((x+j, y+i), cost+1))
                                last_elem = False
                                break
                            
                        if last_elem:
                            queue.append(((x+j, y+i), cost+1))
                        
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

Mat = [['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
       ['.', '.', '.', '#', '#', '#', '#', '.', '.', '.', '.', '.', '.', '.'],
       ['.', '.', '.', '.', '.', '.', '#', '.', '.', '#', '#', '#', '.', '.'],
       ['#', '#', '#', '#', '.', '.', '#', '.', '.', '#', '#', '#', '.', '.'],
       ['.', '.', '.', '.', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.'],
       ['.', '.', '.', '#', '#', '#', '#', '.', '.', '#', '#', '#', '#', '#'],
       ['.', '.', '.', '#', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.'],
       ['.', '.', '.', '#', '.', '.', '#', '#', '#', '#', '#', '#', '.', '.'],
       ['.', '.', '.', '.', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.']]

p = pathfinding_djikstra(Mat, (0,0), (0,8))
pretty_print(Mat, p)
