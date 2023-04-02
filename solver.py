import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
import random
global posWalls, posGoals

class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 


    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper():
        # if the move was a push and the 
        # new position of the player is free
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        # if the new position of that box is free
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    # uppercase if push
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue #if it works don't touch it
    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))



def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            # board is a 3x3 neighborhood of the box

            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                # flip and rotate that neighborhood in all possible direction
                # and check for failed patterns
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                # though this one seems redundant
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]] 


    max_frontier_size = 0
    
    while frontier:

        if len(frontier) > max_frontier_size:
            print("\b" * len(str(max_frontier_size)),end="")
            print(len(frontier),end="")
            max_frontier_size = len(frontier)

        node = frontier.pop()
        # node ~ [ ( playerpos, ( box0, box1, ...) ), ... ]
        node_action = actions.pop()

        # node[-1][-1] ~ ( box0, box1, ...)
        if isEndState(node[-1][-1]):
            return node_action[1:]

        # node[-1] ~ ( playerpos, ( box0, box1, ...) )
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue

                # node + [(newPosPlayer, newPosBox)] ~ [ ( playerpos, ( box0, box1, ...) ), (nplayerpos, ( nbox0, nbox1, ...) ) ]
                frontier.append(node + [(newPosPlayer, newPosBox)])

                actions.append(node_action + [action[-1]])

    print("cannot find the solution!")
    return []



def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    if isEndState(beginBox):
        return []

    startState = (beginPlayer, beginBox) # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    frontier = collections.deque([[startState]]) # store states
    actions = collections.deque([[0]]) # store actions
    exploredSet = set()


    ### Implement breadthFirstSearch here
    # since deque is a generalization of queue and stack 
    # which doesnt specify which side is the start of the queue
    # therefore, this queue (frontier) will use the following interpretation: front --> back
    while frontier:
        node = frontier.popleft() #pop front

        node_action = actions.popleft()

        # check if current state has been reached before
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])

            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)

                # check goal state before adding to queue 
                # since bfs allows early goal test
                if isEndState(newPosBox):
                    # goal state found!
                    return node_action[1:] + [action[-1]]
 
                if isFailed(newPosBox):
                    continue

                frontier.append(node + [(newPosPlayer, newPosBox)])#push to back

                actions.append(node_action + [action[-1]])
        
    
    print("cannot find the solution!")
    return []
    
def cost(actions):
    """A cost function"""
    # basically taxing all actions that doesn't affect
    # the boxes position which force the agent to 
    # choose the shortest path that lead to the box
    return len([x for x in actions if x.islower()])

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = PriorityQueue()
    frontier.push([startState], 0)
    exploredSet = set()

    
    ### Implement uniform cost search here

    while not frontier.isEmpty():
        node = frontier.pop()
        node_action = actions.pop()

        if isEndState(node[-1][-1]):
            # goal state found!
            return node_action[1:]
        
        # check if current state has been reached before
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])

            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)

                if isFailed(newPosBox):
                    continue

                new_cost = cost(node_action[1:])

                frontier.push(node + [(newPosPlayer, newPosBox)],new_cost)

                actions.push(node_action + [action[-1]],new_cost)
        
    
    print("cannot find the solution!")
    return []



# after optimization

def depthFirstSearch_optimized(gameState):
    """Implement depthFirstSearch approach"""
    # presorting the boxes position to reduce number 
    # of possible states as all boxes are the same
    
    beginBox = tuple(sorted(PosOfBoxes(gameState)))
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)

    # removing past states as it is not needed by the algorithm 
    # (i still dont know why it is here since past actions are already 
    # recorded in the actions variable)
    # remove actions variable as actions can be stored in the node itselft
    # actions will be stored as string instead of list 
    # and will be transform back to list later.

    start_node = (startState,"") # state, path

    frontier = collections.deque([start_node])

    exploredSet = set()

    
    while frontier:
        node = frontier.pop()
        # node ~ (( playerpos, ( box0, box1, ...) ), "actions" )
        

        # node[0][-1] ~ ( box0, box1, ...)
        if isEndState(node[0][-1]):
            # transform a string of actions into a list of actions
            return [ i for i in node[-1]]

        # check if current state has been reached before
        if node[0] not in exploredSet:
            exploredSet.add(node[0])
            for action in legalActions(node[0][0], node[0][1]):
                newPosPlayer, newPosBox = updateState(node[0][0], node[0][1], action)

                if isFailed(newPosBox):
                    continue

                frontier.append(((newPosPlayer,tuple(sorted(newPosBox))), node[-1] + action[-1]))

                
    print("cannot find the solution!")
    return []

def actions_shift(actions,number):
    # shuffle the actions list so that loop are harder to occurred
    # but since RNG is slow, just shifting the actions list is enough
    return [actions[(i + number)%len(actions)] for i in range(len(actions))]

def pseudoshuffle(actions,seed):
    source = [i for i in range(len(actions))]

    re = []

    for _ in range(len(actions)):

        re += [source.pop(seed%len(source))]
        seed = (seed * 7757)%8233
    
    return [actions[i] for i in list(re)]




def dfs_with_depth_limit (beginBox, beginPlayer, depth_limit):
    # dfs implementation as a tree search failed 

    startState = (beginPlayer, tuple(sorted(beginBox)))


    start_node = (startState, ' ' ,0) #state, action before, action index

    frontier = collections.deque([start_node])

    exploredSet = dict()



    actions = [i for i in ("*" +  " "*depth_limit)]

    seed =  (int(time.time()*43956673))%283

    while frontier:

        node = frontier.pop()

        # add node action to actions

        actions[node[2]] = node[1]

        if isEndState(node[0][-1]):
            # transform a string of actions into a list of actions
            return [ i for i in actions[1:node[2] + 1]]
        
        if node[2] >= depth_limit:
            continue

        l_actions = legalActions(node[0][0], node[0][1])

        seed = (seed + 7757)%8233

        for action in l_actions:
            newPosPlayer, newPosBox = updateState(node[0][0], node[0][1], action)

            newPosBox = tuple(sorted(newPosBox))

            if isFailed(newPosBox):
                    continue

            if (newPosPlayer, newPosBox) not in exploredSet or exploredSet[(newPosPlayer, newPosBox)] > node[2] + 1:
            
                exploredSet[(newPosPlayer, newPosBox)] = node[2] + 1
                frontier.append(((newPosPlayer, tuple(sorted(newPosBox))),action[-1], node[2] + 1))


    return []


def dfs_with_depth_limit_not_complete (beginBox, beginPlayer, depth_limit):
    # dfs implementation as a tree search failed 

    startState = (beginPlayer, tuple(sorted(beginBox)))


    start_node = (startState, ' ' ,0) #state, action before, action index

    frontier = collections.deque([start_node])

    exploredSet = set()

    can_continue = False



    actions = [i for i in ("*" +  " "*depth_limit)]

    seed =  (int(time.time()*43956673))%283

    while frontier:

        node = frontier.pop()

        # add node action to actions

        actions[node[2]] = node[1]

        if isEndState(node[0][-1]):
            # transform a string of actions into a list of actions
            return [ i for i in actions[1:node[2] + 1]], can_continue
        
        if node[2] >= depth_limit:
            can_continue = True
            continue

        # l_actions = legalActions(node[0][0], node[0][1])
        l_actions = pseudoshuffle(legalActions(node[0][0], node[0][1]),seed)


        seed = (seed + 7757)%8233

        for action in l_actions:
            newPosPlayer, newPosBox = updateState(node[0][0], node[0][1], action)

            newPosBox = tuple(sorted(newPosBox))

            if isFailed(newPosBox):
                    continue

            if (newPosPlayer, newPosBox) not in exploredSet:
            
                exploredSet.add((newPosPlayer, newPosBox))
                frontier.append(((newPosPlayer, newPosBox),action[-1], node[2] + 1))


    return [], can_continue

def iterative_deepening_search(gameState):

    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    depth = 8

    while True:

        print("depth: ", depth)

        # somehow the non complete version works better...

        # solution, can_continue = dfs_with_depth_limit(
        #     beginBox=beginBox,
        #     beginPlayer=beginPlayer,
        #     depth_limit=depth
        #     )

        solution, can_continue = dfs_with_depth_limit_not_complete(
            beginBox=beginBox,
            beginPlayer=beginPlayer,
            depth_limit=depth
            )
        


        
        if len(solution) >= 1:
            return solution
        
        if not can_continue:
            break
        # break
        
        depth *= 2
        # depth += 10
        
    

    print("cannot find the solution!")
    return []



def breadthFirstSearch_optimized(gameState):
    """Implement breadthFirstSearch approach"""
    beginBox = tuple(sorted(PosOfBoxes(gameState)))
    beginPlayer = PosOfPlayer(gameState)

    if isEndState(beginBox):
        return []


    startState = (beginPlayer, beginBox) # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))

    start_node = (startState,"") # e.g. ( ( (2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)) ), "UdLr")

    frontier = collections.deque([start_node]) 

    exploredSet = set()

    while frontier:
        node = frontier.popleft() #pop front

        # check if current state has been reached before
        if node[0] not in exploredSet:
            exploredSet.add(node[0])
            for action in legalActions(node[0][0], node[0][1]):
                newPosPlayer, newPosBox = updateState(node[0][0], node[0][1], action)

                # check goal state before adding to queue 
                # since bfs allows early goal test
                if isEndState(newPosBox):
                    # goal state found!
                    return [ i for i in node[-1]] + [action[-1]]
 
                if isFailed(newPosBox):
                    continue

                frontier.append(((newPosPlayer, tuple(sorted(newPosBox))), node[-1] + action[-1]))#push to back

    
    print("cannot find the solution!")
    return []



def uniformCostSearch_optimized(gameState):
    """Implement uniformCostSearch approach"""
    beginBox = tuple(sorted(PosOfBoxes(gameState)))
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox) # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    # also store cost in node as cost can be updated every action
    start_node = (startState,"",0) #state, path, cost e.g. ( ( (2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)) ), "UdLr", 2)

    frontier = PriorityQueue()
    frontier.push(start_node, 0)
    exploredSet = set()
    
    ### Implement uniform cost search here

    while not frontier.isEmpty():
        node = frontier.pop()
        
        if isEndState(node[0][1]):
            # goal state found!
            # transform a string of actions into a list of actions
            return [ i for i in node[1]]
        
        # check if current state has been reached before
        if node[0] not in exploredSet:
            
            # mark state as reached
            exploredSet.add(node[0])

            for action in legalActions(node[0][0], node[0][1]):

                newPosPlayer, newPosBox = updateState(node[0][0], node[0][1], action)
 
                if isFailed(newPosBox):
                    continue
                
                # if the current action doesnt change the boxes' position
                new_cost = node[2] +  action[-1].islower()

                frontier.push(((newPosPlayer, tuple(sorted(newPosBox))), node[1] + action[-1],new_cost),new_cost)
        
    
    print("cannot find the solution!")
    return []


"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method, level_number = 0):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    if method == 'dfs':
        # result = depthFirstSearch(gameState)
        result = depthFirstSearch_optimized(gameState)
    elif method == 'ids':
        result = iterative_deepening_search(gameState) 
    elif method == 'bfs':
        result = breadthFirstSearch_optimized(gameState) 
    elif method == 'ucs':
        result = uniformCostSearch_optimized(gameState)
    else:
        raise ValueError('Invalid method.')


    duration = (time.time() - time_start) * 1000

    print('Runtime of %s: %.2f ms.' %(method, duration))


    # save results to file
    with open("./results/" + method + ".csv","a") as file:
        file.writelines( str(level_number) +  ", "+ str(duration) + ", " + str(len(result)) + "\n")


    

    print(result)
    return result
