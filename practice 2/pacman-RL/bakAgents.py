from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
import util
from game import Agent, Directions, Actions
from keyboardAgents import KeyboardAgent
import inference
import busters
import numpy as np
from bustersAgents import BustersAgent
from distanceCalculator import Distancer
from random import randint

# random addendum
class queue(list):
    def empty(self):
        if len(self) == 0:
            return True
        else:
            return False

    def push(self, __object) -> None:
        return super().append(__object)

################################################################################
#                                  OUR AGENTS                                  #
################################################################################
'''Agent Made in Tutorial1'''
class Tutorial1_Improved(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        #self.printInfo(gameState)
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman

        # find closest ghost (with index) using manhattan data
        man_dists = gameState.data.ghostDistances
        # find min from list (cant use min() if one ghost is already dead and it
        # becomes None-Type instead of a tuple).  also keep track of distances.
        closest_ghost = 100

        for i in range(len(man_dists)):
            if ((type(man_dists[i]) == int) and (man_dists[i] < closest_ghost)):
                closest_ghost = man_dists[i]
                idx_gho = man_dists.index(closest_ghost)
        if closest_ghost == 100:
            print("Error finding closest ghost -> no living ghosts")
            move = Directions.STOP
            return move

        # get locations of pacman + closest ghost
        loc_pac = list(gameState.getPacmanPosition())
        loc_gho = list(gameState.getGhostPositions()[idx_gho])

        # find closest dimension
        x_diff = loc_gho[0] - loc_pac[0]
        y_diff = loc_gho[1] - loc_pac[1]

        # MOVE OPPOSITE OF WHATS CLOSER
        # for moving E-W; y_diff is smaller
        if (abs(x_diff) > abs(y_diff)):
            if (x_diff <  0) and Directions.WEST in legal:  move = Directions.WEST
            if (x_diff >  0) and Directions.EAST in legal:  move = Directions.EAST
            # if x_diff == 0 there is no match and move to else-case...
        # for moving N-S; x_diff is smaller or they're equal
        elif (abs(x_diff) <= abs(y_diff)):
            if (y_diff <  0) and Directions.SOUTH in legal: move = Directions.SOUTH
            if (y_diff >  0) and Directions.NORTH in legal: move = Directions.NORTH
            if (x_diff == 0) and (y_diff == 0):             move = Directions.STOP
        else:
            move = Directions.STOP

        while move == Directions.STOP:
            move = legal[randint(0,len(legal)-1)]            

        # limitation -> it dsnt have a backup plan if theres a wall in the way lmao

        """ORIGINAL CODE
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        """

        walls = gameState.getWalls()

        move = self.bfs(gameState, walls, [loc_pac[0],loc_pac[1]], [loc_gho[0],loc_gho[1]])
        #print (f"{move} +++ {legal}")

        return move

    def bfs(self, state, walls, pacman, ghost):
        # setup
        pacman = tuple(pacman)
        ghost = tuple(ghost)

        visited = set()
        q = queue()
        q.push(pacman)
        parent = dict()
        parent[pacman] = None

        # scan graph in bfs
        path_found = False
        while not q.empty():
            curr_xy = q.pop(0)
            visited.add(curr_xy)

            if curr_xy == ghost:
                path_found = True
                break

            for next_xy in self.getMovesHypoth(walls, curr_xy):
                if next_xy not in visited:
                    q.append(next_xy)
                    parent[next_xy] = curr_xy
                    visited.add(next_xy)

        # reconstruct path
        path = queue()
        if path_found:
            path.push(curr_xy)
            while parent[curr_xy] is not pacman:
                path.push(parent[curr_xy])
                curr_xy = parent[curr_xy]

            path.reverse()
            #print(f"pacman: ({pacman[0]},{pacman[1]}) & next move ({path[0][0]},{path[0][1]})")

        # figure correct direction from next move.
        # *reversed expectations bc matching next move to pacman (prev move)*
        nexT = path[0]
        if (nexT[0],nexT[1]-1) == pacman:   return Directions.NORTH
        if (nexT[0],nexT[1]+1) == pacman:   return Directions.SOUTH
        if (nexT[0]+1,nexT[1]) == pacman:   return Directions.WEST
        if (nexT[0]-1,nexT[1]) == pacman:   return Directions.EAST

    def getMovesHypoth(self, walls, pman):
        x = pman[0]
        y = pman[1]
        legal = []

        # if north wall is true...
        if not walls[x][y+1]:   legal.append((x,y+1))  # append north location
        if not walls[x][y-1]:   legal.append((x,y-1))
        if not walls[x-1][y]:   legal.append((x-1,y))
        if not walls[x+1][y]:   legal.append((x+1,y))

        return legal
    def printLineData(self, gameState):
        return globalPrintLineData(gameState)

    def manhattan(self, p1, p2):
        return sum(abs(v1-v2) for v1, v2 in zip(p1,p2))



















################################################################################
#      MAKE GLOBAL printLineData() => CONSISTENCY & LESS COPY-PASTED CODE      #
################################################################################

future_score = ""
current_score = "0"

def globalPrintLineData(gameState, *, useOld=False):
    # game score + archive prev score
    global future_score
    global current_score
    if (future_score != ""):
        current_score = future_score
        future_score = f"{gameState.getScore()}"
    else:
        future_score = f"{gameState.getScore()}"
        current_score = f"{0}"

    # pacman position
    pacman_x = gameState.getPacmanPosition()[0]
    pacman_y = gameState.getPacmanPosition()[1]
    pacman_pos = (pacman_x, pacman_y)

    # For each ghost show the manhattan distance + optimal direction to go.
    # Or return 0 & "NULL" if ghost is dead. ghostDistances list used to check state
    ghost_dists_test = gameState.data.ghostDistances  # is it dead?
    # ghost 0
    if (type(ghost_dists_test[0]) == int):
        ghost0_dist = ghost_dists_test[0]
        ghost0_posi = (gameState.getGhostPositions()[0][0], gameState.getGhostPositions()[0][1])
        ghost0_dire = getBestDirection(pacman_pos, ghost0_posi)
    else:
        ghost0_dist = -1
        ghost0_dire = 'NULL'

    # ghost 1
    if (type(ghost_dists_test[1]) == int):
        ghost1_dist = ghost_dists_test[1]
        ghost1_posi = (gameState.getGhostPositions()[1][0], gameState.getGhostPositions()[1][1])
        ghost1_dire = getBestDirection(pacman_pos, ghost1_posi)
    else:
        ghost1_dist = -1
        ghost1_dire = 'NULL'

    # ghost 2
    if (type(ghost_dists_test[2]) == int):
        ghost2_dist = ghost_dists_test[2]
        ghost2_posi = (gameState.getGhostPositions()[2][0], gameState.getGhostPositions()[2][1])
        ghost2_dire = getBestDirection(pacman_pos, ghost2_posi)
    else:
        ghost2_dist = -1
        ghost2_dire = 'NULL'

    # ghost 3
    if (type(ghost_dists_test[3]) == int):
        ghost3_dist = ghost_dists_test[3]
        ghost3_posi = (gameState.getGhostPositions()[3][0], gameState.getGhostPositions()[3][1])
        ghost3_dire = getBestDirection(pacman_pos, ghost3_posi)
    else:
        ghost3_dist = -1
        ghost3_dire = 'NULL'

    # readability -> comebine ghost dist & optimal direction to compound lists
    ghost0_data = [ghost0_dist, ghost0_dire]
    ghost1_data = [ghost1_dist, ghost1_dire]
    ghost2_data = [ghost2_dist, ghost2_dire]
    ghost3_data = [ghost3_dist, ghost3_dire]
    ghost_data = ghost0_data + ghost1_data + ghost2_data + ghost3_data

    # wall test / legal moves
    ORDER = ['North', 'South', 'East', 'West', 'Stop']
    legal_moves = []  # binary boolean
    for x in gameState.getLegalActions():
        if x in ORDER:
            legal_moves.append(1)
        else:
            legal_moves.append(0)

    # get food & capsule stats
    food = gameState.getNumFood()
    capsules = len(gameState.getCapsules())

    # prev action
    prev_action = gameState.getPacmanState().getDirection()

    # compile shortened statistic variables to one string to be appended to csv
    # NOTE: LAST listed attribute is the previous action.
    state = [future_score,current_score,food,capsules] + ghost_data + [prev_action]

    """
    @relation training_tutorial1+

    @attribute future_score numeric
    @attribute current_score numeric
    @attribute food numeric
    @attribute capsules numeric
    @attribute ghost0_dist numeric
    @attribute ghost0_dire {NULL,West,East,North,South}
    @attribute ghost1_dist numeric
    @attribute ghost1_dire {NULL,West,East,North,South}
    @attribute ghost2_dist numeric
    @attribute ghost2_dire {NULL,West,East,North,South}
    @attribute ghost3_dist numeric
    @attribute ghost3_dire {NULL,West,East,North,South}
    @attribute prev_action {Stop,West,East,North,South}

    @data
    """

    return state


# Determines the simplistic optimal cardinal direction to head in to reach a ghost
def getBestDirection(pacman_pos: tuple, ghost_posi: tuple) -> str:
    x_diff = ghost_posi[0] - pacman_pos[0]
    y_diff = ghost_posi[1] - pacman_pos[1]

    if (abs(x_diff) > abs(y_diff)):
        if x_diff >= 0:
            return "East"
        else:
            return "West"
    else:
        if y_diff >= 0:
            return "North"
        else:
            return "South"
