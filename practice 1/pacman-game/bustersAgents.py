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

# default imports
from builtins import range
from builtins import object
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters

# our imports
from random import randint
from wekaI import Weka

class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

    def printLineData(self, gameState):
        instance = [str(gameState.getPacmanPosition()[0]),
            str(gameState.getPacmanPosition()[0]),
            "West"]
        instance = ','.join(instance)
        print (instance)
        return instance

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
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
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

class BasicAgentAA(BustersAgent):

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

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print( gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())
        
        
    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        #self.printInfo(gameState)
        #print (self.printLineData(gameState))
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move


################################################################################
#                                  OUR AGENTS                                  #
################################################################################
# tiny class addendum for aesthetic purposes
class queue(list):
    def empty(self):
        if len(self) == 0:
            return True
        else:
            return False

    def push(self, __object) -> None:
        return super().append(__object)

'''Agent Made in Tutorial1'''
class Tutorial1_yay(BustersAgent):

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

    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1

        # walls + pacman
        w = gameState.getWalls()
        p = gameState.getPacmanPosition()

        # closest ghost
        min_dist = sys.maxsize
        for i in [x for x in gameState.data.ghostDistances if type(x)==int]:
            if i < min_dist:   min_dist = i
        g = gameState.getGhostPositions()[gameState.data.ghostDistances.index(min_dist)]

        # find best direction
        move = self.bfs(gameState, w, p, g)

        return move

    def printLineData(self, gameState):
        return globalPrintLineData(gameState)

    def bfs(self, state, walls, pacman, target):
        # setup
        pacman = tuple(pacman)
        target = tuple(target)

        visited = set()
        q = queue()
        q.push(pacman)
        parent = dict()
        parent[pacman] = None

        # scan graph via bfs
        path_found = False
        while not q.empty():
            curr_xy = q.pop(0)
            visited.add(curr_xy)
            if curr_xy == target:
                path_found = True
                break

            list_to_try = self.getMovesHypoth(walls, curr_xy)
            random.shuffle(list_to_try)
            for next_xy in list_to_try:
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



'''Agent Connected to Weka'''
class WekaAgent(BustersAgent):
    # child class __init__ override to include starting Weka.
    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

        self.weka = Weka()
        self.weka.start_jvm()

    def getAction(self, gameState):
        lineData = globalPrintLineData(gameState)
        print(lineData)
        #lineData = ','.join([str(x) for x in globalPrintLineData(gameState)])

        curr_model = "./newest_j48.model"
        curr_data = "./new_test.arff"
        move = self.weka.predict(curr_model, lineData, curr_data)
        #move = Directions.STOP

        # get legal actions
        legal = gameState.getLegalActions(0)

        # check legality -> correct if not legal.
        if move not in legal:
            print(f"FYI: using random bc weka move not legal\tweka: {move}",end="")
            move = legal[randint(0,len(legal)-1)]
            print(f"random: {move}")
        else:
            #move = legal[randint(0,len(legal)-1)]
            print(f"Using legal Weka prediction!\tweka: {move}")

        return move

    def printLineData(self, gameState):
        return globalPrintLineData(gameState)

################################################################################
#      MAKE GLOBAL printLineData() => CONSISTENCY & LESS COPY-PASTED CODE      #
################################################################################

future_score = None
current_score = 0

def globalPrintLineData(gameState, *, useOld=False):
    # game score + archive prev score
    global future_score
    global current_score
    if (future_score != None):
        current_score = future_score
        future_score = gameState.getScore()
    else:
        future_score = gameState.getScore()
        current_score = 0

    # pacman position
    pacman_x = gameState.getPacmanPosition()[0]
    pacman_y = gameState.getPacmanPosition()[1]
    pacman_pos = (pacman_x, pacman_y)

    # For each ghost show the manhattan distance + optimal direction to go.
    # Or return 0 & "NULL" if ghost is dead. ghostDistances list used to check state
    ghost_dists_test = gameState.data.ghostDistances  # is it dead?
    ghost_data = []
    num_ghosts = gameState.getNumAgents()-1  # agent 0 is pacman
    if num_ghosts >= 4:
        num_ghosts = 4
    if num_ghosts > 0:
        for g in range(0, num_ghosts):
            ghostX_data = [None] * 2

            if type(ghost_dists_test[g]) == int:
                ghostX_data[0] = ghost_dists_test[g]
                ghostX_posi = (gameState.getGhostPositions()[g][0], gameState.getGhostPositions()[g][1])
                ghostX_data[1] = getBestDirection(pacman_pos, ghostX_posi, g)
            else:
                ghostX_data[0] = 500
                ghostX_data[1] = 'NULL'

            ghost_data.append(ghostX_data[0])
            ghost_data.append(ghostX_data[1])

    # keep consistent with data in csv file (4 ghosts always there)
    if num_ghosts < 4:
        for x in range(4-num_ghosts):
            ghost_data.append(500)
            ghost_data.append('NULL')


    ## readability -> comebine ghost dist & optimal direction to compound lists
    #ghost0_data = [ghost0_dist, ghost0_dire]
    #ghost1_data = [ghost1_dist, ghost1_dire]
    #ghost2_data = [ghost2_dist, ghost2_dire]
    #ghost3_data = [ghost3_dist, ghost3_dire]
    #ghost_data = ghost0_data + ghost1_data + ghost2_data + ghost3_data


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
def getBestDirection(pacman_pos: tuple, ghost_posi: tuple, mod: int) -> str:
    x_diff = ghost_posi[0] - pacman_pos[0]
    y_diff = ghost_posi[1] - pacman_pos[1]

    if mod%2 == 0:
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
    elif mod%2 == 1:
        if (abs(x_diff) > abs(y_diff)):
            if x_diff <= 0:
                return "West"
            else:
                return "East"
        else:
            if y_diff <= 0:
                return "South"
            else:
                return "North"

    # should never occur..
    return 'NULL'
