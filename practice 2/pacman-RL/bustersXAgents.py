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
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
import numpy as np
import os.path

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
        self.printInfo(gameState)
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move

    def printLineData(self, gameState):
        return "XXXXXXXXXX"


class QLearningAgentz(BustersAgent):

    #Initialization
    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.epsilon = 0.1
        self.alpha = 0.9
        self.discount = 0.8
        self.cond=False
        self.xOpt=None
        self.yOpt=None
        self.actions = {"North":0, "East":1, "South":2, "West":3}
        if os.path.exists("qtable.txt"):
            self.table_file = open("qtable.txt", "r+")
            self.q_table = self.readQtable()
        else:
            self.table_file = open("qtable.txt", "w+")
            #"*** CHECK: NUMBER OF ROWS IN QTABLE DEPENDS ON THE NUMBER OF STATES ***"
            self.initializeQtable(44)

    def initializeQtable(self, nrows):
        "Initialize qtable"
        self.q_table = np.zeros((nrows,len(self.actions)))

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)
           
        return q_table


    def writeQtable(self):
        "Write qtable to disc"        
        self.table_file.seek(0)
        self.table_file.truncate()

        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item)+" ")
            self.table_file.write("\n")

    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")   
            

    def __del__(self):
        "Destructor. Invokation at the end of each episode"        
        self.writeQtable()
        self.table_file.close()
        
    def optimalObjective(self, state, xOpt, yOpt): #this function distinguish whether we must followa PacMan or a joint hole in the walls to achieve it
        xPacman = state.getPacmanPosition()[0]
        yPacman = state.getPacmanPosition()[1]
        
        walls = state.getWalls()
        
        if walls[xPacman][yPacman + 1] == True and xPacman == xOpt and yPacman < yOpt: # wall North
            xOpt, yOpt= self.searchFalseN(state)
            self.cond=True
            self.xOpt=xOpt
            self.yOpt=yOpt
        elif walls[xPacman][yPacman - 1] == True and xPacman == xOpt and yPacman > yOpt: # wall South
            xOpt, yOpt= self.searchFalseS(state)
            self.cond=True
            self.xOpt=xOpt
            self.yOpt=yOpt

        elif walls[xPacman + 1][yPacman] == True and xPacman < xOpt and yPacman == yOpt: # wall East
            xOpt, yOpt= self.searchFalseE(state)
            self.cond=True
            self.xOpt=xOpt
            self.yOpt=yOpt
          
        elif walls[xPacman - 1][yPacman] == True and xPacman > xOpt and yPacman == yOpt: # wall West
            xOpt, yOpt= self.searchFalseW(state)
            self.cond=True
            self.xOpt=xOpt
            self.yOpt=yOpt
            
        else:
            self.cond=False
            self.xOpt=xOpt
            self.yOpt=yOpt
        
    def searchFalseN(self, state):
        walls=state.getWalls()
        xPacman = state.getPacmanPosition()[0]
        yPacman = state.getPacmanPosition()[1]
        for i in range(0,state.data.layout.width):
            if walls[i][yPacman+1]==False:
                xOptimal = i
                yOptimal = yPacman + 1
                return(xOptimal, yOptimal)
        
        
    def searchFalseS(self, state):
        walls=state.getWalls()
        xPacman = state.getPacmanPosition()[0]
        yPacman = state.getPacmanPosition()[1]
        for i in range(0,state.data.layout.width):
            if walls[i][yPacman-1]==False:
                xOptimal = i
                yOptimal = yPacman - 1
                return(xOptimal, yOptimal)
                
    def searchFalseE(self, state):
        walls=state.getWalls()
        xPacman = state.getPacmanPosition()[0]
        yPacman = state.getPacmanPosition()[1]
        for i in range(0,state.data.layout.height):
            if walls[xPacman+1][i]==False:
                xOptimal = xPacman + 1
                yOptimal = i
                return(xOptimal, yOptimal)
                
    def searchFalseW(self, state):
        walls=state.getWalls()
        xPacman = state.getPacmanPosition()[0]
        yPacman = state.getPacmanPosition()[1]
        for i in range(0,state.data.layout.height):
            if walls[xPacman-1][i]==False:
                xOptimal = xPacman - 1
                yOptimal = i
                return(xOptimal, yOptimal)


    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        """
        
        "*** YOUR CODE HERE ***"    
        #Find closest ghost and save its coordinates

        while None in state.data.ghostDistances:
            state.data.ghostDistances[state.data.ghostDistances.index(None)]=90000   
        min_Gdistance = min(state.data.ghostDistances)
        ghostIndex = state.data.ghostDistances.index(min_Gdistance)
        
        walls=state.getWalls()
        
        xGhost = state.getGhostPositions()[ghostIndex][0]
        yGhost = state.getGhostPositions()[ghostIndex][1]
        xPacman = state.getPacmanPosition()[0]
        yPacman = state.getPacmanPosition()[1]
        
                
        if self.cond==False or (xPacman==self.xOpt and yPacman==self.yOpt):
            xOpt=xGhost
            yOpt=yGhost
            self.optimalObjective(state,xOpt, yOpt)
            
        xOpt=self.xOpt
        yOpt=self.yOpt        


        if walls[xPacman][yPacman + 1] == True: # wall North
            if xPacman < xOpt and yPacman < yOpt:
                return 0
            if xPacman < xOpt and yPacman == yOpt:
                return 1
            if xPacman < xOpt and yPacman > yOpt:
                return 2
            if xPacman == xOpt and yPacman < yOpt:
                return 3
            if xPacman == xOpt and yPacman == yOpt:
                return 4
            if xPacman == xOpt and yPacman > yOpt:
                return 5
            if xPacman > xOpt and yPacman < yOpt:
                return 6
            if xPacman > xOpt and yPacman == yOpt:
                return 7
            if xPacman > xOpt and yPacman > yOpt:
                return 8
            
        if walls[xPacman][yPacman - 1] == True: # wall South
            if xPacman < xOpt and yPacman < yOpt:
                return 9
            if xPacman < xOpt and yPacman == yOpt:
                return 10
            if xPacman < xOpt and yPacman > yOpt:
                return 11
            if xPacman == xOpt and yPacman < yOpt:
                return 12
            if xPacman == xOpt and yPacman == yOpt:
                return 13
            if xPacman == xOpt and yPacman > yOpt:
                return 14
            if xPacman > xOpt and yPacman < yOpt:
                return 15
            if xPacman > xOpt and yPacman == yOpt:
                return 16
            if xPacman > xOpt and yPacman > yOpt:
                return 17

        if walls[xPacman + 1][yPacman] == True: # wall East
            if xPacman < xOpt and yPacman < yOpt:
                return 18
            if xPacman < xOpt and yPacman == yOpt:
                return 19
            if xPacman < xOpt and yPacman > yOpt:
                return 20
            if xPacman == xOpt and yPacman < yOpt:
                return 21
            if xPacman == xOpt and yPacman == yOpt:
                return 22
            if xPacman == xOpt and yPacman > yOpt:
                return 23
            if xPacman > xOpt and yPacman < yOpt:
                return 24
            if xPacman > xOpt and yPacman == yOpt:
                return 25
            if xPacman > xOpt and yPacman > yOpt:
                return 26

        if walls[xPacman - 1][yPacman] == True: # wall West
            if xPacman < xOpt and yPacman < yOpt:
                return 27
            if xPacman < xOpt and yPacman == yOpt:
                return 28
            if xPacman < xOpt and yPacman > yOpt:
                return 29
            if xPacman == xOpt and yPacman < yOpt:
                return 30
            if xPacman == xOpt and yPacman == yOpt:
                return 31
            if xPacman == xOpt and yPacman > yOpt:
                return 32
            if xPacman > xOpt and yPacman < yOpt:
                return 33
            if xPacman > xOpt and yPacman == yOpt:
                return 34
            if xPacman > xOpt and yPacman > yOpt:
                return 35
            
        else: #no walls
            if xPacman < xOpt and yPacman < yOpt:
                return 36
            if xPacman < xOpt and yPacman == yOpt:
                return 37
            if xPacman < xOpt and yPacman > yOpt:
                return 38
            if xPacman == xOpt and yPacman < yOpt:
                return 39
            if xPacman == xOpt and yPacman == yOpt:
                return 40
            if xPacman == xOpt and yPacman > yOpt:
                return 41
            if xPacman > xOpt and yPacman < yOpt:
                return 42
            if xPacman > xOpt and yPacman == yOpt:
                return 43
            if xPacman > xOpt and yPacman > yOpt:
                return 44


    def getQValue(self, state, action):

        """
            Returns Q(state,action)
            Should return 0.0 if we have never seen a state
            or the Q node value otherwise
        """
        position = self.computePosition(state)
        action_column = self.actions[action]       
        return self.q_table[position][action_column]


    def computeValueFromQValues(self, state):
        """
            Returns max_action Q(state,action)
            where the max is over legal actions.  Note that if
            there are no legal actions, which is the case at the
            terminal state, you should return a value of 0.0.
        """
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions)==0:
            return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, state):
        """
            Compute the best action to take in a state.  Note that if there
            are no legal actions, which is the case at the terminal state,
            you should return None.
        """
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions)==0:
            return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """

        # Pick Action
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        action = None

        if len(legalActions) == 0:
                return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            return random.choice(legalActions)
        return self.getPolicy(state)


    def update(self, state, action, nextState, reward):
        """
            The parent class calls this to observe a
            state = action => nextState and reward transition.
            You should do your Q-Value update here

        Q-Learning update:
        if terminal_state:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        else:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))
        """
        "*** YOUR CODE HERE ***"        
        s = self.computePosition(state)
        action_column = self.actions[action]
        Qvalue=self.getQValue(state, action)
        

        if reward!=0: 
            Qvalue=(1-self.alpha)*Qvalue + self.alpha*(reward)
        else:
            Qvalue=(1-self.alpha)*Qvalue + self.alpha*(reward+self.discount*self.computeValueFromQValues(nextState))
            
        self.q_table[s][action_column]=Qvalue


    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"        
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"        
        return self.computeValueFromQValues(state)



    #auxiliar function created by us:
    def countWalls(self, state):
        numberWalls=[0,0,0,0] #[North,East, South, West]
        walls=state.getWalls()
        xPacman = state.getPacmanPosition()[0]
        yPacman = state.getPacmanPosition()[1]
        if walls[xPacman][yPacman + 1] == True:
            numberWalls[0]=1
        if walls[xPacman][yPacman - 1] == True:
            numberWalls[2]=1
        if walls[xPacman + 1][yPacman] == True:
            numberWalls[1]=1
        if walls[xPacman - 1][yPacman] == True:
            numberWalls[3]=1
        
        return numberWalls
        

    def getReward(self, state, action, nextstate):
        "Return the obtained reward"
        
        "*** YOUR CODE HERE ***" 
        r=0
        s = self.computePosition(state)
        while None in state.data.ghostDistances:
            state.data.ghostDistances[state.data.ghostDistances.index(None)]=90000
        min_Gdistance = min(state.data.ghostDistances)
        ghostIndex = state.data.ghostDistances.index(min_Gdistance)
        
        minG=state.data.ghostDistances[ghostIndex]
        next_minG=nextstate.data.ghostDistances[ghostIndex]
        action_column = self.actions[action]
        cond=True
        
        #walls = state.getWalls()
        numberWallsCurrent=self.countWalls(state)
        numberWallsNext=self.countWalls(nextstate)
        if (numberWallsCurrent.count(1)==2 and numberWallsNext.count(1)==3):   #consider 3-wall boxes that lead to 2-wall paths
            index=numberWallsNext.index(0)
            if index==action_column:
                r=1
            else:
                r=-1
            return r


        
        if next_minG!=None:
            if s==0:
                if action_column==1:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==1:
                if action_column==1:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==2:
                if action_column==2 or action_column==1:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==3: #it cannot move N, so we must encourgae it to move to the sides
                if action_column==1 or action_column==3:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==4: #FInal State: no movement
                r=1
            if s==5:
                if action_column==2:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==6:
                if action_column==3:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==7:
                if action_column==3:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==8:
                if action_column==2 or action_column==3:
                    r=1
                else:
                    r=-1
                    cond=False            
            
            
            if s==9: 
                if action_column==0 or action_column==1:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==10:
                if action_column==1:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==11:
                if action_column==1:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==12:
                if action_column==0:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==13: #FInal State: no movement
                r=1
            if s==14:
                if action_column==1 or action_column==3:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==15:
                if action_column==0 or action_column==3:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==16:
                if action_column==3:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==17:
                if action_column==3:
                    r=1
                else:
                    r=-1
                    cond=False
                    
                    
            
            if s==18: 
                if action_column==0:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==19:
                if action_column==0 or action_column==2:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==20:
                if action_column==2:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==21:
                if action_column==0:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==22: #FInal State: no movement
                r=1
            if s==23:
                if action_column==2:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==24:
                if action_column==0 or action_column==3:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==25:
                if action_column==3:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==26:
                if action_column==2 or action_column==3:
                    r=1
                else:
                    r=-1
                    cond=False            
            
            
            if s==27: 
                if action_column==0 or action_column==1:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==28:
                if action_column==1:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==29:
                if action_column==2 or action_column==1:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==30:
                if action_column==0:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==31: #FInal State: no movement
                r=1
            if s==32:
                if action_column==2:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==33:
                if action_column==0:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==34:
                if action_column==0 or action_column==2:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==35:
                if action_column==2:
                    r=1
                else:
                    r=-1
                    cond=False
                    
                    
            if s==36: 
                if action_column==0 or action_column==1:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==37:
                if action_column==1:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==38:
                if action_column==2 or action_column==1:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==39:
                if action_column==0:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==40: #FInal State: no movement
                r=1
            if s==41:
                if action_column==2:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==42:
                if action_column==0 or action_column==3:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==43:
                if action_column==3:
                    r=1
                else:
                    r=-1
                    cond=False
            if s==44:
                if action_column==2 or action_column==3:
                    r=1
                else:
                    r=-1
                    cond=False
                    
                    
            if minG > next_minG and cond:
                r = r + 1

        return r
            


