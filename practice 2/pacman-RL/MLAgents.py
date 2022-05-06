# qLearnAgent.py
# ----------------
# custom added by Alex Oswald to save space

from bustersAgents import BustersAgent
from game import Directions, Actions
from distanceCalculator import Distancer
import util
import sys
import random
import numpy as np
import os.path

# random addendum
class queue(list):
    def empty(self):
        if len(self) == 0:
            return True
        else:
            return False

    def push(self, __object) -> None:
        return super().append(__object)

#############################
# Assignment 2 Target Class #
#############################
class QLearningAgent(BustersAgent):

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True, elapseTimeEnable=True):
        super().__init__(index, inference, ghostAgents, observeEnable, elapseTimeEnable)

        #print("yay we're __init__'ing!!!")

    #Initialization
    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)


        self.epsilon = 0.0
        self.alpha = 0.5
        self.discount = 0.8
        #self.history = [] # (Direction.XXXX, int)
        #if self.history[-1][1] == "broke bitch":
        #    do something else
        #self.history.append((curr_move, int))
        
        self.actions = {"North":0, "East":1, "South":2, "West":3}

        # check layout height / how many rows there should be
        req_height = gameState.data.layout.height

        if os.path.exists("qtable.txt"):
            self.table_file = open("qtable.txt", "r+")
            self.q_table = self.readQtable()

            # fix qtable height if necessary
            if not len(self.q_table[0]) == req_height:
                self.initializeQtable(req_height)
        else:
            self.table_file = open("qtable.txt", "w+")
            #"*** CHECK: NUMBER OF ROWS IN QTABLE DEPENDS ON THE NUMBER OF STATES ***"
            self.initializeQtable(req_height)

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


    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        """
        
        "*** YOUR CODE HERE ***"

        # closest ghost
        min_dist = sys.maxsize
        for i in [x for x in state.data.ghostDistances if type(x)==int]:
            if i < min_dist:   min_dist = i
        min_g_idx = state.data.ghostDistances.index(min_dist)

        # get wall matrix
        walls = state.getWalls()

        # ghost + pacman coordinates
        x_ghost = state.getGhostPositions()[min_g_idx][0]
        y_ghost = state.getGhostPositions()[min_g_idx][1]
        x_pacmn = state.getPacmanPosition()[0]
        y_pacmn = state.getPacmanPosition()[1]

        ##xyx

        return state.data.layout.height-1

        #values = [self.getQValue(state, action) for action in state.getLegalActions()]

        util.raiseNotDefined()


    def getQValue(self, state, action):

        """
            Returns Q(state,action)
            Should return 0.0 if we have never seen a state
            or the Q node value otherwise
        """
        position = self.computePosition(state)
        action_column = self.actions[action]
        return self.q_table[position][action_column]

        util.flipCoin()


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
        if state.getPacmanPosition() == state.getGhostPosition():
            r = 1
        elif state == (3,1):
            r = -1
        else:
            r = 0

        Q_value = self.getQValue(state, action)
        
        if state == (3,2) or state==(3,1):
            Q_value = (1-self.alpha) * Q_value + self.alpha * (r)
        else:
            Q_value = (1-self.alpha) * Q_value + self.alpha * (r+self.discount*self.computeValueFromQValues(nextState))

        position = self.computePosition(state)
        action_column = self.actions[action]

        self.q_table[position][action_column] = Q_value



        # terminal state: munch on ghost


        # +update state: eat dot food


        # -update state: moved farther from closest ghost. (by 1 or by 2 is even worse)



        util.raiseNotDefined()



    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)

    def getReward(self, state, action, nextstate):
        "Return the obtained reward"
        
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    ## custom addition ##
    # Determines the simplistic optimal cardinal direction to head in to reach a ghost
    def getBestDirection(self, pacman_pos: tuple, ghost_posi: tuple) -> str:
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

    def bfs(self, state, walls, pacman, ghost):
        # setup
        pacman = tuple(pacman)
        ghost = tuple(ghost)

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
            if curr_xy == ghost:
                path_found = True
                break
            for next_xy in random.shuffle(self.getMovesHypoth(walls, curr_xy)):
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
