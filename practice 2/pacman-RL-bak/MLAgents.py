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


        self.epsilon = 0.1
        self.alpha = 0.9
        self.discount = 0.8
        
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
        xy_ghost = state.getGhostPositions()[min_g_idx]
        xy_pacmn = state.getPacmanPosition()

        optimal_dir = self.bfs(state, walls, xy_pacmn, xy_ghost)
        if optimal_dir == Directions.NORTH:
            row = state.data.layout.height-2
        elif optimal_dir == Directions.SOUTH:
            row = state.data.layout.height
        else:
            row = state.data.layout.height-1

        return row

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
        r = reward
        s = self.computePosition(state)
        action_col = self.actions[action]
        q_val = self.getQValue(state, action)

        if r != 0:
            q_val = (1-self.alpha) * q_val + self.alpha * (r)
        else:
            q_val = (1-self.alpha) * q_val + self.alpha * (r+self.discount*self.computeValueFromQValues(nextState))

        self.q_table[s][action_col] = q_val
        return q_val


    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)

    def getReward(self, state, action, nextstate):
        "Return the obtained reward"
        
        "*** YOUR CODE HERE ***"
        r = 1

        # terminal state: munch on ghost
        # next pacman location
        xy_pacman = list(nextstate.getPacmanPosition())
        if xy_pacman in state.getGhostPositions():
            r += 2

        # +update state: eat dot food
        if state.hasFood(xy_pacman[0],xy_pacman[1]):
            r += 0.5

        # if dist increases or decreases...  (to closest ghost)
        min_dist_og = sys.maxsize
        for i in [x for x in state.data.ghostDistances if type(x)==int]:
            if i < min_dist_og:   min_dist_og = i
        min_dist_next = sys.maxsize
        for i in [x for x in nextstate.data.ghostDistances if type(x)==int]:
            if i < min_dist_next:   min_dist_next = i
        
        if (min_dist_next-min_dist_og) > 0:
            r -= 0.25
        if (min_dist_next-min_dist_og) < 0:
            r += 0.25

        # check bfs
        w = state.getWalls()
        p = state.getPacmanPosition()
        g = state.getGhostPositions()[state.data.ghostDistances.index(min_dist_og)]
        optimal_dir = self.bfs(state, w, p, g)
        #chosen = nextstate.getPacmanState().getDirection()
        if action == self.actions[optimal_dir]:
            r += 0.75
        else:
            r -= 0.25

        return r


        # -update state: moved farther from closest ghost. (by 1 or by 2 is even worse)



        util.raiseNotDefined()


    #######################
    # CUSTOM MEMBER FUNCS #
    #######################
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
