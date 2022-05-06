# qLearnAgent.py
# ----------------
# custom added by Alex Oswald to save space

from bustersAgents import BustersAgent
import util
from game import Directions
import numpy as np
import os.path

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
        self.actions = {"North":0, "East":1, "South":2, "West":3}
        if os.path.exists("qtable.txt"):
            self.table_file = open("qtable.txt", "r+")
            self.q_table = self.readQtable()
        else:
            self.table_file = open("qtable.txt", "w+")
            #"*** CHECK: NUMBER OF ROWS IN QTABLE DEPENDS ON THE NUMBER OF STATES ***"
            self.initializeQtable(9)

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

        values = [self.getQValue(state, action) for action in state.getLegalActions(state)]

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
