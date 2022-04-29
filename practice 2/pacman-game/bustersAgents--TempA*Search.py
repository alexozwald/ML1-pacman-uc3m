# A* Search Algo Implementation

https://abhinavcreed13.github.io/projects/ai-project-search/

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    # class to represent SearchNode
    class SearchNode:
        """
            Creates node: <state, action, f(s), g(s), h(s), parent_node>
        """
        def __init__(self, state, action=None, g=None, h=None,
                     parent=None):
            self.state = state
            self.action = action
            self.parent = parent
            # heuristic value
            self.h = h
            # combined cost
            if parent:
                self.g = g + parent.g
            else:
                self.g = 0
            # evaluation function value
            self.f = self.g + self.h

        def extract_solution(self):
            """ Gets complete path from goal state to parent node """
            action_path = []
            search_node = self
            while search_node:
                if search_node.action:
                    action_path.append(search_node.action)
                search_node = search_node.parent
            return list(reversed(action_path))


    # make search node function
    def make_search_node(state, action=None, cost=None, parent=None):
        if hasattr(problem, 'heuristicInfo'):
            if parent:
                # same parent - avoid re-calculation
                # for reducing computations in logic
                if parent == problem.heuristicInfo["parent"]:
                    problem.heuristicInfo["sameParent"] = True
                else:
                    problem.heuristicInfo["sameParent"] = False
            # adding parent info for reducing computations
            problem.heuristicInfo["parent"] = parent
        # get heuristic value
        h_value = heuristic(state, problem)
        return SearchNode(state, action, cost, h_value, parent)

    # create open list
    open = util.PriorityQueue()
    node = make_search_node(problem.getStartState())
    open.push(node, node.f)
    closed = set()
    best_g = {}  # maps states to numbers

    # run until open list is empty
    while not open.isEmpty():
        node = open.pop()  # pop-min

        if node.state not in closed or node.g < best_g[node.state]:
            closed.add(node.state)
            best_g[node.state] = node.g

            # goal-test
            if problem.isGoalState(node.state):
                return node.extract_solution()

            # expand node
            successors = problem.getSuccessors(node.state)
            for succ in successors:
                child_node = make_search_node(succ[0],succ[1],succ[2], node)
                if child_node.h < float("inf"):
                    open.push(child_node, child_node.f)

    # no solution
    util.raiseNotDefined()
