# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
from collections.abc import Callable

import pacman
import searchAgents
import util  # util.PriorityQueueWithFunction will be useful to you
from explored import Explored


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class SearchNode:
    """
    Node in a search graph/tree
    """
    def __init__(self, problem, state, parent, action, g_fn, h_fn):
        """
        Create a new search state
        :param problem:  Current problem
        :param state:  current state
        :param parent:  parent SearchNode, use None for initial
        :param action:  action that transitions us from parent node to current node
        :param g_fn: function to estimate cost from start to current node, expects
        :param h_fn:  fucntion to estimate:  current state to goal
        """

        self.state = state
        self.parent = parent
        self.action = action
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        self.g = g_fn(self)
        self.h = h_fn(self, problem)

    def get_state(self):
        """
        Return search node's problem state
        :return:  state
        """
        return self.state

    def get_g(self):
        """
        Return cost/estimated cost g from start to current search node
        :return: cost g
        """
        return self.g

    def get_h(self):
        """
        Return cost/estimated cost h from current search node to goal
        :return: cost h
        """
        return self.h

    def get_f(self):
        """
        Return cost/estimated cost from start to goal
        :return: total cost
        """
        return self.g + self.h

    def get_depth(self):
        """
        Return the number of predecessor search nodes before this one (depth in tree)
        :return: depth
        """
        return self.depth

    def __str__(self):
        """
        :return:  String representation of SearchNode
        """
        if self.parent == None:
            s = f"({self.state}<-Start, f {self.f} = {self.g} + {self.h} (g+h), depth={self.depth})"
        else:
            s = f"({self.state}<-{self.action} from {self.parent.state}, f {self.f} = {self.g} + {self.h} (g+h), depth={self.depth})"
        return s

    def get_path(self):
        """
        Return path from start to node
        :return:  List of tuples:  (action, state) from start to this SearchNode
        """

        # Chase our tail through the SearchNodes until we get to the root
        node = self
        path_nodes = []
        while node is not None:
            path_nodes.append((node.state, node.action))
            node = node.parent
        # Change path_nddes from node --> start to start --> node
        path_nodes.reverse()

        return path_nodes


def graph_search(problem, g, h, verbose=False, debug=False):
    """

    :param problem: Instance of problem class to solve
    :param g: Function for estimating cost from start to a SearchNode
    :param h: Function for estimating cost from a SearchNode to a goal
    :param verbose:  if True, print information about search, e.g. progress messages
    :param debug:  if True, print information that is helpful for debugging
    :return: list of actions to solve problem or None
    """
    # Hint:  Maintain frontier with util.PriorityQueueWithFunction
    # sorted using one of the getter functions provided in SearchNode

    startNode = SearchNode(problem, problem.startState, None, None, g, h)
    startNode.state = (problem.startState, None, None)
    # q
    frontier = util.PriorityQueue()
    # add startNode to q
    frontier.push(startNode, startNode.g + startNode.h)
    done = found = False
    # initialize empty explored set
    explored = Explored().set
    # explored.add(problem)
    gs = problem.goal
    actions = []

    # while we are not done...
    while not done:
        currentNode = frontier.pop()        # pop first item off the q
        explored.add(currentNode.state[0])     # mark it as explored

        successors = problem.getSuccessors(currentNode.state[0])       # get its successors(its child nodes)

        # For every successor node of current node...
        for successor in successors:


            # child node is a new SearchNode(problem, parent, curState,  action, g, h)
            child = SearchNode(problem, successor, currentNode, successor[1], g, h)

            # if child state hasn't been explored...
            if child.state[0] not in explored:
                # if child is the goal state...
                if problem.isGoalState(child.state[0]):
                    done = found = True
                    action = child.action
                    # actions = []    # empty list of actions.

                    # back track the path and append to actions list
                    while child.parent != None:

                        actions.append(child.action)
                        child = child.parent
                    return actions
                else:
                    frontier.push(child, child.g + child.h)


    return None










def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

class DepthFirstSearch:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to define g and h functions appropriately for
    a depth first search that will be conducted by the graph_search algorithm
    that you implement above.


    """
    # The @classmethod decorator indicates that the next function to be defined
    # is a class function.  By tradition, the first argument to a class method is
    # cls (much like self is the first argument to instance methods)
    @classmethod
    def g(cls, node: SearchNode):
        """
        Fill in appropriate comments
        """
        raise NotImplementedError()

    @classmethod
    def h(cls, node: SearchNode, problem):
        """
        Fill in appropriate comments
        """
        raise NotImplementedError()


    @classmethod
    def search(cls, problem):
        """
        Fill in appropriate comments
        """
        raise NotImplementedError()

def depthFirstSearch(problem):
    """
    The pacman framework is not equipped to access member functions of classes as
    parameters.  Work around this by defining a function that accesses the class
    search function.
    :param problem:
    :return:
    """
    return DepthFirstSearch.search(problem)

class BreadthFirstSearch:
    """
    Expand the search tree level by level
    """
    # Hint:  Start by setting up signatures even if you don't implement right away.
    # Python cannot compile this moudule until BFS and A* classes have at least a dummy search signature.
    # Use signatures from DepthFirstSearch

    @classmethod
    def g(cls, node: SearchNode):
        """
        Fill in appropriate comments
        """


        return node.get_depth()

    @classmethod
    def h(cls, node: SearchNode, problem):
        """
        Fill in appropriate comments
        """


        return nullHeuristic(problem)

    @classmethod
    def search(cls, problem):
        """
        Fill in appropriate comments
        """

        path = graph_search(problem, BreadthFirstSearch.g, BreadthFirstSearch.h, True, True)
        path.reverse()
        return path




def breadthFirstSearch(problem):
    """
    The pacman framework is not equipped to access member functions of classes as
    parameters.  Work around this by defining a function that accesses the class
    search function.
    :param problem:
    :return:
    """
    return BreadthFirstSearch.search(problem)


class AStarSearch:
    """
    Expand the search tree based on a consistent heuristic
    """
    # Use signatures from DepthFirstSearch
    @classmethod
    def g(cls, node: SearchNode):
        """
        Fill in appropriate comments
        """
        raise NotImplementedError()

    @classmethod
    def h(cls, node: SearchNode, problem):
        """
        Fill in appropriate comments
        """
        raise NotImplementedError()

    @classmethod
    def search(cls, problem):
        """
        Fill in appropriate comments
        """
        raise NotImplementedError()

def aStarSearch(problem):
    """
    The pacman framework is not equipped to access member functions of classes as
    parameters.  Work around this by defining a function that accesses the class
    search function.

    Conduct an A* search
    :param problem:
    :return:
    """
    return AStarSearch.search(problem)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    return AStarSearch.search(problem)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
