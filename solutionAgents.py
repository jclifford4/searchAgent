
from pacman import Directions
from game import Agent, Actions
from pacmanAgents import LeftTurnAgent


class GoldTimidAgent(Agent):
    """
    A simple reflex agent for PacMan
    """

    def __init__(self):
        super().__init__()
        self.secondaryAgent = LeftTurnAgent()

    def inDanger(self, pacman, ghost, dist=3):
        """inDanger(pacman, ghost) - Is the pacman in danger
        For better or worse, our definition of danger is when the pacman and
        the specified ghost are:
           in the same row or column,
           the ghost is not scared,
           and the agents are <= dist units away from one another

        If the pacman is not in danger, we return Directions.STOP
        If the pacman is in danger we return the direction to the ghost.
        """

        danger = Directions.STOP  # no danger until we find otherwise

        # No danger if ghosts are scared.
        if not ghost.isScared():
            # Ghosts are not scared, look out...

            # Pacman position
            xy1 = pacman.getPosition()

            # Ghost position and direction
            xy2 = ghost.getPosition()

            # Find x, y distances between agents.
            # If in the same row or column, check to see if they are within
            # dist units of each other.
            deltas = [ (xy2[i]-xy1[i]) for i in range(2)]
            absdeltas = [abs(d) for d in deltas]
            # With numpy, we could use argmin to determine which axis is the
            # minimum instead of looping as we do here.
            if min(absdeltas) == 0:  # same row/col
                # Return the direction to the ghost if close enough
                for idx in range(len(deltas)):
                    ax = (idx + 1) % 2  # get idx of opposite axis
                    if deltas[idx] == 0 and absdeltas[ax] <= dist:
                        # A ghost is in the same row and column and close.
                        # Find direction of ghost
                        danger = Actions.vectorToDirection(deltas)
                        break

        return danger

    def getAction(self, state):
        """
        state - GameState
        """
        me = state.getPacmanState()
        mypos = me.getPosition()

        others = state.getGhostStates()

        legal = state.getLegalPacmanActions()

        action = None  # no action yet
        for ghost in others:
            dangerFrom = self.inDanger(me, ghost)
            if dangerFrom != Directions.STOP:
                # Oh oh, ghost nearby.
                # Move away from the ghost.  Try the opposite direction first,
                # then left, right
                flightDirections = [Directions.REVERSE, Directions.LEFT,
                                Directions.RIGHT]
                for dir in flightDirections:
                    newHeading = dir[dangerFrom]
                    if newHeading in legal:
                        action = newHeading
                        break

                if action is None:
                    if dangerFrom in legal:
                        action = dangerFrom  # Continue into danger...
                    else:
                        action = Directions.STOP  # no legal move

        if action is None:
            action = self.secondaryAgent.getAction(state)


        return action
