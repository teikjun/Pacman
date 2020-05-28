# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None


def closestCapsule(pos, capsules, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        # if food[pos_x][pos_y]:
        #     return dist

        if (pos_x, pos_y) in capsules:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None


class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

class NewExtractor(FeatureExtractor):
    """
    Design you own feature extractor here. You may define other helper functions you find necessary.
    - number of ghosts one and two steps away
    - number of munchies one or two steps away
    - sum of food dist

    """
    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        capsules = state.getCapsules()
        ghost_states = state.getGhostStates()

        scared_ghosts = list()
        normal_ghosts = list()

        for g in ghost_states:
            if g.scaredTimer and (g.scaredTimer > 2):
                scared_ghosts.append(g)
            else:
                normal_ghosts.append(g)

        scared_ghosts_positions = map(lambda g: g.getPosition(), scared_ghosts)
        normal_ghosts_positions = map(lambda g: g.getPosition(), normal_ghosts)

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        closest_food = None
        closest_scared_ghost = None
        closest_normal_ghost = None
        closest_capsule = None

        fringe = [(next_x, next_y, 0)]
        expanded = set()

        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if closest_food is None:
                if food[pos_x][pos_y]:
                    closest_food = dist
                    
            if closest_normal_ghost is None:
                if (pos_x, pos_y) in normal_ghosts_positions:
                    closest_normal_ghost = dist

            if closest_scared_ghost is None:
                if (pos_x, pos_y) in scared_ghosts_positions:
                    closest_scared_ghost = dist

            if closest_capsule is None:
                if (pos_x, pos_y) in capsules:
                    closest_capsule = dist

            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))

        if closest_food is not None:
            features["closest-food"] = float(closest_food) / (walls.width * walls.height)

        if closest_normal_ghost is not None:
            features["closest-normal-ghost"] = float(closest_normal_ghost) / (walls.width * walls.height)

        if closest_scared_ghost is not None:
            features["closest-scared-ghost"] = float(closest_scared_ghost) / (walls.width * walls.height)

        scared_ghost_ratio = len(scared_ghosts) / len(ghost_states)

        if closest_capsule is not None:
            features["closest-cap"] = float(closest_capsule) / (walls.width * walls.height)

        # count the number of ghosts 1-step away
        # features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in normal_ghosts_positions)

        # features["#-of-sghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in scared_ghosts_positions)

        num_normal_ghosts_one_step_away = 0
        num_scared_ghosts_one_step_away = 0
        num_normal_ghosts_two_step_away = 0
        num_scared_ghosts_two_step_away = 0

        for g in normal_ghosts_positions:
            for p in Actions.getLegalNeighbors(g, walls):
                if p == (next_x, next_y):
                    num_normal_ghosts_one_step_away +=1

                for q in Actions.getLegalNeighbors(p, walls):
                    if q == (next_x, next_y):
                        num_normal_ghosts_two_step_away +=1

                
        for g in scared_ghosts_positions:
            for p in Actions.getLegalNeighbors(g, walls):
                if p == (next_x, next_y):
                    num_scared_ghosts_one_step_away +=1

                for q in Actions.getLegalNeighbors(p, walls):
                    if q == (next_x, next_y):
                        num_scared_ghosts_two_step_away +=1
                


        features["#-of-ghosts-1-step-away"] = num_normal_ghosts_one_step_away
        features["#-of-sghosts-1-step-away"] = num_scared_ghosts_one_step_away

        features["#-of-ghosts-2-step-away"] = num_normal_ghosts_two_step_away
        features["#-of-sghosts-2-step-away"] = num_scared_ghosts_two_step_away


        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        if not features["#-of-sghosts-1-step-away"] and ((next_x, next_y) in scared_ghosts_positions) and ((next_x, next_y) not in normal_ghosts_positions):
            features["eats-ghost"] = 1.5

        if not features["#-of-ghosts-1-step-away"] and ((next_x, next_y) in capsules):
            features["eats-cap"] = 1.0
    

        features.divideAll(10.0)
        return features


        
