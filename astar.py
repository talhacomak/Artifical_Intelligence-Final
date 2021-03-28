#!/usr/bin/env python
# manual
class Node():
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
# Adapted from: https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
def astar(maze, start, end):
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    open_list = []
    closed_list = []

    open_list.append(start_node)

    while len(open_list) > 0:

        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        open_list.pop(current_index)
        closed_list.append(current_node)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] 

        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:

            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            new_node = Node(current_node, node_position)
            children.append(new_node)

        for child in children:

            for closed_child in closed_list:
                if child == closed_child:
                    continue
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            open_list.append(child)

class AStarAgent():
    def __init__(self):
        self.maze = [[1,1,1,1,1,1,1,1],
                    [1,0,0,0,0,0,0,1],
                    [1,0,1,0,1,1,0,1],
                    [1,0,0,0,1,1,0,1],
                    [1,0,1,0,1,0,0,1],
                    [1,0,1,0,0,0,1,1],
                    [1,1,1,1,1,1,1,1]]
    def find_new_path(self, source_point, destination_point):
        start = source_point
        end   = destination_point
        path = astar(self.maze, start, end)
        return path
    def remove_grid_from_maze(self, x, y):
        self.maze[x][y] = 1

"""
tiles:
- [grass, grass, grass, grass, grass, grass, grass, grass]
- [grass, curve_left/W, straight/E, straight/E, straight/E, straight/E, curve_left/N, grass]
- [grass, straight/S, grass, grass, grass, grass, straight/S, grass]
- [grass, straight/S, grass, grass, grass, grass, straight/S, grass]
- [grass, straight/S, grass, grass, curve_left/W, straight/E, curve_left/E, grass]
- [grass, curve_left/S, straight/E, straight/E, curve_left/E, grass, grass, grass]
- [grass, grass, grass, grass, grass, grass, grass, grass]
[[1,1,1,1,1,1,1,1],
[1,0,0,0,0,0,0,1],
[1,0,1,0,1,1,0,1],
[1,0,0,0,1,1,0,1],
[1,0,1,0,1,0,0,1],
[1,0,0,0,0,0,1,1],
[1,1,1,1,1,1,1,1]]
"""

