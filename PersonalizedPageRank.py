import time

import numpy as np
import pandas as pd

"""Function to take user input
1 user input is taken :
    1. input_images : As set of 3 input image nodes taken from the user for PPR Algorithm
"""


def userInput():
    input_nodes = []
    print('Please enter input image IDs :')
    for _ in range(0, 3):
        input_nodes.append(input())

    return input_nodes


"""
Main function to calculate the Page Rank of all the nodes of the graph.
This function takes 4 optional input parameters.
    1. graph - Mandatory parameter of type DataFrame which takes an adjacency matrix of the graph.
    2. input_nodes - Optional parameter which accepts a list of input graph nodes. If input nodes are not provided, it is taken as input from user.
    3. beta - Damping factor.   Default Value : 0.85
    4. epsilon : Error Threshold of Page Rank scores. Default Value : 0.0000001
"""


def pageRank(graph, input_nodes=None, beta=0.85, epsilon=0.0000001):
    # Intialize Matrix M used in PageRank calculation
    M = graph / np.sum(graph, axis=0)

    # Initializing Teleportation matrix and Page Rank Scores with Zeros for all graph nodes
    nodes = len(graph)
    teleportation_matrix = np.zeros(nodes)
    pageRankScores = np.zeros(nodes)

    # Takes user input if input_nodes are not provided
    if input_nodes is None:
        input_nodes = userInput()

    # Updating Teleportation and Page Rank Score Matrices with 1/num_of_input_nodes for the input nodes.
    for node_id in input_nodes:
        teleportation_matrix[int(node_id)] = 1 / len(input_nodes)
        pageRankScores[int(node_id)] = 1 / len(input_nodes)

    print('Calculating Personalized PageRank Scores with a Damping Factor of ' + str(beta) + '...')

    # Calculating Page Rank Scores
    while True:
        oldPageRankScores = pageRankScores
        pageRankScores = (beta * np.dot(M, pageRankScores)) + ((1 - beta) * teleportation_matrix)
        if np.linalg.norm(pageRankScores - oldPageRankScores) < epsilon:
            break

    # Normalizing Page Rank Scores
    pageRankScores = pageRankScores / sum(pageRankScores)

    return pageRankScores


if __name__ == '__main__':
    start_time = time.time()

    file_name = 'NodeGraph.csv'
    adjacency_matrix = pd.DataFrame(pd.read_csv(file_name, index_col=0).values)

    pageRankScores = pageRank(graph=adjacency_matrix, beta=0.5)

    print(pageRankScores)

    end_time = time.time()
    print('Total Time : ', end_time - start_time)
