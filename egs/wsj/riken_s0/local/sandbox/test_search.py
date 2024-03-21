from math import log
from numpy import array
from numpy import argmax
import sys

# beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -log(row[j])]
                all_candidates.append(candidate)

        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:k]
        print(all_candidates)
        print("----")

    return sequences

# # define a sequence of 10 words over a vocab of 5 words
# data = [[0.1, 0.2, 0.3, 0.4, 0.5],
#         [0.5, 0.4, 0.3, 0.2, 0.1],
# 	[0.1, 0.2, 0.3, 0.4, 0.5],
# 	[0.5, 0.4, 0.3, 0.2, 0.1],
# 	[0.1, 0.2, 0.3, 0.4, 0.5],
# 	[0.5, 0.4, 0.3, 0.2, 0.1],
# 	[0.1, 0.2, 0.3, 0.4, 0.5],
# 	[0.5, 0.4, 0.3, 0.2, 0.1],
# 	[0.1, 0.2, 0.3, 0.4, 0.5],
# 	[0.5, 0.4, 0.3, 0.2, 0.1]]

# define a sequence of 10 words over a vocab of 5 words
data = [[0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3],
	[0.2, 0.3, 0.4],
	[0.3, 0.2, 0.1]]

data = array(data)
# decode sequence
result = beam_search_decoder(data, 2)
# print result
for seq in result:
    print(seq)
# [[[0], 1.2039728043259361], [[1], 0.916290731874155], [[2], 0.6931471805599453]]
# ----
# [[[2, 0], 0.4804530139182014], [[2, 1], 0.6351243373717793], [[2, 2], 0.8345303547893733], [[1, 0], 0.6351243373717793], [[1, 1], 0.8395887053184746], [[1, 2], 1.1031891220323908]]
# ----
# [[[2, 0, 0], 0.7732592957431818], [[2, 0, 1], 0.5784523625139449], [[2, 0, 2], 0.4402346437542523], [[2, 1, 0], 1.0221931876757278], [[2, 1, 1], 0.7646724295611531], [[2, 1, 2], 0.5819585439214754]]
# ----
# [[[2, 0, 2, 0], 0.5300305386022366], [[2, 0, 2, 1], 0.7085303260250136], [[2, 0, 2, 2], 1.0136777281280855], [[2, 0, 1, 0], 0.6964409130648773], [[2, 0, 1, 1], 0.930983162767017], [[2, 0, 1, 2], 1.3319357869317971]]
# ----
# [[2, 0, 2, 0], 0.5300305386022366]
# [[2, 0, 1, 0], 0.6964409130648773]


# def greedy_decoder(data):
#     # index for largest probability each row
#     return [argmax(s) for s in data]
# result = greedy_decoder(data)
# print(result)
# greedy decoder

