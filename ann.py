import numpy as np
import __future__
from typing import List
import random
import io
from dataclasses import dataclass
import pickle
import time

"""
This is a python implementation of an approximate nearest neighbor search algorithm.
The algorithm is inspired by Spotify recommendation algo Annoy: https://github.com/spotify/annoy. 
We recursively build random hyperplanes to split the vector space into two parts. At query time, 
we find the approximate nearest neighbors by traversing the tree. 
Then build a forest of trees to improve the accuracy.

Hyperplane: (x - x0) * n = 0

Basic benchmark:
- 5 trees, 1m vectors, 300d: 2ms/query
"""


class Hyperplane:
    # (x - x0) * n = 0
    n: np.ndarray
    x0: float

    def __init__(self, n, x0) -> None:
        self.n = n
        self.x0 = x0

    def is_on_top(self, u: np.ndarray):
        return np.dot(u - self.x0, self.n) > 0


class Tree:
    @dataclass
    class Node:
        # non leaf node
        left = None
        right = None
        hyperplane: Hyperplane = None

        # leaf node
        is_leaf: bool = False
        ids: List[int] = None

        def __init__(
            self, is_leaf=False, ids=None, left=None, right=None, hyperplane=None
        ) -> None:
            self.is_leaf = is_leaf
            self.ids = ids
            self.left = left
            self.right = right
            self.hyperplane = hyperplane

    root: Node
    vecs: np.ndarray
    MAX_NODE_SIZE: int = 10

    def __init__(self, vecs) -> None:
        self.vecs = vecs
        self.root = self.index(list(range(len(vecs))))

    def index(self, ids: List[int]) -> Node:
        # logging
        if len(ids) > 10_000:
            print(f"Indexing {len(ids)} vectors..")

        # leaf node
        if len(ids) <= self.MAX_NODE_SIZE:
            return Tree.Node(is_leaf=True, ids=ids)

        # build hyperplane
        vs = random.sample(ids, 2)
        a, b = self.vecs[vs[0]], self.vecs[vs[1]]
        n = a - b
        x0 = (a + b) / 2
        hp = Hyperplane(n=n, x0=x0)

        # split vectors based on hyperplane
        left_ids, right_ids = [], []
        for i in ids:
            if hp.is_on_top(self.vecs[i]):
                left_ids.append(i)
            else:
                right_ids.append(i)

        # build node
        left_node = self.index(left_ids)
        right_node = self.index(right_ids)
        return Tree.Node(left=left_node, right=right_node, hyperplane=hp)

    def query(self, u: np.ndarray, top_k: int = 10) -> List[int]:
        # find at least top_k candidates
        def _query(node: Tree.Node, num: int):
            if node.is_leaf:
                return node.ids

            primary, secondary = None, None
            if node.hyperplane.is_on_top(u):
                primary = node.left
                secondary = node.right
            else:
                primary = node.right
                secondary = node.left
            cands = _query(primary, num)
            if len(cands) < num:
                cands.extend(_query(secondary, num - len(cands)))
            return cands

        return _query(self.root, top_k)


class Forest:
    trees: List[Tree]
    vecs: np.ndarray

    def __init__(self, vecs: np.ndarray, num_trees: int) -> None:
        self.vecs = vecs
        self.trees = [Tree(vecs=vecs) for _ in range(num_trees)]

    def query(self, u: np.ndarray, top_k: int = 10) -> List[int]:
        ids = set()
        for t in self.trees:
            ids.update(t.query(u, top_k))
        ids = list(ids)
        # n x d
        candidates = self.vecs[ids]
        # n x 1
        dists = np.dot(candidates, u)
        # [(id, dist)]
        ids_dists = [(idx, dists[i]) for i, idx in enumerate(ids)]
        sorted_dists = sorted(ids_dists, key=lambda x: x[1], reverse=True)
        top_ids = [i[0] for i in sorted_dists[:top_k]]
        return top_ids


def load_vectors(fname):
    fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(" ")
        map_obj = map(float, tokens[1:])
        data[tokens[0]] = list(map_obj)
    return data


# vector_map = load_vectors("data/wiki-news-300d-1M.vec")
# names = list(vector_map.keys())
# values = list(vector_map[k] for k in names)
# vectors = np.array(values)

# np.save("data/vectors.npy", vectors)
# with open("data/names.txt", "w") as file:
#     for item in names:
#         file.write("%s\n" % item)

vectors = np.load("data/vectors.npy")
with open("data/names.txt", "r") as file:
    names = [line.strip() for line in file.readlines()]

# forest = Forest(vecs=vectors, num_trees=5)
with open("data/forest.pkl", "rb") as f:
    forest = pickle.load(f)

target = "king"
index = names.index(target)
res = forest.query(u=vectors[index], top_k=10)
matches = [names[i] for i in res]
print(matches)

# benchmark
start_time = time.time()
num = 10000
for i in range(num):
    idx = random.sample(range(len(vectors)), 1)[0]
    res = forest.query(u=vectors[idx], top_k=10)
end_time = time.time()
print(f"Querying latency: {(end_time - start_time) / float(num):.4f}s")
