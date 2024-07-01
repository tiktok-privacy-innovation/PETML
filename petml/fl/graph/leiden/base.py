# Copyright 2024 TikTok Pte. Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Auxiliary classes for Leiden algorithm.
The Louvain algorithm starts from a singleton partition, with each vertex being assigned to its own community,
and repeatedly moves vertices into new communities with positive modularity gains until the modularity reaches the maximum.

Terms:
- nodes: The node id belong to origin graph.
- vertex: The node belong to Aggrated Graph, A vertex may contain a set of nodes.
During the iterative process of the algorithm, similar vertexs can gradually combine together to form one larger vertex.
- community: A community may contain a set of Vertex.
"""

import collections
import copy
import typing as t


class Vertex:
    """
    Vertex class for Leiden algorithm.

    Parameters
    ----------
    vid : int or str
        Vertex id.

    cid : int or str
        Vommunity id.

    nodes : set
        The node id of the original graph it contains, when merged into a vertex,
        records which original nodes it is composed of.

    degree : int
        Degree of the vertex.

    is_ghost : bool
        Whether the vertex is a ghost vertex. Ghost nodes indicate that some information exists
        on the opposite node.  Ghost nodes exist on both sides simultaneously

    is_cross_merge : bool
        Used to represent merged cross-border supernodes: where some nodes are on one side
        and some nodes are on the other side.
        Due to the fact that cross-border supernodes belong to both the server and client side,
        we defines cross-border supernodes to only perform local move judgment on the server side.
        However, some neighbors of cross-border supernodes are invisible to the server, so it is necessary
        for the client side to calculate the Q values of some neighbor nodes.
    """

    def __init__(self,
                 vid: t.Union[int, str],
                 cid: t.Union[int, str] = None,
                 nodes: t.Set[t.Union[int, str]] = None,
                 degree: int = 0,
                 is_ghost: bool = False,
                 is_cross_merge: bool = False):

        self.vid = vid
        if cid is None:
            cid = vid
        self.cid = cid
        self.nodes = nodes or set()
        self.degree = degree
        self.is_ghost = is_ghost
        self.is_cross_merge = is_cross_merge

    def __repr__(self):
        return f"""
Vertex(
    vid : {self.vid}
    cid : {self.cid}
    nodes : {self.nodes}
    degree : {self.degree}
    is_ghost : {self.is_ghost}
    is_cross_merge : {self.is_cross_merge}
)
"""

    def merge(self, other: "Vertex"):
        """Merge other vertex into self.
        """
        self.nodes.update(other.nodes)
        self.degree += other.degree
        self.is_ghost = self.is_ghost or other.is_ghost
        self.is_cross_merge = self.is_cross_merge or other.is_cross_merge


class Community:
    """
    Community class for Leiden algorithm.

    Parameters
    ----------
    cid : int or str
        Community id.

    vertexs : set
        The vertex it contains.

    weight : float
        The weight of the community.

    is_ghost : bool
        Whether the community is a ghost community.
    """

    def __init__(self,
                 cid: t.Union[int, str],
                 vertexs: t.Set[t.Union[int, str]] = None,
                 weight: float = 0,
                 is_ghost: bool = False):
        self.cid = cid
        self.vertexs = vertexs or set()
        self.weight = weight
        self.is_ghost = is_ghost

    def __repr__(self):
        return f"""
Community(
    cid : {self.cid}
    vertexs : {self.vertexs}
    weight : {self.weight}
    is_ghost : {self.is_ghost}
)
"""

    def remove_vertex(self, vid: t.Union[int, str], degree: float):
        self.vertexs.remove(vid)
        self.weight -= degree

    def add_vertex(self, vid: t.Union[int, str], degree: float):
        if vid not in self.vertexs:
            self.vertexs.add(vid)
            self.weight += degree


class Graph:
    """
    Graph class for Leiden algorithm.

    We use a dict to store the graph. The key is the node id, and the value is a dict,
    the key is the neighbor node id, and the value is the weight.
    """

    def __init__(self) -> None:
        self.data = collections.defaultdict(dict)

    def from_dict(self, d: dict):
        self.data = copy.deepcopy(d)

    def neighbors(self, node_id: t.Union[int, str]) -> t.Iterable:
        return self.data[node_id].keys()

    def weight(self, node_id1: t.Union[int, str], node_id2: t.Union[int, str]) -> float:
        if node_id1 is None or node_id2 is None:
            return 0
        return self.data[node_id1].get(node_id2, 0)

    def update_weight(self, node_id1: t.Union[int, str], node_id2: t.Union[int, str], weight: float):
        self.data[node_id1][node_id2] = weight
