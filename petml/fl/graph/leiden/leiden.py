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

import time
import copy
import json
import random

import numpy as np
import pandas as pd
import petace.securenumpy as snp

from petml.fl.base import FlBase
from .base import Vertex, Community, Graph
from .utils import Cache, Message


class Leiden(FlBase):
    """
    Leiden alogrithm.

    Parameters
    ----------------
    max_move : int
        The limit times of local node move.

    max_merge : int
        The limit times of merge meta nodes.
    """

    def __init__(self, max_move: int = 10, max_merge: int = 10):
        super().__init__()
        self.max_move = max_move
        self.max_merge = max_merge

        self.vertex_table: dict = {}
        self.community_table: dict = {}
        self.local_vertexs: dict = {}
        self.G_vertex_vertex = Graph()
        self.G_vertex_community = Graph()
        self.total_weight: float = 0

        self.cache = Cache()
        self.first_loop = True
        self.next_visit_sequence = set()

    def set_infra(self, party_id, federation, mpc_engine):
        self.party_id = party_id
        self.peer_party = 1 - party_id
        self._federation = federation
        self._mpc_engine = mpc_engine

    def send(self, state, data=None):
        message = Message(state, data)
        self._federation.remote(json.dumps(message.dumps()).encode())

    def recv(self) -> Message:
        return Message(**json.loads(self._federation.get().decode()))

    def __init_attr(self, V, G):
        self.local_vertexs = V
        self.G_vertex_vertex.from_dict(G)
        self.G_vertex_community.from_dict(G)

        # init vertex_table
        for vid, adj_info in G.items():
            for adj_vid, weight in adj_info.items():
                if adj_vid > vid:
                    self.total_weight += weight
                    is_ghost = False
                    if vid not in V or adj_vid not in V:
                        is_ghost = True
                    self.__update_vertex_table(V, vid, weight, is_ghost)
                    self.__update_vertex_table(V, adj_vid, weight, is_ghost)

        # init community_table
        for vid, vertex in self.vertex_table.items():
            self.community_table[vid] = Community(vid, {vid}, vertex.degree, vertex.is_ghost)

    def __update_vertex_table(self, V, vid, weight, is_ghost):
        if vid not in self.vertex_table:
            vertex = Vertex(vid)
            if vid in V:
                vertex.nodes = {vid}
            self.vertex_table[vid] = vertex
        vertex = self.vertex_table[vid]
        vertex.degree += weight
        vertex.is_ghost = is_ghost or vertex.is_ghost

    def sync_total_weight(self):
        self.send(MessageState.SyncTotalWeight, self.total_weight)
        recdata = self.recv()
        self.total_weight += recdata.data
        self.logger.info('sync total_weight successfully')

    def move_nodes(self, vid, new_cid):
        """move vertex vid to a new community new_cid."""
        old_cid = self.vertex_table[vid].cid
        self.move_nodes_local(vid, old_cid, new_cid)

        # Ghost nodes need to change information on both sides at the same time
        if self.vertex_table[vid].is_ghost:
            self.send_move_node_info(vid, old_cid, new_cid)

    def move_nodes_local(self, vid, old_cid, new_cid):
        """move vertex vid to a new community new_cid locally."""
        if new_cid not in self.community_table:
            self.community_table[new_cid] = Community(new_cid, is_ghost=True)

        self.vertex_table[vid].cid = new_cid
        self.community_table[old_cid].remove_vertex(vid, self.vertex_table[vid].degree)
        self.community_table[new_cid].add_vertex(vid, self.vertex_table[vid].degree)
        self.community_table[
            new_cid].is_ghost = self.community_table[new_cid].is_ghost or self.vertex_table[vid].is_ghost

        for neighbor_vid in self.G_vertex_vertex.neighbors(vid):
            old_cid_weight = self.G_vertex_community.weight(neighbor_vid, old_cid) - self.G_vertex_vertex.weight(
                vid, neighbor_vid)
            new_cid_weight = self.G_vertex_community.weight(neighbor_vid, new_cid) + self.G_vertex_vertex.weight(
                vid, neighbor_vid)
            self.G_vertex_community.update_weight(neighbor_vid, old_cid, old_cid_weight)
            self.G_vertex_community.update_weight(neighbor_vid, new_cid, new_cid_weight)

            if neighbor_vid in self.local_vertexs and self.vertex_table[neighbor_vid].cid != new_cid:
                self.next_visit_sequence.add(neighbor_vid)

    def send_move_node_info(self, vid, old_cid, new_cid):
        self.send(MessageState.SyncMoveNodeInfo, {'vid': vid, 'old_cid': old_cid, 'new_cid': new_cid})

    def receive_move_node_info(self, info: dict):
        vid = info["vid"]
        old_cid = info["old_cid"]
        new_cid = info["new_cid"]
        self.move_nodes_local(vid, old_cid, new_cid)

    def send_MPC_find_max_direction(self, move_flag, max_Q):
        self.send(MessageState.MPCFindMaxDirection)
        return self.MPC_find_max_direction(True, move_flag, max_Q)

    def receive_MPC_find_max_direction(self, _):
        self.MPC_find_max_direction(False)

    def send_move_node(self, vid, max_direction):
        self.send(MessageState.MoveNode, {'vid': vid, "max_direction": max_direction})
        recdata = self.recv()
        self.receive_move_node_info(recdata)

    def receive_move_node(self, info: dict):
        vid = info["vid"]
        max_direction = info["max_direction"]
        self.move_nodes(vid, max_direction)

    def send_clear_data(self):
        self.cache.clear()
        self.send(MessageState.ClearCache)

    def receive_clear_data(self, _):
        self.cache.clear()

    def sync_MPC_Q_info(self, vid, w_cids):
        self.cache.vid = vid
        cid = self.vertex_table[vid].cid
        self.cache.cid = cid
        self.cache.w_cids += w_cids
        self.ask_sync_Q_info(vid, cid, w_cids)

    def ask_sync_Q_info(self, vid, cid, w_cids):
        m = len(w_cids)
        info = {"vid": None, "cid": None, "w_cids": [None for _ in range(m)]}
        if self.vertex_table[vid].is_ghost:
            info['vid'] = vid
        if self.community_table[cid].is_ghost:
            info['cid'] = cid

        for i, w_cid in enumerate(w_cids):
            if self.community_table[w_cid].is_ghost:
                info['w_cids'][i] = w_cid

        self.send(MessageState.SyncQInfo, info)

    def receive_sync_Q_info(self, info: dict):
        self.cache.vid = info["vid"]
        self.cache.cid = info["cid"]
        self.cache.w_cids += info["w_cids"]

    def prepare_Q_data(self, vid, cid, w_cids):
        weight_c1 = []
        weight_u_c1 = []
        weight_u_c0 = 0
        weight_c0 = 0
        degree_u = 0

        if cid is not None:
            weight_c0 = self.community_table[cid].weight
        if vid is not None:
            degree_u = self.vertex_table[vid].degree
        if vid is not None and cid is not None:
            weight_u_c0 = 2 * self.G_vertex_community.weight(vid, cid)

        for w_cid in w_cids:
            weight_c1_tmp = 0
            weight_u_c1_tmp = 0
            if w_cid is not None:
                weight_c1_tmp = self.community_table[w_cid].weight
                if vid is not None:
                    weight_u_c1_tmp = 2 * self.G_vertex_community.weight(vid, w_cid)
            weight_c1.append(weight_c1_tmp)
            weight_u_c1.append(weight_u_c1_tmp)

        # to avoid some petace bug
        weight_c1 = np.reshape(weight_c1, (-1, 1)) * 1.0
        weight_u_c1 = np.reshape(weight_u_c1, (-1, 1)) * 1.0
        shape = weight_c1.shape
        weight_u_c0 = np.broadcast_to(weight_u_c0, shape) * 1.0
        weight_c0 = np.broadcast_to(weight_c0, shape) * 1.0
        degree_u = np.broadcast_to(degree_u, shape) * 1.0

        return weight_u_c0, weight_u_c1, weight_c0, weight_c1, degree_u

    def calculate_and_find_maxQ(self, vid, w_cids):
        """
        Consider move vertex u from community c0 to community c1,
        we use Q to measure the modularity gain:

        Q = weight_c1 - weight_c0 + (degree_u * degree_c0 - degree_u * degree_u - degree_u * degree_c1) / m

        where:
        - weight_u_c1: the weight of the edge between vertices u and community c1
        - weight_u_c0: the weight of the edge between vertices u and community c0
        - degree_u: the degree of vertex u
        - weight_c0: the sum of the weights of edges incident to two vertices in c0
        - weight_c1: the sum of the weights of edges incident to two vertices in c1
        - m: the sum of all of the edge weights in the graph
        """
        flag = True
        c0 = self.vertex_table[vid].cid
        weight_u_c0, weight_u_c1, weight_c0, weight_c1, degree_u = self.prepare_Q_data(vid, c0, w_cids)
        Q = weight_u_c1 - weight_u_c0 + degree_u * (weight_c0 - degree_u - weight_c1) / self.total_weight
        max_Q = np.max(Q)
        max_direction = w_cids[np.argmax(Q)]
        if max_Q < 0 or max_direction is None:
            flag = False
            max_Q = -1
            max_direction = None
        return (flag, max_Q, max_direction)

    def judge_if_move_node(self, vid):
        self.send_clear_data()
        w_cid_intern = []
        w_cid_MPC = []
        move_flag = False
        max_Q = -1
        max_direction = None
        viewed_vertex = {self.vertex_table[vid].cid}
        for w_vid in self.G_vertex_vertex.neighbors(vid):
            w_cid = self.vertex_table[w_vid].cid
            if w_cid not in viewed_vertex:
                viewed_vertex.add(w_cid)
                if self.vertex_table[vid].is_ghost or self.community_table[w_cid].is_ghost:
                    w_cid_MPC.append(w_cid)
                else:
                    w_cid_intern.append(w_cid)

        if w_cid_intern:
            move_flag, max_Q, max_direction = self.calculate_and_find_maxQ(vid, w_cid_intern)

        if w_cid_MPC:
            self.sync_MPC_Q_info(vid, w_cid_MPC)

        # For cross-border super nodes, because there may be some neighbor servers that cannot be observed,
        # it is necessary to ask clients whether there is a possibility of moving.
        if self.vertex_table[vid].is_cross_merge:
            self.ask_if_move_merge_node(vid)

        if len(self.cache.w_cids) > 0:
            new_max_direction = self.send_MPC_find_max_direction(move_flag, max_Q)
            if new_max_direction != -1:
                if vid in self.vertex_table and new_max_direction in self.community_table:
                    self.move_nodes(vid, new_max_direction)
                else:
                    self.send_move_node(vid, new_max_direction)
                return True

        if move_flag:
            self.move_nodes(vid, max_direction)
            return True
        return False

    def ask_if_move_merge_node(self, vid):
        self.send(MessageState.IfMoveMergeNode, vid)
        message = self.recv()
        if message.state != MessageState.EndMoveMergeNode:
            self.receive_sync_Q_info(message)
            # receive end state
            self.recv()

    def receive_if_move_merge_node(self, vid):
        v_MPC = []
        for w_cid in self.G_vertex_community.neighbors(vid):
            if self.vertex_table[vid].is_ghost or self.community_table[w_cid].is_ghost:
                v_MPC.append(w_cid)
        self.sync_MPC_Q_info(vid, v_MPC)
        self.send(MessageState.EndMoveMergeNode)

    def build_visit_sequnce(self):
        if self.first_loop:
            visit_sequence = list(self.local_vertexs)
            self.first_loop = False
        else:
            visit_sequence = list(self.next_visit_sequence)
        self.next_visit_sequence = set()
        random.shuffle(visit_sequence)
        return visit_sequence

    def first_stage(self):
        visit_sequence = self.build_visit_sequnce()
        move_flag = False
        for v_vid in visit_sequence:
            if self.judge_if_move_node(v_vid):
                move_flag = True
        self.send(MessageState.FirstStageEnd, move_flag)
        return move_flag

    def listen_local_move(self):
        handler = {
            MessageState.SyncQInfo: self.receive_sync_Q_info,
            MessageState.IfMoveMergeNode: self.receive_if_move_merge_node,
            MessageState.SyncMoveNodeInfo: self.receive_move_node_info,
            MessageState.MPCFindMaxDirection: self.receive_MPC_find_max_direction,
            MessageState.MoveNode: self.receive_move_node,
            MessageState.ClearCache: self.receive_clear_data,
        }
        while True:
            message = self.recv()
            state = message.state
            if state == MessageState.FirstStageEnd:
                return message.data
            handler[state](message.data)
            continue

    def second_stage(self):
        self.merge_vertex()
        self.first_loop = True

    def merge_vertex(self):
        """merge vertexs belong to the same community
        """
        new_community_vertices = {}
        new_vertex_vertex = {}
        new_local_vertexs = set()
        for cid, community in self.community_table.items():
            if not community.vertexs:
                continue
            new_vertex = Vertex(cid)
            is_cross_merge = False
            exist_local_vertex = False
            exist_no_local_vertex = False

            for vid in community.vertexs:
                new_vertex.merge(self.vertex_table[vid])
                if vid not in self.local_vertexs:
                    exist_no_local_vertex = True
            if new_vertex.nodes:
                exist_local_vertex = True

            # There are two standards for cross-border vertex:
            # 1. One of the sub-vertex is a cross-border vertex
            # 2. Contains both its own node and the other node
            new_vertex.is_cross_merge = new_vertex.is_cross_merge or (exist_no_local_vertex and exist_local_vertex)
            new_vertex.is_cross_merge = is_cross_merge
            new_community = Community(cid, {cid}, new_vertex.degree, is_ghost=new_vertex.is_ghost)
            new_community_vertices[cid] = new_community
            new_vertex_vertex[cid] = new_vertex

            # How to determine the ownership of a node, if one of the following
            # conditions exists, the node belongs to you:
            # 1. this vertext is not cross_merge
            # 2. this vertext is cross_merge and you are server
            if (not new_vertex.is_cross_merge and new_vertex.nodes) \
                    or (new_vertex.is_cross_merge and self.party_id == 0):
                new_local_vertexs.add(cid)

        G = Graph()
        for cid1 in new_community_vertices:
            for vid in self.community_table[cid1].vertexs:
                for cid2 in self.G_vertex_community.neighbors(vid):
                    if cid2 > cid1 and cid2 in new_community_vertices:
                        G.update_weight(cid1, cid2, self.G_vertex_community.weight(vid, cid2) + G.weight(cid1, cid2))
                        G.update_weight(cid2, cid1, self.G_vertex_community.weight(vid, cid2) + G.weight(cid2, cid1))

        self.community_table = new_community_vertices
        self.vertex_table = new_vertex_vertex
        self.G_vertex_vertex = copy.deepcopy(G)
        self.G_vertex_community = copy.deepcopy(G)
        self.local_vertexs = new_local_vertexs

    def get_partition(self):
        partition = []
        for vid, node in self.vertex_table.items():
            for v in node.nodes:
                partition.append([v, vid])
        return pd.DataFrame(partition, columns=["user_id", "cluster_id"])

    def MPC_find_max_direction(self, launch=True, move_flag=None, max_Q=None):
        self.logger.debug("Start MPC_find_max_direction")
        t1 = time.time()
        max_index, max_value = self.MPC_calculate_max_Q(launch)
        directions = self.cache.w_cids
        new_max_direction = -1

        if launch:
            max_index = int(max_index.flatten()[0])
            max_value = max_value.flatten()[0]
            max_val = 0
            if max_value > max_val:
                if directions[max_index] is not None:
                    state = MessageState.MPCEnd
                    data = None
                else:
                    state = MessageState.QueryDirection
                    data = max_index
                self.send(state, data)
                response = self.recv()

                if state == MessageState.QueryDirection:
                    new_max_direction = response.data
                else:
                    new_max_direction = directions[max_index]
                max_val = max_value
            else:
                self.send(MessageState.MPCEnd)
                self.recv()

            if move_flag and max_Q > max_val:
                new_max_direction = -1
        else:
            message = self.recv()
            if message.state == MessageState.MPCEnd:
                self.send(MessageState.OK)
            else:
                max_index = data["data"]
                self.send(MessageState.OK, directions[max_index])
        t2 = time.time()
        self.logger.debug(f"MPC_find_max_direction time cost: {t2-t1}")
        return new_max_direction

    def MPC_calculate_max_Q(self, launch):
        vid = self.cache.vid
        cid = self.cache.cid
        w_cids = self.cache.w_cids
        weight_u_c0, weight_u_c1, weight_c0, weight_c1, degree_u = self.prepare_Q_data(vid, cid, w_cids)
        self.logger.debug(f"MPC_find_max_direction data shape: {np.shape(weight_u_c0)}")
        if launch:
            party_id1 = self.party_id
            party_id2 = self.peer_party
        else:
            party_id1 = self.peer_party
            party_id2 = self.party_id

        weight_u_c0_priv = snp.array(weight_u_c0, party=party_id1) + snp.array(weight_u_c0, party=party_id2)
        weight_u_c1_priv = snp.array(weight_u_c1, party=party_id1) + snp.array(weight_u_c1, party=party_id2)
        weight_c0_priv = snp.array(weight_c0, party=party_id1) + snp.array(weight_c0, party=party_id2)
        degree_u_priv = snp.array(degree_u, party=party_id1) + snp.array(degree_u, party=party_id2)
        weight_c1_priv = snp.array(weight_c1, party=party_id1) + snp.array(weight_c1, party=party_id2)

        Q_rpiv = weight_u_c1_priv - weight_u_c0_priv + degree_u_priv * (weight_c0_priv - degree_u_priv -
                                                                        weight_c1_priv) * (1. / self.total_weight)

        max_index, max_value = snp.argmax_and_max(Q_rpiv, axis=0)
        max_index = max_index.reveal_to(party_id1)
        max_value = max_value.reveal_to(party_id1)
        return max_index, max_value

    def fit_server(self):
        iter_phase_I = 0
        iter_phase_II = 0

        state = TrainState.state0
        while True:
            if state == TrainState.state0:
                if iter_phase_I < self.max_move:
                    self.send(TrainState.state0)
                    move_flag1 = self.first_stage()
                    move_flag2 = self.listen_local_move()
                    self.logger.info(f"phase I: {iter_phase_I}")
                    if (move_flag1 or move_flag2):
                        iter_phase_I += 1
                        continue
                state = TrainState.state1

            if state == TrainState.state1:
                if iter_phase_I > 0 and iter_phase_II < self.max_merge:
                    self.send(TrainState.state1)
                    self.second_stage()
                    self.recv()
                    iter_phase_I = 0
                    iter_phase_II += 1
                    self.logger.info(f"phase II: {iter_phase_II}")
                    state = TrainState.state0
                    continue
                state = TrainState.state2
                break

        self.send(TrainState.state2)

    def fit_client(self):
        while True:
            state = self.recv().state
            if state == TrainState.state0:
                self.listen_local_move()
                self.first_stage()
                continue
            if state == TrainState.state1:
                self.second_stage()
                self.send(MessageState.OK)
                continue
            if state == TrainState.state2:
                break

    def transform(self, user_weights: dict, local_nodes: set) -> pd.DataFrame:
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        user_weights: dict.
            The weight of user-user weight.

        local_nodes : set.
            The local nodes of each party.

        Returns
        -------
        cluster: csv file
            The result of Leiden algorithm.
        """
        snp.set_vm(self._mpc_engine.engine)
        self.__init_attr(local_nodes, user_weights)
        self.sync_total_weight()
        if self.party_id == 0:
            self.fit_server()
        else:
            self.fit_client()
        partition = self.get_partition()
        return partition


class TrainState:
    state0 = "launch local move"
    state1 = "launch merge Vertex"
    state2 = "save partition results"


class MessageState:
    OK = "ok"
    SyncTotalWeight = "sync_total_graph_weight"
    SyncMoveNodeInfo = "sync_move_node_info"
    MPCFindMaxDirection = "mpc_find_max_direction"
    MoveNode = "move_node"
    ClearCache = "clear_cache"
    SyncQInfo = "sync_q_info"
    IfMoveMergeNode = "if_move_merge_node"
    EndMoveMergeNode = "end_move_merge_node"
    FirstStageEnd = "first_stage_end"
    MPCEnd = "mpc_end"
    QueryDirection = "query_direction"
