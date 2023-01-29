import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F

class ClusterNode(object):
    def __init__(self, vec, left=None, right=None, distance=-1, id=None, count=1):
        """
        :param vec: 保存两个数据聚类后形成新的中心
         :param left: 左节点
         :param right:  右节点
         :param distance: 两个节点的距离
         :param id: 用来标记哪些节点是计算过的
         :param count: 这个节点的叶子节点个数
        """
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.id = id
        self.count = count

class Hierarchical(object):
    def __init__(self, k = 1):
        assert k > 0
        self.k = k
        self.labels = None

    def fit(self, x):
        nodes = [ClusterNode(vec=v, id=i) for i,v in enumerate(x)]
        distances = {}
        point_num, future_num = np.shape(x)  # 特征的维度
        self.labels = [ -1 ] * point_num
        currentclustid = -1
        while len(nodes) > self.k:
            min_dist = math.inf
            nodes_len = len(nodes)
            closest_part = None  # 表示最相似的两个聚类
            for i in range(nodes_len - 1):
                for j in range(i + 1, nodes_len):
                    # 为了不重复计算距离，保存在字典内
                    d_key = (nodes[i].id, nodes[j].id)
                    if d_key not in distances:
                        distances[d_key] = cos_dist(nodes[i].vec, nodes[j].vec)
                    d = distances[d_key]
                    if d < min_dist:
                        min_dist = d
                        closest_part = (i, j)
            # 合并两个聚类
            part1, part2 = closest_part
            node1, node2 = nodes[part1], nodes[part2]
            new_vec = [ (node1.vec[i] * node1.count + node2.vec[i] * node2.count ) / (node1.count + node2.count)
                        for i in range(future_num)]  ##??
            new_node = ClusterNode(vec=new_vec,
                                   left=node1,
                                   right=node2,
                                   distance=min_dist,
                                   id=currentclustid,
                                   count=node1.count + node2.count)
            currentclustid -= 1
            del nodes[part2], nodes[part1]  # 一定要先del索引较大的
            nodes.append(new_node)
        self.nodes = nodes
        self.calc_label()

    def calc_label(self):
        """
        调取聚类的结果
        """
        for i, node in enumerate(self.nodes):
            # 将节点的所有叶子节点都分类
            self.leaf_traversal(node, i)

    def leaf_traversal(self, node: ClusterNode, label):
        """
        递归遍历叶子节点
        """
        if node.left == None and node.right == None:
            self.labels[node.id] = label
        if node.left:
            self.leaf_traversal(node.left, label)
        if node.right:
            self.leaf_traversal(node.right, label)
# k-means
class node():
    def __init__(self, id):
        self.cls = -1

    def label(self, cls):
        self.cls = cls

class center():
    def __init__(self, k, dim):
        self._init_coord(k, dim)

    def _init_coord(self, k, dim):
        self.coord = []
        for i in range(k):
            self.coord.append(torch.randn([1024], device='cuda'))

class cluster():
    def __init__(self, feats, k):
        self.feats = feats # tensor[n x 1024] 样本特征向量
        self.dim = feats.size(1) # dimension of samples' features: 1024
        self.k = k
        self.centrods = center(self.k, self.dim) # list[tensor(1024)]: 质心特征向量

        self.nodes = [node(i) for i in range(len(self.feats))] # list[]每个样本的所属簇的索引号
        self.max_iters = 5 # 最大迭代周期

    def _init_index(self):
        self.index = []
        for i in enumerate(self.feats):
            self.index.append(-1)

    def forward(self):
        # label
        for i, feat in enumerate (self.feats):
            flag = -1
            min_dis = 2
            #print(len(self.centrods.coord))
            for j, centrod in enumerate(self.centrods.coord):
                dis = self.cosin_dist(feat, centrod)
                if dis < min_dis:
                    flag = j
                    min_dis = dis
            self.nodes[i].label(flag)

        # update centroids' coordnates
        cent_temp = torch.zeros([self.k, self.dim], device='cuda')
        cent_num = np.zeros([self.k])
        for i, feat in enumerate(self.feats):
            idx = self.nodes[i].cls
            cent_temp[idx] = cent_temp[idx] + feat
            cent_num[idx] += 1
	
        for i, local in enumerate(self.centrods.coord):
            local = cent_temp[i]/cent_num[i]

    def run(self):
        iter = 1
        while iter<self.max_iters:
            self.forward()
            iter += 1
        self.group()

    def cosin_dist(self, vec1, vec2):
        """
        :param vec1: tensor[1x1024]
        :param vec2: tensor[1x1024]
        :return: cosin distance
        """
        def norm(vec):
            dot = vec*vec
            return torch.sqrt(dot.sum())

        dot = vec1 * vec2
        dist = 1 - ( dot.sum() / (norm(vec1) * norm(vec2)))
        return dist

    def group(self):
        self.cls_group = []
        for node in self.nodes:
            self.cls_group.append(node.cls)
        for i,feat in enumerate(self.centrods.coord):
            self.centrods.coord[i] = feat.unsqueeze(0)

def main():
    feats = torch.rand([321, 1024], device='cuda')
    kmeans = cluster(feats, k=10)
    kmeans.run()
    print(kmeans.centrods.coord[0].size())
    feat = torch.cat(kmeans.centrods.coord, dim=0)
    print(feat.size())

def complete(feats, cls_score):
    cls_score = F.softmax(cls_score, dim=-1)
    print(cls_score)
    top_idx = torch.argmax(cls_score[:,0], dim=0)
    print(top_idx)
    add_num = 10 - len(feats)
    add_feats = []
    for i in range(add_num):
        add_feats.append(feats[top_idx].unsqueeze(0))
    add_feats = torch.cat(add_feats, dim=0)
    feats = torch.cat([feats, add_feats], dim=0)
    print(feats.size())

if __name__=='__main__':
    #main()
    #feats = torch.rand([5, 1024], device='cuda')
    #cls_score = torch.randn([5, 2], device='cuda')
    #complete(feats, cls_score)
    a = [1, 2, 3]
    idx = torch.argmax(a)
    print(idx)