#!usr/bin/env python
#-*- coding:utf-8 -*-

# this model is for test
# JialongLi 2020/03/13

from itertools import islice
import networkx as nx
import matplotlib.pyplot as plt
import xlrd
import copy
import random
from matplotlib import style

# 边的连接关系
route_edges = [
(0, 1), (1, 2), (2, 3), (3, 0),						# 区域1
(4, 5), (5, 6), (6, 7), (7, 4),						# 区域2
(8, 9), (9, 10), (10, 11), (11, 8),					# 区域3
(12, 13), (13, 14), (14, 15), (15, 12),				# 区域4
(0, 4), (4, 12), (12, 8), (8, 0), (0, 16), (12, 17)]		# 环及数据中心

# 18个节点的坐标
pos = [
(-3, 0), (-5, 1), (-7, 0), (-5, -1),		# 区域1
(0, 3), (1, 5), (0, 7), (-1, 5),			# 区域2
(0, -3), (1, -5), (0, -7), (-1, -5),		# 区域3
(3, 0), (5, 1), (7, 0), (5, -1),			# 区域4
(-3, 10), (3, 10)]							# 2个数据中心		# 返回一个无向图


def k_shortest_paths(G, source, target, k, weight=None):
	return list(islice(nx.shortest_simple_paths(G, source, target,weight=weight), k))

def graph_init():
	lws = [2.0 for i in range(len(route_edges))]			# 边的宽度，随负载增加
	ncolors = ['1' for i in range(18)]					# 节点的颜色，随负载增加，0~1的范围
	ecolors = ['1' for i in range(len(route_edges))]		# 边的颜色，不变
	return lws, ncolors, ecolors

if __name__ == '__main__':
	G_graph = nx.Graph()
	G_graph.add_nodes_from([i for i in range(18)])
	G_graph.add_edges_from(route_edges)

	lws, ncolors, ecolors  = graph_init()

	plt.clf()
	nx.draw(G_graph, pos, with_labels=False, node_color=ncolors, node_shape="o",
			node_size=500, width=lws, edge_color=ecolors)
	plt.xlim(-8, 8)		# 设置首界面X轴坐标范围
	plt.ylim(-8, 12)	# 设置首界面Y轴坐标范围
	#plt.style.use('dark_background')
	plt.style.use('fivethirtyeight')

	fig_path = './state/env_' + str(00) + '.png'
	plt.savefig(fig_path)

