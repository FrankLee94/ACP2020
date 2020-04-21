# !usr/bin/env python
# -*- coding:utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt


# 状态初始化，没有进行任何操作时候的图像
def stateInit():
	G = nx.Graph()
	G.add_nodes_from([i for i in range(18)])
	# 18个节点的坐标
	pos = [
	(-3, 0), (-5, 1), (-7, 0), (-5, -1),		# 区域1
	(0, 3), (1, 5), (0, 7), (-1, 5),			# 区域2
	(0, -3), (1, -5), (0, -7), (-1, -5),		# 区域3
	(3, 0), (5, 1), (7, 0), (5, -1),			# 区域4
	(-3, 10), (3, 10)]							# 2个数据中心
	# 边的连接关系
	route_edges = [
	(0, 1), (1, 2), (2, 3), (3, 0),						# 区域1
	(4, 5), (5, 6), (6, 7), (7, 4),						# 区域2
	(8, 9), (9, 10), (10, 11), (11, 8),					# 区域3
	(12, 13), (13, 14), (14, 15), (15, 12),				# 区域4
	(0, 4), (4, 12), (12, 8), (8, 0), (0, 16), (12, 17)]# 环及数据中心

	G.add_edges_from(route_edges)
	lws = [0.0 for i in range(len(route_edges))]			# 边的宽度，随负载增加
	ecolors = ['0' for i in range(len(route_edges))]	# 边的颜色，不变
	ncolors = ['0.1' for i in range(18)]				# 节点的颜色，随负载增加，0~1的范围
	nx.draw(G, pos, with_labels=False, node_color=ncolors, node_shape="o", node_size=500, width=lws, edge_color=ecolors)
	
	plt.xlim(-8, 8)			# 设置首界面X轴坐标范围
	plt.ylim(-8, 12)		# 设置首界面Y轴坐标范围
	plt.savefig("env_init.png")


if __name__ == '__main__':
	stateInit()
