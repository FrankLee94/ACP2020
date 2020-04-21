# !usr/bin/env python
# -*- coding:utf-8 -*-

# this model is for hierarchy resource allocation in mec
# JialongLi 2020/03/07

import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt

REQ_NUM = 10000			# 请求的总数目
CPU_ONE = 150.0			# 本地节点CPU总量
CPU_TWO = 500.0			# 二级节点CPU总量
BD_WIDTH = 5000			# 5000M的带宽表示线宽为1

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
(-3, 10), (3, 10)]							# 2个数据中心




# 图形初始化
def graph_init():
	G = nx.Graph()
	G.add_nodes_from([i for i in range(18)])
	G.add_edges_from(route_edges)
	return G


# 算法初始化
# current_load：每个节点CPU的使用率
# vm_locate_index：存储请求的分类及具体位置，分类有local, neigh, DC，具体位置为本节点id
def initial():
	current_load = [0 for i in range(16)]		# 节点负载,对应每个节点的颜色
	vm_locate_index = {}						# 部署位置,具体位置，及使用的路径
	edge_width = {}								# 储存每条边的负载，对应线的宽度
	for item in route_edges:
		edge_width[item] = 0
		reverse_item = (item[1], item[0])
		edge_width[reverse_item] = 0
	return current_load, vm_locate_index, edge_width


# 某个请求接入，更新虚拟机所在节点的负载
def fill_current_load(current_load, ReqNo, vm_locate_index, cpu):
	if vm_locate_index[ReqNo][0] == 'DC':		# DC的CPU资源无限
		pass
	else:
		vm_locate = vm_locate_index[ReqNo][1]		# 虚拟机的具体位置
		current_load[vm_locate] += cpu


# 某个请求离开，更新虚拟机所在节点的负载
def rele_current_load(current_load, ReqNo, vm_locate_index, cpu):
	if vm_locate_index[ReqNo][0] == 'DC':			# DC的CPU资源无限
		pass
	else:
		vm_locate = vm_locate_index[ReqNo][1]		# 虚拟机的具体位置
		current_load[vm_locate] -= cpu


# 某个请求接入，更新路径负载
def fill_edge_width(edge_width, ReqNo, vm_locate_index, bandwidth):
	shortest_path = vm_locate_index[ReqNo][2]
	if len(shortest_path) == 0:
		pass
	else:
		for i in range(len(shortest_path) - 1):
			edge = (shortest_path[i], shortest_path[i+1])
			edge_width[edge] += bandwidth


# 某个请求离开，更新路径负载
def rele_edge_width(edge_width, ReqNo, vm_locate_index, bandwidth):
	shortest_path = vm_locate_index[ReqNo][2]
	if len(shortest_path) == 0:
		pass
	else:
		for i in range(len(shortest_path) - 1):
			edge = (shortest_path[i], shortest_path[i+1])
			edge_width[edge] -= bandwidth


# 状态发生改变，重新计算画图参数
def state_change(current_load, edge_width, lws, ncolors):
	for i in range(len(current_load)):
		if i % 4 == 0:		# neigh节点
			ncolors[i] = str(round(current_load[i] / CPU_TWO, 3))
		else:
			ncolors[i] = str(round(current_load[i] / CPU_ONE, 3))
	for i in range(len(route_edges)):		# lws的参数和route_edges是按顺序一一对应的
		item = (route_edges[i][0], route_edges[i][1])
		reverse_item = (route_edges[i][1], route_edges[i][0])		# 某些边有双向流量
		lws[i] = round((edge_width[item] + edge_width[reverse_item]) / BD_WIDTH, 2)


# 输入对应参数，画出图形
def graph_draw(G, idx, status, ncolors, ecolors, lws):
	plt.clf()
	nx.draw(G, pos, with_labels=False, node_color=ncolors, node_shape="o", node_size=500, width=lws, edge_color=ecolors)
	plt.xlim(-12, 12)			# 设置首界面X轴坐标范围
	plt.ylim(-12, 12)		# 设置首界面Y轴坐标范围
	fig_path = './state/env_' + str(idx) + '_' + status + '.png'
	plt.savefig(fig_path)


# reward函数，根据当前的状态及动作给出相应的reward
def reward(action, area_id, node_id, current_load, cpu, bandwidth, delay_sen):
	reward = 0
	if action == 'local':
		node = area_id * 4 + node_id + 1
		if current_load[node] + cpu <= CPU_ONE:			# 成功接入本地，奖励50
			reward += 50
		else:
			reward -= 1000	# 没有充足资源却选择接入，奖励-1000
	elif action == 'neigh':
		neigh_node = area_id * 4
		if current_load[neigh_node] + cpu <= CPU_TWO:	# 成功接入neigh，奖励50
			reward += 50
		else:
			reward -= 1000	# 没有充足资源却选择接入，奖励-1000
		if delay_sen == 0:
			pass
		else:
			reward -= 20		# 延时敏感，奖励-20
		reward -= (int(bandwidth/1000) + 1) * 2		# 占用带宽，带宽越大，惩罚越大，2-6之间
	else:		# 数据中心
		reward += 50		# 成功接入数据中心，奖励50
		if delay_sen == 0:
			pass
		else:
			reward -= 40		# 延时敏感，奖励-40
		reward -= (int(bandwidth/1000) + 1) * 3		# 占用带宽，带宽越大，惩罚越大，3-9之间
	return reward


# *****************************对比算法1：先来先服务*******************************


# 为某个请求布置一个虚拟机
# locate_flag有三种情况，分别是'local', 'neigh', 'DC'
def find_locate_fcfs(G, current_load, area_id, node_id, cpu):
	node = area_id * 4 + node_id + 1
	shortest_path = []
	# 尝试使用local节点
	if current_load[node] + cpu <= CPU_ONE:
		locate_flag = 'local'
		vm_locate = node
		return locate_flag, vm_locate, shortest_path
	# 尝试使用邻居节点
	neigh_node = area_id * 4
	if current_load[neigh_node] + cpu <= CPU_TWO:
		locate_flag = 'neigh'
		vm_locate = neigh_node
		shortest_path = random.choice([p for p in nx.all_shortest_paths(G, source=node, target=vm_locate)])
		return locate_flag, vm_locate, shortest_path
	# 使用数据中心
	locate_flag = 'DC'
	vm_locate = random.choice([16, 17])		# 从2个数据中心中随机取一个
	shortest_path = random.choice([p for p in nx.all_shortest_paths(G, source=node, target=vm_locate)])
	return locate_flag, vm_locate, shortest_path


# 先来先服务模型：先使用本节点，接着使用临近节点，最后使用DC。不区分延时敏感与否
def fcfs(G, current_load, vm_locate_index, edge_width, traffic_file_sort_path):
	lws = [0.0 for i in range(len(route_edges))]			# 边的宽度，随负载增加
	ncolors = ['0.0' for i in range(18)]					# 节点的颜色，随负载增加，0~1的范围
	ecolors = ['0' for i in range(len(route_edges))]		# 边的颜色，不变
	df = pd.read_excel(traffic_file_sort_path)
	total_reward = 0
	for index, row in df.iterrows(): 
		# print(index)
		ReqNo = row['ReqNo']
		if row['status'] == 'arrive':
			locate_flag, vm_locate, shortest_path = find_locate_fcfs(G, current_load, 
				row['area_id'], row['node_id'], row['cpu'])
			vm_locate_index[ReqNo] = [locate_flag, vm_locate, shortest_path]		# 位置分类 + 具体位置
			
			total_reward += reward(locate_flag, row['area_id'], row['node_id'], 
				current_load, row['cpu'], row['bandwidth'], row['delay_sen'])
			fill_current_load(current_load, ReqNo, vm_locate_index, row['cpu'])
			fill_edge_width(edge_width, ReqNo, vm_locate_index, row['bandwidth'])
			state_change(current_load, edge_width, lws, ncolors)
			# if index < 20 or index > 19950:
			# graph_draw(G, index, 'arrive', ncolors, ecolors, lws)
			# print(edge_width)
		else:		# 'leave'
			rele_current_load(current_load, ReqNo, vm_locate_index, row['cpu'])
			rele_edge_width(edge_width, ReqNo, vm_locate_index, row['bandwidth'])
			state_change(current_load, edge_width, lws, ncolors)
			if index > 19998:
				ncolors = ['0.0' for i in range(18)]
				graph_draw(G, index, 'leave', ncolors, ecolors, lws)
	print('total reward')
	print(total_reward)


if __name__ == '__main__':
	traffic_file_sort_ph = './traffic_data/traffic_sort_20.xlsx'
	G_topo = graph_init()
	curr_load, vm_locate_idx, e_width = initial()
	fcfs(G_topo, curr_load, vm_locate_idx, e_width, traffic_file_sort_ph)
