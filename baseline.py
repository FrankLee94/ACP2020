# !usr/bin/env python
# -*- coding:utf-8 -*-

# this model is for hierarchy resource allocation in mec
# JialongLi 2020/03/07

import networkx as nx
import traffic


class Baselines:
	def __init__(self):
		self.REQ_NUM = 10000  # 请求的总数目
		self.CPU_ONE = 150.0  # 本地节点CPU总量
		self.CPU_TWO = 500.0  # 二级节点CPU总量
		self.RAM_ONE = 150.0  # 本地节点RAM总量
		self.RAM_TWO = 500.0  # 二级节点RAM总量
		# 边的连接关系，16条边
		self.route_edges = [
			(0, 1), (1, 2), (3, 0),					# 区域1
			(4, 5), (5, 6), (7, 4),					# 区域2
			(8, 9), (9, 10), (11, 8),				# 区域3
			(12, 13), (13, 14), (15, 12),			# 区域4
			(0, 4), (12, 8), (0, 16), (12, 17)]		# 环及数据中心
		# 18个节点的坐标
		self.pos = [
			(-3, 0), (-5, 1), (-7, 0), (-5, -1),		# 区域1
			(0, 3), (1, 5), (0, 7), (-1, 5),			# 区域2
			(0, -3), (1, -5), (0, -7), (-1, -5),		# 区域3
			(3, 0), (5, 1), (7, 0), (5, -1),			# 区域4
			(-3, 10), (3, 10)]							# 2个数据中心
		self.df = None 				# dataframe, 读取的excel
		self.idx = None 			# 序号
		self.G = None 				# 用来做拓扑
		self.vm_locate_idx = None 	# 用来画图
		self.curr_load = None
		self.e_width = None
		self.max_bd = None
		self.actions = ['local', 'neigh', 'DC']

	# 图形初始化
	def graph_init(self):
		G = nx.Graph()
		G.add_nodes_from([i for i in range(18)])
		G.add_edges_from(self.route_edges)
		return G

	# 初始化各条边的最大带宽，注意双向带宽合为单向带宽
	def edge_init(self):
		self.max_bd = {}
		self.e_width = {}
		for item in self.route_edges:
			reverse_item = (item[1], item[0])
			self.e_width[item] = 0					# 储存每条边的负载，对应线的宽度
			self.e_width[reverse_item] = 0
			if item[0] > 15 or item[1] > 15:		# 数据中心链路带宽50000M
				self.max_bd[item] = 40000
				self.max_bd[reverse_item] = 40000
			elif item[0] % 4 == 0 and item[1] % 4 == 0:
				self.max_bd[item] = 40000		# neigh之间链路带宽40000M
				self.max_bd[reverse_item] = 40000
			else:
				self.max_bd[item] = 40000		# local之间链路带宽40000M
				self.max_bd[reverse_item] = 40000

	# 算法初始化, current_load：每个节点CPU的使用率
	# vm_locate_idx：存储请求的分类及具体位置，分类有local, neigh, DC，具体位置为本节点id
	def initial(self):
		# 节点负载,对应每个节点的颜色, [0, 0]分别表示cpu 和 ram
		self.curr_load = [[0, 0] for i in range(16)]
		self.vm_locate_idx = {}						# 部署位置,具体位置，及使用的路径
		self.G = self.graph_init()
		self.edge_init()
		self.df = traffic.get_new_df()				# 读取新的随机事件
		self.idx = 0

	# 某个请求接入，更新虚拟机所在节点的负载
	def fill_current_load(self, row):
		if self.vm_locate_idx[row['ReqNo']][0] == 'DC':  # DC的CPU资源无限
			pass
		else:
			vm_locate = self.vm_locate_idx[row['ReqNo']][1]  # 虚拟机的具体位置
			self.curr_load[vm_locate][0] += row['cpu']
			self.curr_load[vm_locate][1] += row['ram']

	# 某个请求离开，更新虚拟机所在节点的负载
	def rele_current_load(self, row):
		if self.vm_locate_idx[row['ReqNo']][0] == 'DC':  # DC的CPU资源无限
			pass
		else:
			vm_locate = self.vm_locate_idx[row['ReqNo']][1]  # 虚拟机的具体位置
			self.curr_load[vm_locate][0] -= row['cpu']
			self.curr_load[vm_locate][1] -= row['ram']

	# 某个请求接入，更新路径负载
	def fill_edge_width(self, row):
		shortest_path = self.vm_locate_idx[row['ReqNo']][2]
		if len(shortest_path) == 0:
			pass
		else:
			for i in range(len(shortest_path) - 1):
				edge = (shortest_path[i], shortest_path[i + 1])
				self.e_width[edge] += row['bandwidth']

	# 某个请求离开，更新路径负载
	def rele_edge_width(self, row):
		shortest_path = self.vm_locate_idx[row['ReqNo']][2]
		if len(shortest_path) == 0:
			pass
		else:
			for i in range(len(shortest_path) - 1):
				edge = (shortest_path[i], shortest_path[i + 1])
				self.e_width[edge] -= row['bandwidth']

	# 判断链路上是否有足够带宽接入
	def is_enough_bd(self, row, shortest_path):
		is_enough = True
		for i in range(len(shortest_path) - 1):			# 逐段链路检查
			edge = (shortest_path[i], shortest_path[i+1])
			if self.e_width[edge] + row['bandwidth'] > self.max_bd[edge]:
				is_enough = False
				break
		return is_enough

	# 先使用local，然后是neigh，最后是DC
	def local_first(self, row):
		node = row['area_id'] * 4 + row['node_id'] + 1
		# 尝试使用local节点
		shortest_path = []
		if self.curr_load[node][0] + row['cpu'] <= self.CPU_ONE and \
			self.curr_load[node][1] + row['ram'] <= self.RAM_ONE:
			locate_flag = 'local'
			vm_locate = node
			self.vm_locate_idx[row['ReqNo']] = [locate_flag, vm_locate, shortest_path]
			return
		
		# 尝试使用邻居节点
		neigh_node = row['area_id'] * 4
		shortest_path = nx.shortest_path(self.G, source=node, target=neigh_node)
		if self.curr_load[neigh_node][0] + row['cpu'] <= self.CPU_TWO and \
			self.curr_load[neigh_node][1] + row['ram'] <= self.RAM_TWO and \
			self.is_enough_bd(row, shortest_path):
			locate_flag = 'neigh'
			vm_locate = neigh_node
			self.vm_locate_idx[row['ReqNo']] = [locate_flag, vm_locate, shortest_path]
			return
		
		# 使用数据中心
		if row['area_id'] < 2:			# 区域0和1使用数据中心‘16’
			DC_node = 16
		else:
			DC_node = 17
		shortest_path = nx.shortest_path(self.G, source=node, target=DC_node)
		if self.is_enough_bd(row, shortest_path):
			locate_flag = 'DC'
			vm_locate = DC_node
			self.vm_locate_idx[row['ReqNo']] = [locate_flag, vm_locate, shortest_path]
			return
		
		# 阻塞
		locate_flag = 'block'
		vm_locate = -1
		shortest_path = []
		self.vm_locate_idx[row['ReqNo']] = [locate_flag, vm_locate, shortest_path]

	# 先使用DC，接着使用neigh，最后是local
	def dc_first(self, row):
		node = row['area_id'] * 4 + row['node_id'] + 1
		# 使用数据中心
		if row['area_id'] < 2:			# 区域0和1使用数据中心‘16’
			DC_node = 16
		else:
			DC_node = 17
		shortest_path = nx.shortest_path(self.G, source=node, target=DC_node)
		if self.is_enough_bd(row, shortest_path):
			locate_flag = 'DC'
			vm_locate = DC_node
			self.vm_locate_idx[row['ReqNo']] = [locate_flag, vm_locate, shortest_path]
			return
		
		# 尝试使用邻居节点
		neigh_node = row['area_id'] * 4
		shortest_path = nx.shortest_path(self.G, source=node, target=neigh_node)
		if self.curr_load[neigh_node][0] + row['cpu'] <= self.CPU_TWO and \
			self.curr_load[neigh_node][1] + row['ram'] <= self.RAM_TWO and \
			self.is_enough_bd(row, shortest_path):
			locate_flag = 'neigh'
			vm_locate = neigh_node
			self.vm_locate_idx[row['ReqNo']] = [locate_flag, vm_locate, shortest_path]
			return
		
		# 尝试使用local节点
		shortest_path = []
		if self.curr_load[node][0] + row['cpu'] <= self.CPU_ONE and \
			self.curr_load[node][1] + row['ram'] <= self.RAM_ONE:
			locate_flag = 'local'
			vm_locate = node
			self.vm_locate_idx[row['ReqNo']] = [locate_flag, vm_locate, shortest_path]
			return
		
		# 阻塞
		locate_flag = 'block'
		vm_locate = -1
		shortest_path = []
		self.vm_locate_idx[row['ReqNo']] = [locate_flag, vm_locate, shortest_path]

	# reward函数，根据当前的状态及动作给出相应的reward
	def get_reward(self, row):
		reward = 0
		action = self.vm_locate_idx[row['ReqNo']][0]
		if action == 'local':
			reward += 800
		elif action == 'neigh':
			reward += 800
			if row['delay_sen'] == 0:
				pass
			else:
				reward -= 200  # 延时敏感，奖励-200
			reward -= (int(row['bandwidth'] / 1000) + 1) * 100  # 占用带宽，带宽越大，惩罚越大，100-300之间
		elif action == 'DC':		# 数据中心
			reward += 800  # 成功接入数据中心，奖励50
			if row['delay_sen'] == 0:
				pass
			else:
				reward -= 400		# 延时敏感，奖励-400
			reward -= (int(row['bandwidth'] / 1000) + 1) * 200  # 占用带宽，带宽越大，惩罚越大，200-600之间
		else:			# block
			reward -= 1000
		return reward

	# 统计信息
	def stastics(self):
		block_num = 0
		traff_DC = 0		# 到达DC的总流量
		traff_neigh = 0		# 到达neigh的总流量
		laten_sen = 0		# 延时敏感业务总延时
		laten_uns = 0		# 延时不敏感业务总延时
		sen_count = 0		# 延时敏感接入业务的总数量
		uns_count = 0		# 延时不敏感接入业务的总数量
		location_count = {'block': 0, 'local': 0, 'neigh': 0, 'DC': 0}
		for index, row in self.df.iterrows():
			if row['status'] == 'arrive':
				if self.vm_locate_idx[row['ReqNo']][0] == 'block':
					block_num += 1
				elif self.vm_locate_idx[row['ReqNo']][0] == 'DC':
					traff_DC += row['bandwidth']
					if row['delay_sen'] == 1:		# 延时敏感业务，DC延时100
						laten_sen += 100
						sen_count += 1
					else:
						laten_uns += 100
						uns_count += 1
				elif self.vm_locate_idx[row['ReqNo']][0] == 'neigh':
					traff_neigh += row['bandwidth']
					if row['delay_sen'] == 1:		# 延时敏感业务，neigh延时100
						laten_sen += 30
						sen_count += 1
					else:
						laten_uns += 30
						uns_count += 1
				else:		# 'local'
					if row['delay_sen'] == 1:		# 延时敏感业务，local延时0
						laten_sen += 0
						sen_count += 1
					else:
						laten_uns += 0
						uns_count += 1
			else:
				pass
		for key, value in self.vm_locate_idx.items():
			location_count[value[0]] += 1
		for key, value in location_count.items():
			print(key, value)
		print('blocking rate:  ' + str(round(block_num / self.REQ_NUM * 100, 2)) + '%')
		print('DC traffic:  ' + str(round(traff_DC / 1000, 0)) + 'Gb')
		print('neigh traffic:  ' + str(round(traff_neigh / 1000, 0)) + 'Gb')
		print('DC + neigh traffic:  ' + str(round((traff_DC + traff_neigh) / 1000, 0)) + 'Gb')
		print('average latency for sensitive:  ' + str(round(laten_sen / sen_count, 2)))
		print('average latency for unsensitive:  ' + str(round(laten_uns / uns_count, 2)))
		print('average latency:  ' + str(round((laten_sen + laten_uns) / (sen_count + uns_count), 2)))

# *****************************对比算法1：先来先服务*******************************

	# 先来先服务模型：先使用本节点，接着使用临近节点，最后使用DC。不区分延时敏感与否
	def fcfs(self):
		self.initial()
		total_reward = 0
		for index, row in self.df.iterrows(): 
			if row['status'] == 'arrive':
				self.local_first(row)
				reward = self.get_reward(row)
				total_reward += reward
				self.fill_current_load(row)
				self.fill_edge_width(row)
			else:		# 'leave'
				self.rele_current_load(row)
				self.rele_edge_width(row)
		print('firs-come first-serve: ')
		print('total reward: ' + str(round(total_reward / 1000, 0)))
		self.stastics()

# *****************************对比算法2：延时敏感优先*******************************

	# 延时敏感业务：先使用local，然后是neigh，最后是DC
	# 延时不敏感业务：先使用DC，接着使用neigh，最后是local
	def dsrf(self):
		self.initial()
		total_reward = 0
		for index, row in self.df.iterrows(): 
			if row['status'] == 'arrive':
				if row['delay_sen'] == 1:
					self.local_first(row)		# 延时敏感优先使用local
				else:
					self.dc_first(row)
				reward = self.get_reward(row)
				total_reward += reward
				self.fill_current_load(row)
				self.fill_edge_width(row)
			else:		# 'leave'
				self.rele_current_load(row)
				self.rele_edge_width(row)
		print('\n')
		print('delay-sensitive request first: ')
		print('total reward: ' + str(round(total_reward / 1000, 0)))
		self.stastics()

# *****************************对比算法3：大带宽优先*******************************

	# 大带宽业务：> 2000M为大带宽业务，优先使用本地，
	# 小带宽业务优先使用数据中心
	def hbdf(self):
		self.initial()
		total_reward = 0
		for index, row in self.df.iterrows(): 
			if row['status'] == 'arrive':
				if row['bandwidth'] > 2000:
					self.local_first(row)		# 大带宽优先使用local
				else:
					self.dc_first(row)
				reward = self.get_reward(row)
				total_reward += reward
				self.fill_current_load(row)
				self.fill_edge_width(row)
			else:		# 'leave'
				self.rele_current_load(row)
				self.rele_edge_width(row)
		print('\n')
		print('huge bandwidth first: ')
		print('total reward: ' + str(round(total_reward / 1000, 0)))
		self.stastics()

# *****************************对比算法4：计算不密集优先*******************************

	# 对于计算密集任务，优先使用数据中心
	# 计算密集：cpu 或者 ram 大于25即为计算密集任务
	def curf(self):
		self.initial()
		total_reward = 0
		for index, row in self.df.iterrows(): 
			if row['status'] == 'arrive':
				if row['cpu'] > 25 or row['ram'] > 25:
					self.dc_first(row)		# 计算密集优先使用DC
				else:
					self.local_first(row)
				reward = self.get_reward(row)
				total_reward += reward
				self.fill_current_load(row)
				self.fill_edge_width(row)
			else:		# 'leave'
				self.rele_current_load(row)
				self.rele_edge_width(row)
		print('\n')
		print('computing-unintensive request first: ')
		print('total reward: ' + str(round(total_reward / 1000, 0)))
		self.stastics()


# main函数
if __name__ == '__main__':
	for i in range(10):
		print('num: ' + str(i+1))
		baseline_fcfs = Baselines()
		baseline_fcfs.fcfs()
		print('num: ' + str(i+1))
		baseline_dsrf = Baselines()
		baseline_dsrf.dsrf()
		print('num: ' + str(i+1))
		baseline_hbdf = Baselines()
		baseline_hbdf.hbdf()
		print('num: ' + str(i+1))
		baseline_curf = Baselines()
		baseline_curf.curf()
