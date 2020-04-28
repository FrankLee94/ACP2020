import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib import style
import cv2
import io
import numpy as np
import PIL
import reinforce.network.traffic as traffic
import traffic


class Environment:
	def __init__(
			self
	):
		self.action_num = 3
		self.REQ_NUM = 10000  # 请求的总数目
		self.CPU_ONE = 150.0  # 本地节点CPU总量
		self.CPU_TWO = 500.0  # 二级节点CPU总量
		self.RAM_ONE = 150.0  # 本地节点RAM总量
		self.RAM_TWO = 500.0  # 二级节点RAM总量
		self.BD_WIDTH = 5000  # 5000M的带宽表示线宽为1
		# 边的连接关系，16条边
		self.route_edges = [
			(0, 1), (1, 2), (3, 0),					# 区域1
			(4, 5), (5, 6), (7, 4),					# 区域2
			(8, 9), (9, 10), (11, 8),				# 区域3
			(12, 13), (13, 14), (15, 12),			# 区域4
			(0, 4), (12, 8), (0, 16), (12, 17)]		# 环及数据中心
		# 用来画图的边的关系，16+12条边
		self.graph_edges = [
			(0, 1), (1, 2), (3, 0),					# 区域1
			(4, 5), (5, 6), (7, 4),					# 区域2
			(8, 9), (9, 10), (11, 8),				# 区域3
			(12, 13), (13, 14), (15, 12),			# 区域4
			(0, 4), (12, 8), (0, 16), (12, 17),		# 环及数据中心
			(18, 30), (19, 31), (20, 32), (21, 33), (22, 34), (23, 35),     # 描绘请求
			(24, 36), (25, 37), (26, 38), (27, 39), (28, 40), (29, 41)]     # 描绘请求
		# 18 + 24个节点的坐标
		self.pos = [
			(-3, 0), (-5, 1), (-7, 0), (-5, -1),		# 区域1
			(0, 3), (1, 5), (0, 7), (-1, 5),			# 区域2
			(0, -3), (1, -5), (0, -7), (-1, -5),		# 区域3
			(3, 0), (5, 1), (7, 0), (5, -1),			# 区域4
			(-3, 10), (3, 10),							# 2个数据中心
			(-5, -16), (-7, -17), (-5, -18), (1, -12), (0, -10), (-1, -12),		# 区域1和2的向下平移
			(1, -22), (0, -24), (-1, -22), (5, -16), (7, -17), (5, -18),		# 区域3和4的向下平移
			(-4, -16), (-6, -17), (-4, -18), (2, -12), (1, -10), (0, -12),		# 虚拟节点
			(2, -22), (1, -24), (0, -22), (6, -16), (8, -17), (6, -18)]			# 虚拟节点

		self.df = None 				# dataframe, 读取的excel
		self.idx = None 			# 序号
		self.G = None 				# 用来做拓扑
		self.G_graph = None 		# 用来画图
		self.vm_locate_idx = None 	# dict, ['local', 2, [2, 3]]，某请求的接入类别，节点，路径
		self.max_bd = None 			# 边的最大带宽
		self.curr_load = None 		# 每个节点的当前负载
		self.e_width = None 		# 每条边的当前负载
		self.ncolors = None 		# 节点颜色
		self.ecolors = None 		# 边颜色
		self.nsize = None 			# 节点大小
		self.esize = None 			# 边大小
		self.actions = ['local', 'neigh', 'DC']

	# 拓扑初始化，返回一个无向图
	def topo_init(self):
		G = nx.Graph()
		G.add_nodes_from([i for i in range(18)])
		G.add_edges_from(self.route_edges)
		self.G = G

	# 观测到的图像初始化
	# 黑色背景，初始节点为黑色，表示负载为0，边初始为白色1
	def graph_init(self):
		G_graph = nx.Graph()
		G_graph.add_nodes_from([i for i in range(42)])
		G_graph.add_edges_from(self.graph_edges)
		self.G_graph = G_graph
		# 节点的颜色，初始为黑色0，随cpu负载增加，0~1的范围
		self.ncolors = ['0.0' for i in range(42)]
		# 边的颜色，16+12条边。前16条边恒定为白色，其宽度表示负载大小
		# 后12条边，白色表示延时敏感，半灰色表示延时不敏感
		self.ecolors = ['1' for i in range(len(self.graph_edges))]
		self.nsize = [0 for i in range(42)]							# 节点的大小
		self.esize = [0.0 for i in range(len(self.graph_edges))]		# 边的宽度，随负载增加

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

	# 算法初始化
	# curr_load：每个节点CPU的使用率
	# vm_locate_idx：存储请求的分类及具体位置，分类有local, neigh, DC，具体位置为本节点id
	def initial(self):
		# 节点负载,对应每个节点的颜色, [0, 0]分别表示cpu 和 ram
		self.curr_load = [[0, 0] for i in range(16)]
		self.vm_locate_idx = {}					# 部署位置,具体位置，及使用的路径
		self.e_width = {}						# 储存每条边的负载，对应线的宽度
		self.idx = 0
		self.df = traffic.get_new_df()			# 读取随机事件
		self.topo_init()						# 拓扑初始化
		self.graph_init()						# 刻画状态的变量
		self.edge_init()

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

	# 重置请求那部分的画图参数,原始有18个节点，16条边，多了24个节点，12条边
	def reset_add_req(self):
		for i in range(24):		# 多出来24个节点
			self.ncolors[i+18] = '0'		# 黑色节点
		for i in range(12):		# 多出来12条边
			self.ecolors[i+16] = '0'		# 黑色边
			self.esize[i+16] = 0.0			# 边宽为0

	# 添加某个请求，进行画图参数更新
	# cpu越大节点越白,值越大; ram越大节点越大
	def state_add_req(self, row):
		self.reset_add_req()
		node = row['area_id'] * 3 + 18 + row['node_id']		# 节点位置
		# 节点颜色表示CPU负载大小，不高于本地CPU
		self.ncolors[node] = str(round(row['cpu'] / self.CPU_ONE, 3))
		self.nsize[node] = row['cpu'] * 2		# 请求的节点大小2-60
		# 线宽表示带宽
		self.esize[16+node-18] = round(row['bandwidth'] / self.BD_WIDTH, 2)
		if row['delay_sen']:
			self.ecolors[16+node-18] = '1'			# 延时敏感，白色
		else:
			self.ecolors[16+node-18] = '0.5'		# 延时不敏感，半灰度
		
	# 状态发生改变，重新计算画图参数
	def state_change(self):
		# 接入之后cpu增加，节点负载增加，变白
		for i in range(len(self.curr_load)):
			if i % 4 == 0:  	# neigh节点
				self.ncolors[i] = str(round(self.curr_load[i][0] / self.CPU_TWO, 3))
			else:		# local 节点
				self.ncolors[i] = str(round(self.curr_load[i][0] / self.CPU_ONE, 3))
		# 接入之后ram增加，节点变大
		for i in range(len(self.curr_load)):
			self.nsize[i] = self.curr_load[i][1] * 2
		# 接入之后流量增加，边的宽度变大
		for i in range(len(self.route_edges)):  # esize的参数和route_edges是按顺序一一对应的
			item = (self.route_edges[i][0], self.route_edges[i][1])
			reverse_item = (self.route_edges[i][1], self.route_edges[i][0])  # 某些边有双向流量
			self.esize[i] = round((self.e_width[item] + self.e_width[reverse_item]) / self.BD_WIDTH, 2)

	# 请求离开，相当于环境随机改变
	def env_rd_change(self, row):
		if self.vm_locate_idx[row['ReqNo']][0] == 'block':  # 该请求并未接入，Pass
			pass
		else:
			self.rele_current_load(row)
			self.rele_edge_width(row)
			self.state_change()

	# 环境对某个动作的反应
	def env_react(self, action, row):
		# 对某个动作的可行性进行分析
		locate_flag, vm_locate, shortest_path = self.action_judge(action, row)
		self.vm_locate_idx[row['ReqNo']] = [locate_flag, vm_locate, shortest_path]  # 位置分类 + 具体位置
		if locate_flag == 'block':
			pass
		else:
			self.fill_current_load(row)
			self.fill_edge_width(row)
			self.state_change()

	# 环境对某个动作判断是否可以接入，并储存必要信息
	def action_judge(self, action, row):
		node = row['area_id'] * 4 + row['node_id'] + 1
		locate_flag = 'block'
		vm_locate = 'node'

		# 尝试接入本地
		if action == 'local':
			shortest_path = []
			if self.curr_load[node][0] + row['cpu'] <= self.CPU_ONE and \
				self.curr_load[node][1] + row['ram'] <= self.RAM_ONE:		# 成功接入本地
				locate_flag = 'local'
				vm_locate = node
			else:  # 动作无效，阻塞
				pass
			return locate_flag, vm_locate, shortest_path

		# 尝试接入临近节点
		if action == 'neigh':
			neigh_node = row['area_id'] * 4
			shortest_path = nx.shortest_path(self.G, source=node, target=neigh_node)
			if self.curr_load[neigh_node][0] + row['cpu'] <= self.CPU_TWO and \
				self.curr_load[neigh_node][1] + row['ram'] <= self.RAM_TWO and \
				self.is_enough_bd(row, shortest_path):		# 成功接入neigh
				locate_flag = 'neigh'
				vm_locate = neigh_node
			else:
				pass
			return locate_flag, vm_locate, shortest_path

		# 尝试接入数据中心
		if action == 'DC':
			if row['area_id'] < 2:			# 区域0和1使用数据中心‘16’
				DC_node = 16
			else:
				DC_node = 17
			shortest_path = nx.shortest_path(self.G, source=node, target=DC_node)
			if self.is_enough_bd(row, shortest_path):		# 成功接入DC
				locate_flag = 'DC'
				vm_locate = DC_node
			return locate_flag, vm_locate, shortest_path

	# reward函数，根据当前的状态及动作给出相应的reward
	def get_reward(self, action, row):
		reward = 0
		locate_flag, vm_locate, shortest_path = self.action_judge(action, row)
		if locate_flag == 'local':
			reward += 800
		elif locate_flag == 'neigh':
			reward += 800
			if row['delay_sen'] == 0:
				pass
			else:
				reward -= 200  # 延时敏感，奖励-200
			reward -= (int(row['bandwidth'] / 1000) + 1) * 100		# 占用带宽，带宽越大，惩罚越大，100-300之间
		elif locate_flag == 'DC':		# 数据中心
			reward += 800		# 成功接入数据中心，奖励50
			if row['delay_sen'] == 0:
				pass
			else:
				reward -= 400		# 延时敏感，奖励-400
			reward -= (int(row['bandwidth'] / 1000) + 1) * 200		# 占用带宽，带宽越大，惩罚越大，200-600之间
		else:			# block
			reward -= 1000
		return reward

	# 从三个动作中随机取一个
	def random_action(self):
		actions = [0, 1, 2]
		action = random.choice(actions)
		return action
	
	# 观察到的是图像
	def observation(self, ReqNo):
		plt.clf()
		nx.draw(self.G_graph, self.pos, with_labels=False, node_color=self.ncolors, node_shape="o",
				node_size=self.nsize, width=self.esize, edge_color=self.ecolors)
		plt.xlim(-8, 10)			# 设置首界面X轴坐标范围
		plt.ylim(-25, 12)			# 设置首界面Y轴坐标范围
		plt.style.use('dark_background')
		buffer_ = io.BytesIO()
		fig_path = './state/env_' + str(ReqNo) + '.png'
		plt.savefig(fig_path)		# 正式版记得删掉
		plt.savefig(buffer_, format='png')
		buffer_.seek(0)
		dataPIL = PIL.Image.open(buffer_)
		data = np.asarray(dataPIL)
		state = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
		state = cv2.resize(state, (512, 512))
		# cv2.imwrite(fig_path,state)
		return state

	'''
	def observation(self, ReqNo):
		plt.clf()
		nx.draw(self.G_graph, self.pos, with_labels=False, node_color=self.ncolors, node_shape="o",
				node_size=self.nsize, width=self.esize, edge_color=self.ecolors)
		plt.xlim(-8, 10)			# 设置首界面X轴坐标范围
		plt.ylim(-25, 12)			# 设置首界面Y轴坐标范围
		plt.style.use('dark_background')
		fig_path = './state/env_' + str(ReqNo) + '.png'
		plt.savefig(fig_path)
		return 'state'
	'''

	# 根据动作执行下一步操作
	def step(self, action_idx):
		action = self.actions[action_idx]
		is_done = False
		row = self.df.iloc[self.idx]
		reward = self.get_reward(action, row)       # 该动作获得的奖励
		self.env_react(action, row)

		if row['ReqNo'] == self.REQ_NUM - 1:
			is_done = True
			state = self.observation(row['ReqNo'])
			return state, float(reward), is_done

		while True:
			self.idx += 1
			print(self.idx)
			if self.df.iloc[self.idx]['status'] == 'leave':
				row = self.df.iloc[self.idx]
				self.env_rd_change(row)
			else:
				self.state_add_req(self.df.iloc[self.idx])
				break

		row = self.df.iloc[self.idx]
		state = self.observation(row['ReqNo'])
		return state, float(reward), is_done

	def start(self):
		self.initial()
		row = self.df.iloc[self.idx]
		self.state_add_req(row)
		return self.observation(0)


if __name__ == '__main__':
	env = Environment()
	while True:
		observation = env.start()
		while True:
			one_action = env.random_action()		# 执行动作
			environment, one_reward, done = env.step(one_action)
			if done:
				# 游戏结束
				print("done")
				break

