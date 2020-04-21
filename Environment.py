import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib import style
import cv2


class Environment:
	def __init__(
			self
	):
		self.REQ_NUM = 10000  # 请求的总数目
		self.CPU_ONE = 150.0  # 本地节点CPU总量
		self.CPU_TWO = 500.0  # 二级节点CPU总量
		self.BD_WIDTH = 5000  # 5000M的带宽表示线宽为1
		# 边的连接关系，22条边
		self.route_edges = [
			(0, 1), (1, 2), (2, 3), (3, 0),         # 区域1
			(4, 5), (5, 6), (6, 7), (7, 4),         # 区域2
			(8, 9), (9, 10), (10, 11), (11, 8),     # 区域3
			(12, 13), (13, 14), (14, 15), (15, 12),  # 区域4
			(0, 4), (4, 12), (12, 8), (8, 0), (0, 16), (12, 17)]  # 环及数据中心
		# 用来画图的边的关系，22+12条边
		self.graph_edges = [
			(0, 1), (1, 2), (2, 3), (3, 0),         # 区域1
			(4, 5), (5, 6), (6, 7), (7, 4),         # 区域2
			(8, 9), (9, 10), (10, 11), (11, 8),     # 区域3
			(12, 13), (13, 14), (14, 15), (15, 12),  # 区域4
			(0, 4), (4, 12), (12, 8), (8, 0), (0, 16), (12, 17),  # 环及数据中心
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
			(2, -22), (1, -24), (0, -22), (6, -16), (8, -17), (6, -18)]

		self.df = None 				# dataframe, 读取的excel
		self.idx = None 			# 序号
		self.G = None 				# 用来做拓扑
		self.G_graph = None 		# 用来画图
		self.vm_locate_idx = None 	# 用来画图
		self.curr_load = None
		self.e_width = None
		self.ncolors = None
		self.ecolors = None
		self.lws = None

	# 拓扑初始化，返回一个无向图
	def topo_init(self):
		G = nx.Graph()
		G.add_nodes_from([i for i in range(18)])
		G.add_edges_from(self.route_edges)
		return G

	# 观测到的图像初始化
	# 黑色背景，初始节点为黑色，表示负载为0，边初始为白色1
	def graph_init(self):
		G_graph = nx.Graph()
		G_graph.add_nodes_from([i for i in range(42)])
		G_graph.add_edges_from(self.graph_edges)
		# 节点的颜色，初始为黑色0，随负载增加，0~1的范围
		ncolors = ['0.0' for i in range(42)]
		ecolors = ['1' for i in range(len(self.graph_edges))]	# 边的颜色
		lws = [0.0 for i in range(len(self.graph_edges))]		# 边的宽度，随负载增加
		return G_graph, ecolors, ncolors, lws

	# 算法初始化
	# curr_load：每个节点CPU的使用率
	# vm_locate_idx：存储请求的分类及具体位置，分类有local, neigh, DC，具体位置为本节点id
	def initial(self):
		traffic_file_sort_ph = './traffic_data/traffic_sort_erlang20_num20.xlsx'
		self.df = pd.read_excel(traffic_file_sort_ph)		# 读取随机事件
		self.G = self.topo_init()							# 拓扑初始化
		self.curr_load = [0 for i in range(16)]				# 节点负载,对应每个节点的颜色
		self.vm_locate_idx = {}								# 部署位置,具体位置，及使用的路径
		self.e_width = {}									# 储存每条边的负载，对应线的宽度
		for item in self.route_edges:
			self.e_width[item] = 0
			reverse_item = (item[1], item[0])
			self.e_width[reverse_item] = 0
		self.G_graph, self.ecolors, self.ncolors, self.lws = self.graph_init()		# 刻画状态的变量
		self.idx = 0

	# 某个请求接入，更新虚拟机所在节点的负载
	def fill_current_load(self, ReqNo, cpu):
		if self.vm_locate_idx[ReqNo][0] == 'DC':  # DC的CPU资源无限
			pass
		else:
			vm_locate = self.vm_locate_idx[ReqNo][1]  # 虚拟机的具体位置
			self.curr_load[vm_locate] += cpu

	# 某个请求离开，更新虚拟机所在节点的负载
	def rele_current_load(self, ReqNo, cpu):
		if self.vm_locate_idx[ReqNo][0] == 'DC':  # DC的CPU资源无限
			pass
		else:
			vm_locate = self.vm_locate_idx[ReqNo][1]  # 虚拟机的具体位置
			self.curr_load[vm_locate] -= cpu

	# 某个请求接入，更新路径负载
	def fill_edge_width(self, ReqNo, bandwidth):
		shortest_path = self.vm_locate_idx[ReqNo][2]
		if len(shortest_path) == 0:
			pass
		else:
			for i in range(len(shortest_path) - 1):
				edge = (shortest_path[i], shortest_path[i + 1])
				self.e_width[edge] += bandwidth

	# 某个请求离开，更新路径负载
	def rele_edge_width(self, ReqNo, bandwidth):
		shortest_path = self.vm_locate_idx[ReqNo][2]
		if len(shortest_path) == 0:
			pass
		else:
			for i in range(len(shortest_path) - 1):
				edge = (shortest_path[i], shortest_path[i + 1])
				self.e_width[edge] -= bandwidth

	# 重置请求那部分的画图参数,原始有18个节点，22条边，多了24个节点，12条边
	def reset_add_req(self):
		for i in range(24):		# 多出来的节点
			self.ncolors[i+18] = '0'		# 黑色节点
		for i in range(12):		# 多出来的边
			self.ecolors[i+22] = '0'		# 黑色边
			self.lws[i+22] = 0.0			# 边宽为0

	# 添加某个请求，进行画图参数更新，cpu越大节点越白
	def state_add_req(self, row):
		self.reset_add_req()
		node = row['area_id'] * 3 + 18 + row['node_id']
		# 节点颜色表示CPU负载大小，不高于本地CPU
		self.ncolors[node] = str(round(row['cpu'] / self.CPU_ONE, 3))
		self.lws[22+node-18] = round(row['bandwidth'] / self.BD_WIDTH, 2)		# 线宽表示带宽
		if row['delay_sen']:
			self.ecolors[22+node-18] = '1'			# 延时敏感，白色
		else:
			self.ecolors[22+node-18] = '0.5'		# 延时不敏感，半灰度
		
	# 状态发生改变，重新计算画图参数
	def state_change(self):
		for i in range(len(self.curr_load)):
			if i % 4 == 0:  # neigh节点
				self.ncolors[i] = str(round(self.curr_load[i] / self.CPU_TWO, 3))
			else:
				self.ncolors[i] = str(round(self.curr_load[i] / self.CPU_ONE, 3))
		for i in range(len(self.route_edges)):  # lws的参数和route_edges是按顺序一一对应的
			item = (self.route_edges[i][0], self.route_edges[i][1])
			reverse_item = (self.route_edges[i][1], self.route_edges[i][0])  # 某些边有双向流量
			self.lws[i] = round((self.e_width[item] + self.e_width[reverse_item]) / self.BD_WIDTH, 2)

	# 请求离开，相当于环境随机改变
	def env_rd_change(self, row):
		if self.vm_locate_idx[row['ReqNo']][0] == 'block':  # 该请求并未接入，Pass
			pass
		else:
			self.rele_current_load(row['ReqNo'], row['cpu'])
			self.rele_edge_width(row['ReqNo'], row['bandwidth'])
			self.state_change()

	# 环境对某个动作的反应
	def env_react(self, action, row):
		# 对某个动作的可行性进行分析
		locate_flag, vm_locate, shortest_path = self.action_judge(action, row)
		self.vm_locate_idx[row['ReqNo']] = [locate_flag, vm_locate, shortest_path]  # 位置分类 + 具体位置
		if locate_flag == 'block':
			pass
		else:
			self.fill_current_load(row['ReqNo'], row['cpu'])
			self.fill_edge_width(row['ReqNo'], row['bandwidth'])
			self.state_change()

	# 环境对某个动作判断是否可以接入，并储存必要信息
	def action_judge(self, action, row):
		node = row['area_id'] * 4 + row['node_id'] + 1
		locate_flag = 'block'
		vm_locate = 'node'
		shortest_path = []

		if action == 'local':
			if self.curr_load[node] + row['cpu'] <= self.CPU_ONE:  # 成功接入本地
				locate_flag = action
				vm_locate = node
				shortest_path = []
			else:  # 动作无效，阻塞
				pass
			return locate_flag, vm_locate, shortest_path

		if action == 'neigh':
			neigh_node = row['area_id'] * 4
			if self.curr_load[neigh_node] + row['cpu'] <= self.CPU_TWO:  # 成功接入neigh
				locate_flag = action
				vm_locate = neigh_node
				shortest_path = random.choice([p for p in nx.all_shortest_paths(self.G, source=node, target=vm_locate)])
			else:
				pass
			return locate_flag, vm_locate, shortest_path

		if action == 'DC':
			node = row['area_id'] * 4 + row['node_id'] + 1
			locate_flag = 'DC'
			vm_locate = random.choice([16, 17])  # 从2个数据中心中随机取一个
			shortest_path = random.choice([p for p in nx.all_shortest_paths(self.G, source=node, target=vm_locate)])
		return locate_flag, vm_locate, shortest_path

	# reward函数，根据当前的状态及动作给出相应的reward
	def get_reward(self, action, row):
		reward = 0
		if action == 'local':
			node = row['area_id'] * 4 + row['node_id'] + 1
			if self.curr_load[node] + row['cpu'] <= self.CPU_ONE:  # 成功接入本地，奖励50
				reward += 50
			else:
				reward -= 1000  # 没有充足资源却选择接入，奖励-1000
		elif action == 'neigh':
			neigh_node = row['area_id'] * 4
			if self.curr_load[neigh_node] + row['cpu'] <= self.CPU_TWO:  # 成功接入neigh，奖励50
				reward += 50
			else:
				reward -= 1000  # 没有充足资源却选择接入，奖励-1000
			if row['delay_sen'] == 0:
				pass
			else:
				reward -= 20  # 延时敏感，奖励-20
			reward -= (int(row['bandwidth'] / 1000) + 1) * 2  # 占用带宽，带宽越大，惩罚越大，2-6之间
		else:  # 数据中心
			reward += 50  # 成功接入数据中心，奖励50
			if row['delay_sen'] == 0:
				pass
			else:
				reward -= 40  # 延时敏感，奖励-40
			reward -= (int(row['bandwidth'] / 1000) + 1) * 3  # 占用带宽，带宽越大，惩罚越大，3-9之间
		return reward

	# 从三个动作中随机取一个
	def random_action(self):
		actions = ['local', 'neigh', 'DC']
		action = random.choice(actions)
		return action
	
	# 观察到的是图像
	def observation(self, ReqNo):
		plt.clf()
		nx.draw(self.G_graph, self.pos, with_labels=False, node_color=self.ncolors, node_shape="o",
				node_size=300, width=self.lws, edge_color=self.ecolors)
		plt.xlim(-8, 10)			# 设置首界面X轴坐标范围
		plt.ylim(-25, 12)			# 设置首界面Y轴坐标范围
		plt.style.use('dark_background')
		fig_path = './state/env_' + str(ReqNo) + '.png'
		plt.savefig(fig_path)
		return cv2.imread(fig_path)

	# 根据动作执行下一步操作
	def step(self, action):
		is_done = False
		row = self.df.iloc[self.idx]
		reward = self.get_reward(action, row)       # 该动作获得的奖励
		self.env_react(action, row)

		if row['ReqNo'] == 9999:
			is_done = True
			state = self.observation(row['ReqNo'])
			return state, reward, is_done

		while True:
			self.idx += 1
			if self.df.iloc[self.idx]['status'] == 'leave':
				row = self.df.iloc[self.idx]
				self.env_rd_change(row)
			else:
				self.state_add_req(self.df.iloc[self.idx])
				break

		row = self.df.iloc[self.idx]
		state = self.observation(row['ReqNo'])
		return state, reward, is_done

	def start(self):
		self.initial()
		row = self.df.iloc[self.idx]
		self.state_add_req(row)
		return self.observation(0)


if __name__ == '__main__':	
	env = Environment()
	observation = env.start()
	while True:
		one_action = env.random_action()  # 执行动作
		environment, one_reward, done = env.step(one_action)
		if done:		# 游戏结束
			break
