import networkx as nx
import random
import matplotlib.pyplot as plt
import cv2
import io
import numpy as np
import PIL
# import reinforce.network.traffic as traffic
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
			(1, 0), (3, 0), (4, 0), (0, 16),		# 节点0
			(2, 1), (5, 4), (7, 4), (6, 5), 		# 其他节点
			(9, 8), (11, 8), (8, 12), (10, 9), 		# 其他节点
			(13, 12), (15, 12), (12, 17), (14, 13)]	# 其他节点
		# 用来画图的边的关系，16+12条边，注意顺序！！！
		self.graph_edges = [
			(1, 0), (3, 0), (4, 0), (0, 16),		# 节点0
			(2, 1), (5, 4), (7, 4), (6, 5), 		# 其他节点
			(9, 8), (11, 8), (8, 12), (10, 9), 		# 其他节点
			(13, 12), (15, 12), (12, 17), (14, 13),	# 其他节点
			(18, 30), (19, 31), (20, 32), (21, 33), (22, 34), (23, 35),     # 描绘请求
			(24, 36), (25, 37), (26, 38), (27, 39), (28, 40), (29, 41)]     # 描绘请求
		# 18 + 24个节点的坐标
		self.pos = [
			(-2, 4), (-3, 5), (-4, 4), (-3, 3),			# 区域1
			(0, 5), (1, 6), (0, 7), (-1, 6),			# 区域2
			(0, 3), (1, 2), (0, 1), (-1, 2),			# 区域3
			(2, 4), (3, 5), (4, 4), (3, 3),				# 区域4
			(-2, 7), (2, 7),							# 2个数据中心
			(-3, -2), (-4, -3), (-3, -4), (1, -1), (0, 0), (-1, -1),		# 区域1和2的向下平移
			(1, -5), (0, -6), (-1, -5), (3, -2), (4, -3), (3, -4),			# 区域3和4的向下平移
			(-2, -2), (-3, -3), (-2, -4), (2, -1), (1, 0), (0, -1),			# 虚拟节点
			(2, -5), (1, -6), (0, -5), (4, -2), (5, -3), (4, -4)]			# 虚拟节点

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
	def graph_init(self):
		G_graph = nx.Graph()
		G_graph.add_nodes_from([i for i in range(42)])
		G_graph.add_edges_from(self.graph_edges)
		self.G_graph = G_graph
		# 节点的颜色，表示cpu负载，初始为黑色0，0~1的范围
		self.ncolors = ['0.0' for i in range(42)]
		# 节点的大小，表示ram负载，初始为0，参数最大为400，继续变大会导致节点重合
		self.nsize = [0 for i in range(42)]
		# 边的颜色，16+12条边。前16条边恒定为白色1，其宽度表示负载大小
		# 后12条边，白色表示延时敏感，半灰色表示延时不敏感
		self.ecolors = ['1' for i in range(len(self.graph_edges))]
		# 边的宽度，16+12条边，其宽度表示负载大小
		self.esize = [0.0 for i in range(len(self.graph_edges))]# 边的宽度，随负载增加

	# 初始化各条边的最大带宽，注意双向带宽合为单向带宽
	def edge_init(self):
		self.max_bd = {}
		self.e_width = {}
		for item in self.route_edges:
			self.e_width[item] = 0			# 储存每条边的负载，对应线的宽度
			self.max_bd[item] = 40000		# 每条边最大的带宽

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
			self.nsize[i+18] = 0			# 大小为0
		for i in range(12):		# 多出来12条边
			self.ecolors[i+16] = '0'		# 黑色边
			self.esize[i+16] = 0.0			# 边宽为0

	# 添加某个请求的画图
	# cpu越大节点越白,值越大; ram越大节点越大
	def state_add_req(self, row):
		self.reset_add_req()
		node = row['area_id'] * 3 + 18 + row['node_id']		# 节点位置
		position = self.graph_edges.index((node, node + 12))		# 找到索引边
		# 节点颜色表示CPU大小，不高于本地CPU
		self.ncolors[node] = str(round(row['cpu'] / self.CPU_ONE, 3))
		# 节点大小表示RAM大小，1-30范围，直接表示
		self.nsize[node] = row['ram']		# 请求的节点大小1-30
		# 颜色表示延时敏感与否
		if row['delay_sen']:
			self.ecolors[position] = '1'			# 延时敏感，白色
		else:
			self.ecolors[position] = '0.5'		# 延时不敏感，半灰度
		# 线宽表示带宽
		self.esize[position] = round(row['bandwidth'] / self.BD_WIDTH, 2)
		
	# 状态发生改变，重新计算画图参数
	def state_change(self):
		# ncolors表示cpu负载
		for i in range(len(self.curr_load)):
			if i % 4 == 0:  	# neigh节点
				self.ncolors[i] = str(round(self.curr_load[i][0] / self.CPU_TWO, 3))
			else:				# local 节点
				self.ncolors[i] = str(round(self.curr_load[i][0] / self.CPU_ONE, 3))
		# nsize表示ram负载
		for i in range(len(self.curr_load)):
			self.nsize[i] = self.curr_load[i][1]
		# 边的宽度表示链路带宽
		for i in range(len(self.route_edges)):  # 
			item = self.route_edges[i]
			self.esize[i] = round(self.e_width[item] / self.BD_WIDTH, 2)

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
			reward += 250
		elif locate_flag == 'neigh':
			reward += 250
			if row['delay_sen'] == 0:
				pass
			else:
				reward -= 50  # 延时敏感，奖励-50
			reward -= (int(row['bandwidth'] / 1000) + 1) * 10		# 占用带宽，带宽越大，惩罚越大，10-30之间
		elif locate_flag == 'DC':		# 数据中心
			reward += 250		# 成功接入数据中心，奖励250
			if row['delay_sen'] == 0:
				pass
			else:
				reward -= 100		# 延时敏感，奖励-100
			reward -= (int(row['bandwidth'] / 1000) + 1) * 20		# 占用带宽，带宽越大，惩罚越大，20-60之间
		else:			# block
			reward -= 1000
		return reward

	# 从三个动作中随机取一个
	def random_action(self):
		actions = [0, 1, 2]
		action = random.choice(actions)
		return action
	
	# 使用fcfs测试reward是否正确
	def fcfs_action(self):
		row = self.df.iloc[self.idx]
		action = 0

		node = row['area_id'] * 4 + row['node_id'] + 1
		# 尝试使用local节点
		if self.curr_load[node][0] + row['cpu'] <= self.CPU_ONE and \
			self.curr_load[node][1] + row['ram'] <= self.RAM_ONE:
			action = 0
			return action
		
		# 尝试使用邻居节点
		neigh_node = row['area_id'] * 4
		shortest_path = nx.shortest_path(self.G, source=node, target=neigh_node)
		if self.curr_load[neigh_node][0] + row['cpu'] <= self.CPU_TWO and \
			self.curr_load[neigh_node][1] + row['ram'] <= self.RAM_TWO and \
			self.is_enough_bd(row, shortest_path):
			action = 1
			return action
		
		# 使用数据中心
		if row['area_id'] < 2:			# 区域0和1使用数据中心‘16’
			DC_node = 16
		else:
			DC_node = 17
		shortest_path = nx.shortest_path(self.G, source=node, target=DC_node)
		if self.is_enough_bd(row, shortest_path):
			action = 2
			return action
		return action
	
	'''
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
		# plt.savefig(fig_path)		# 正式版记得删掉
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
		return 'state'
		plt.clf()
		nx.draw(self.G_graph, self.pos, with_labels=False, node_color=self.ncolors, node_shape="o",
				node_size=self.nsize, width=self.esize, edge_color=self.ecolors)
		plt.xlim(-7, 8)			# 设置首界面X轴坐标范围
		plt.ylim(-7, 8)			# 设置首界面Y轴坐标范围
		plt.style.use('dark_background')
		buffer_ = io.BytesIO()
		fig_path = './state/env_' + str(ReqNo) + '.png'
		plt.savefig(buffer_, format='png')
		buffer_.seek(0)
		dataPIL = PIL.Image.open(buffer_)
		data = np.asarray(dataPIL)
		state = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
		state = cv2.resize(state, (512, 512))
		cv2.imwrite(fig_path, state)
		return 'state'

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

	# 开始第一步
	def start(self):
		self.initial()
		row = self.df.iloc[self.idx]
		self.state_add_req(row)
		return self.observation(0)


if __name__ == '__main__':
	total_reward = 0
	env = Environment()
	while True:
		observation = env.start()
		while True:
			one_action = env.random_action()		# 执行动作
			# one_action = env.fcfs_action()
			environment, one_reward, done = env.step(one_action)
			total_reward += one_reward
			if done:
				# 游戏结束
				print("done")
				break
		break
	print(total_reward/1000)
