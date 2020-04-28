# !usr/bin/env python
# -*- coding:utf-8 -*-

# this model is for traffic generation
# JialongLi 2020/03/09

import random
import math
import pandas as pd

NODE_NUM = 3				# 每个区域产生业务的节点数目
AREA_NUM = 4				# 一共4个区域
RATIO_DELAY_SEN = 0.3		# 延时敏感业务占比
MAX_CPU = 30				# 单个请求最大的CPU
MAX_RAM = 30				# 单个请求最大的RAM
REQ_NUM = 10000				# 请求的总数目


# 产生一个请求Probability_Poisson
def event_generation(arr_rate, ser_rate):
	probability_poisson = random.random()
	if probability_poisson == 0.0 or probability_poisson == 1.0:
		probability_poisson = 0.5
	interval = -(1e6 / arr_rate) * math.log(1 - probability_poisson)   # event interval
	interval = int(round(interval))
	probability_poisson = random.random()
	persist_time = -(1e6 / ser_rate) * math.log(1 - probability_poisson)  # event service time
	persist_time = int(round(persist_time))
	if interval == 0:			# 避免出现两个业务间隔时间为0的情况
		interval = 1
	if persist_time == 0:		# 避免出现某个业务持续时间为0的情况
		persist_time = 1

	area_id = random.randint(0, AREA_NUM - 1)		# 从0开始到AREA_NUM - 1
	node_id = random.randint(0, NODE_NUM - 1)		# 从0开始到NODE_NUM - 1
	cpu = random.randint(1, MAX_CPU)				# 从1开始到MAX_CPU
	ram = random.randint(1, MAX_RAM)				# 从1开始到MAX_RAM
	bandwidth = (random.randint(1, 60)) * 50		# 50 - 3000 M, granularity is 50M
	if random.random() > RATIO_DELAY_SEN: 
		delay_sen = 0   # 0代表延时不敏感，概率1 - RATIO_DELAY_SEN
	else:
		delay_sen = 1   # 1代表延时敏感，概率RATIO_DELAY_SEN
	return interval, persist_time, area_id, node_id, cpu, ram, bandwidth, delay_sen


# 产生原始的请求集合 + 排序
def traffic_generation(arr_rate, ser_rate):
	traff_info = {
		'ReqNo': [], 'area_id': [], 'node_id': [], 'cpu': [], 'ram': [], 'timing': [],
		'persist_time': [], 'bandwidth': [], 'delay_sen': [], 'status': []}
	
	absolute_time = 0
	for i in range(REQ_NUM):
		interval, persist_time, area_id, node_id, cpu, ram, bandwidth, delay_sen = event_generation(arr_rate, ser_rate)
		absolute_time += interval
		arrive_time = absolute_time
		leave_time = arrive_time + persist_time
		for j in range(2):		# 一个请求分为到达和离开两部分
			traff_info['ReqNo'].append(i)
			traff_info['area_id'].append(area_id)
			traff_info['node_id'].append(node_id)
			traff_info['cpu'].append(cpu)
			traff_info['ram'].append(ram)
			traff_info['persist_time'].append(persist_time)
			traff_info['bandwidth'].append(bandwidth)
			traff_info['delay_sen'].append(delay_sen)
		traff_info['timing'].append(arrive_time)
		traff_info['timing'].append(leave_time)
		traff_info['status'].append('arrive')
		traff_info['status'].append('leave')
	df = pd.DataFrame(traff_info)
	df = df.sort_values(by='timing', axis=0, ascending=True)
	return df


# 获取最新的df
def get_new_df():
	single_erlang = 20
	arrive_rate = NODE_NUM * AREA_NUM * single_erlang
	service_rate = 1
	df = traffic_generation(arrive_rate, service_rate)
	return df
