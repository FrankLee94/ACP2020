# !usr/bin/env python
# -*- coding:utf-8 -*-

# 此模块负责奖励函数的计算
# JialongLi 2020/05/14


# reward函数，根据当前的状态及动作给出相应的reward
def get_reward(flag, action, row):
	reward = 0
	if flag == '111':
		if action == 'local':
			reward += 8
		elif action == 'neigh':
			if row['delay_sen'] == 0:
				pass
			else:
				reward -= 2		# 延时敏感，奖励-2
			reward -= (int(row['bandwidth'] / 1000) + 1) * 1	#带宽惩罚 ，1-3之间
		else:
			if row['delay_sen'] == 0:
				pass
			else:
				reward -= 4		# 延时敏感，奖励-4
			reward -= (int(row['bandwidth'] / 1000) + 1) * 2	#带宽惩罚 ，2-6之间

	if flag == '110':
		if action == 'local':
			reward += 8
		elif action == 'neigh':
			if row['delay_sen'] == 0:
				pass
			else:
				reward -= 2		# 延时敏感，奖励-2
			reward -= (int(row['bandwidth'] / 1000) + 1) * 1	#带宽惩罚 ，1-3之间
		else:
			reward -= 10

	if flag == '101':
		if action == 'local':
			reward += 8
		elif action == 'neigh':
			reward -= 10
		else:
			if row['delay_sen'] == 0:
				pass
			else:
				reward -= 4		# 延时敏感，奖励-4
			reward -= (int(row['bandwidth'] / 1000) + 1) * 2	#带宽惩罚 ，2-6之间


	if flag == '100':
		if action == 'local':
			reward += 8
		elif action == 'neigh':
			reward -= 10
		else:
			reward -= 10

	if flag == '011':
		if action == 'local':
			reward -= 10
		elif action == 'neigh':
			reward += 8
			if row['delay_sen'] == 0:
				pass
			else:
				reward -= 2		# 延时敏感，奖励-2
			reward -= (int(row['bandwidth'] / 1000) + 1) * 1	#带宽惩罚 ，1-3之间
		else:
			reward += 8
			if row['delay_sen'] == 0:
				pass
			else:
				reward -= 4		# 延时敏感，奖励-4
			reward -= (int(row['bandwidth'] / 1000) + 1) * 2	#带宽惩罚 ，2-6之间


	if flag == '010':
		if action == 'local':
			reward -= 10
		elif action == 'neigh':
			reward += 8
		else:
			reward -= 10

	if flag == '001':
		if action == 'local':
			reward -= 10
		elif action == 'neigh':
			reward -= 10
		else:
			reward += 8

	if flag == '000':
		reward = 0

	return reward