from gym.envs.registration import register

register(
		id = 'roomworld-v0',
		entry_point = 'gridworlds.envs:RoomWorldEnv'
		)


