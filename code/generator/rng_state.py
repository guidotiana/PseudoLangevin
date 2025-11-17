import numpy as np


""" ###################### """
""" Numpy random generator """
""" ###################### """

# Save numpy generator state
def save_np_state(f):
	state = np.random.get_state()
	state = np.array(state, dtype='object')
	with open(f, 'wb') as ff:
		np.save(ff, [state])

# Load numpy generator state
def load_np_state(f):
	with open(f, 'rb') as ff:
		state = np.load(ff, allow_pickle='True')
	state = tuple(state[0])
	np.random.set_state(state)
