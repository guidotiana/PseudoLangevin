import torch


# Choose a cost function
def choose_cost(name:str, device = 'cpu', **kwargs):
	assert name in ('MSE', 'BCE', 'CE', 'CE_FF', 'Hinge'), f"choose_cost(): invalid value for 'name' variable ({name}). Allowed values: 'MSE', 'BCE', 'CE', 'CE_FF',  'Hinge'."
	if not 'mean' in kwargs.keys(): kwargs['mean'] = True


	if name == 'MSE':
		if kwargs['mean']: Cost = lambda fx, y: ((fx - y)**2).mean()
		else: Cost = lambda fx, y: (fx - y)**2


	elif name == 'BCE':
		if not 'zeta' in kwargs.keys():
			kwargs['zeta'] = 0.5
		k = 1. / (2. * kwargs['zeta'])

		def Cost(fx, y):
			y_onehot = torch.nn.functional.one_hot(y, num_classes=2).float() * 2 - 1
			loss = k * torch.log(1. + torch.exp(-fx * y_onehot / k))
			loss = loss.sum(dim=1)  # sommo sulle 2 classi per ogni esempio
			if kwargs.get('mean', True):
				return loss.mean()
			else:
				return loss


	elif name == 'CE':
		if not 'zeta' in kwargs.keys(): kwargs['zeta'] = 0.5
		k = 1./(2.*kwargs['zeta'])

		def Cost(fx, masks, k=k):
			log_probs = torch.nn.functional.log_softmax(fx, dim= -1)
			costs = [
				log_probs[iseq, mask[:,0], mask[:,1]].mean()
				for iseq, mask in enumerate(masks)
			]
			total_cost = sum(costs)/len(masks)
			return -k * total_cost


	elif name == 'CE_FF':

		if not 'zeta' in kwargs.keys(): kwargs['zeta'] = 0.5
		k = 1./(2.*kwargs['zeta'])

		def Cost(fx, masks, k=k):
			softmax_fx = torch.nn.functional.log_softmax(fx, dim= -1)
			costs = []
			for iseq, mask in enumerate(masks):
				if len(mask) != 0:
					cost = softmax_fx[iseq, mask[:, 0], mask[:, 1]].mean()
					costs.append(cost)
				else:
					costs.append(torch.tensor(0))
			total_cost = sum(costs)/len(masks)
			return -k * total_cost


	else:
		if not 'k' in kwargs.keys(): kwargs['k'] = 0.
		if not 'g' in kwargs.keys(): kwargs['g'] = 1.
		if kwargs['mean']: Cost = lambda fx, y: torch.pow(torch.max(kwargs['k']-torch.mul(fx, y), torch.zeros_like(y)), kwargs['g']).mean()
		else: Cost = lambda fx, y: torch.pow(torch.max(kwargs['k']-torch.mul(fx, y), torch.zeros_like(y)), kwargs['g'])

	return Cost



# Choose a metric function
def choose_metric(name:str, device = 'cpu', **kwargs):
	assert name in ('accuracy_BCE', 'accuracy_CE', 'accuracy_CE_FF'), f"choose_metric(): invalid value for 'name' variable ({name}). Allowed values: 'accuracy_BCE', 'accuracy_CE', 'accuracy_CE_FF."

	if name == 'accuracy_BCE':
		def Metric(fx, y):
			preds = fx.argmax(dim=1)
			return (preds == y).float().mean()

	elif name == 'accuracy_CE':
		def Metric(fx, masks):
			softmax_fx = torch.nn.functional.softmax(fx, dim= -1)
			total_mean = 0.0

			for iseq, mask in enumerate(masks):
				count = 0.
				predictions = torch.argmax(softmax_fx[iseq], dim=1)[mask[:,0]]
				for i, x in enumerate( mask[:,1]):
					count += (predictions[i] == x)
				total_mean += count/len(predictions)

			return total_mean / len(masks)

	else:
		def Metric(fx, masks):
			softmax_fx = torch.nn.functional.softmax(fx, dim= -1)
			total_mean = 0.0

			for iseq, mask in enumerate(masks):
				if len(mask) != 0:
					predictions = torch.argmax(softmax_fx[iseq], dim=1)[mask[:,0]]
					total_mean += (predictions == torch.tensor([x for x in mask[:,1]]).to(device)).float().mean()#.item()            

			return total_mean / len(masks)

	return Metric
