class ConfigDict(dict):
	def __init__(self, filename):
		dict.__init__(self)
		for line in open(filename):
			if line.startswith('#'):
				continue
			if not '=' in line:
				continue
			split = [x.strip() for x in line.split('=')]
			self[split[0]] = split[1]	
