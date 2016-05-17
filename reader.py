import unicodedata
import numpy as np
import glob
import os

class Reader(object):

	def __init__(self, path):

		self.path = path
		self.paths_list = [] 

	def _get_directories(self):

		os.chdir(self.path)

		for dirname, dirnames, filenames in os.walk('.'):
			for filename in filenames:
				path = (os.path.join(dirname, filename))
				self.paths_list.append(path)
				
	def get_paths(self):

		self._get_directories()

		return self.paths_list

# r = Reader("data/spa_corpus/corpus-20090418")
# l = r.get_paths()
# s = l[5][1:].split("/")
# c = s[0]
# n = s[1][:10]
# g = n[1:2]
# p = n[3:4]
# t = n[-1]

# print(g, p, t, n)




