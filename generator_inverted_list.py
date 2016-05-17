from nltk import word_tokenize
import re
import unicodedata
import reader


class GeneratorInvertedList(object):

	def __init__(self, file_list):
		self.file_list = file_list
		self.write_path = ""
		self.read_paths = []
		self.inverted_list = {}
		self.document_id = None
		self.no_stopwords_tokens = []
		self.inverted_list_csv = None


	def _tokenize_text(self, text):
		return word_tokenize(unicode(text, errors='replace').upper())
		#return word_tokenize(unicodedata.normalize('NFKD', text).encode('ascii','ignore').upper().decode('utf-8'))

	def _contains_number(self,token):
		p = re.compile('\d+')

		if p.findall(token) == []:
			return False
		else:
			return True		
	

	def _remove_stopwords(self,tokens):
		
		no_stopwords = []

		for token in tokens:
			if len(token) > 1 and self._contains_number(token) is False:
				no_stopwords.append(token)
			else: 
				pass

		return no_stopwords
	

	def _get_paths(self):
		
		for path in self.file_list:

			splited_path = path[1:].split("/")

			if len(splited_path) > 1:
				pass


	def _insert_inverted_list(self, document_id, tokens):

		for token in tokens:
			if not token in self.inverted_list:
				self.inverted_list[token] = []
				#self.terms_number +=  1
			self.inverted_list[token].append(document_id)			

	def _read_files(self):

		for path in self.file_list[1:]:

			splited_path = path[2:].split("/")

			category = splited_path[0] 
			file_name = splited_path[1][:10]
			# group = file_name[1:2]
			# person = file_name[3:4]
			# task = file_name[-1]

			print(path)
			file = open(path, 'r')

			content = self._tokenize_text(file.read())

			no_stopwords_content = self._remove_stopwords(content)
			self._insert_inverted_list(file_name, content)




			
	def execute(self):
		self._read_files()

r = reader.Reader("data/spa_corpus/corpus-20090418")
l = r.get_paths()
g = GeneratorInvertedList(l)

g.execute()
						
