import pattern.de
import pattern.en
import string

class PosTagger:

	def parse(self, sent, lang):
		""" Tag a specific sentence
			Args:
				lang: 'en' or 'de'
		"""

		penn_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS' 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'] + \
			[c for c in string.punctuation]
		pos_sent = []
		sent_with_pos = []
		if lang == 'en':
			sent_with_pos = pattern.en.parse(sent, tokenize=False, chunks=False)
		elif lang == 'de':
			sent_with_pos = pattern.de.parse(sent, tokenize=False, chunks=False)
		else:
			raise ValueError("Language not supported. Enter 'en' or 'de'.")
		for word in sent_with_pos.split(" "):
			pos = word.split('/')[1]
			if pos in penn_tags:
				pos_sent.append(pos)
			else: # proper name
				pos = u'NNP'
				pos_sent.append(pos)
		
		return pos_sent 	
