import pattern.de
import pattern.en
import string

class PosTagger:

        posclasses = {'CC':'V', 'CD':'N', 'DT':'IN', 'EX':'IN', 'FW':'N', 'IN':'IN', 
                      'JJ':'J', 'JJR':'J', 'JJS':'J', 'LS':'.', 'MD':'V', 
                      'NN':'N', 'NNS':'N', 'NNP':'N', 'NNPS':'N', 'PDT':'IN', 
                      'POS':'IN', 'PRP':'N', 'PRP$':'N', 'RB':'J', 'RBR':'J', 
                      'RBS':'J', 'RP':'IN', 'SYM':'.', 'TO':'P', 'UH':'V', 
                      'VB':'V', 'VBD':'V', 'VBG':'V', 'VBN':'V', 'VBP':'V', 
                      'VBZ':'V', 'WDT':'W', 'WP':'W', 'WP$':'W', 'WRB':'W',
                      '!':'.', '"':'.', '#':'.', '$':'.', '%':'.', '&':'.',
                      '\'':'.', '(':'.', ')':'.', '*':'.', '+':'.', ',':'.',
                      '-':'.', '.':'.', '/':'.', ':':'.', ';':'.', '<':'.', 
                      '=':'.', '>':'.', '?':'.', '@':'.', '[':'.', '\\':'.',
                      ']':'.', '^':'.', '_':'.', '`':'.', '{':'.', '|':'.',
                      '}':'.','~':'.'}

        def get_POS_class(self, pos):
                return self.posclasses.get(pos,pos)

	def parse(self, sent, lang):
		""" Tag a specific sentence
			Args:
				lang: 'en' or 'de'
		"""
		
		sent_len = len(sent)

		penn_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'] + \
			[c for c in string.punctuation]
		pos_sent = []
		sent_with_pos = []
		sent_str = ' '.join(sent)
		if lang == 'en':
			sent_with_pos = pattern.en.parse(sent_str, tokenize=False, chunks=False)
		elif lang == 'de':
			sent_with_pos = pattern.de.parse(sent_str, tokenize=False, chunks=False)
		else:
			raise ValueError("Language not supported. Enter 'en' or 'de'.")
		for word in sent_with_pos.split(" "):
			pos = word.split('/')[1]
			if pos in penn_tags:
				pos_sent.append(pos)
			else: # OOV with unicode error
				pos = u'NNP' # assume proper noun
				pos_sent.append(pos)

		# Check sentence length same as POS tag length
		if sent_len != len(pos_sent):
			raise ValueError("POS tagged sentence not same length:{:s}\n{:s}".format(sent, pos_sent))
		
		return pos_sent
