# '''

# 	I want to receive recognized text from my CRNN algorithm, and clean it so that we can make sense out of it. 
# 	Specifically, given text, I want to:

# 		> Split text into seperate words
# 		> Match text to actual words using a dictionary and rules (i.e. 'flook' becomes 'floor')

# '''

# from math import log
# import re
# import string

# # Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
# # words = open("words-by-frequency.txt").read().split()
# # words = open("one-grams.txt").read().split()[0::2]

# # wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
# # maxword = max(len(x) for x in words)

# # def infer_spaces(s):
# # 	"""Uses dynamic programming to infer the location of spaces in a string
# # 	without spaces."""

# # 	# Find the best match for the i first characters, assuming cost has
# # 	# been built for the i-1 first characters.
# # 	# Returns a pair (match_cost, match_length).
# # 	def best_match(i):
# # 		candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
# # 		return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

# # 	# Build the cost array.
# # 	cost = [0]
# # 	for i in range(1,len(s)+1):
# # 		c,k = best_match(i)
# # 		cost.append(c)

# # 	# Backtrack to recover the minimal-cost string.
# # 	out = []
# # 	i = len(s)
# # 	while i>0:
# # 		c,k = best_match(i)
# # 		assert c == cost[i]
# # 		out.append(s[i-k:i])
# # 		i -= k

# # 	return " ".join(reversed(out))

# import re
# from collections import Counter
# from autocorrect import spell

# def words(text): return re.findall(r'\w+', text.lower())
# # 
# # WORDS = Counter(words(open("words-by-frequency.txt").read()))
# WORDS = open("one-grams.txt").read().split()[0::2]

# PROB = open("one-grams.txt").read().split()

# ONLY_PROBS = open("one-grams.txt").read().split()[1::2]

# def P(word): 
# 	"Probability of `word`."
# 	return int(ONLY_PROBS[WORDS.index(word)]) / 588124220187

# def correction(word): 
# 	"Most probable spelling correction for word."
# 	if word.isalpha():
# 		return max(candidates(word), key=P)
# 	else:
# 		return word

# def candidates(word): 
# 	"Generate possible spelling corrections for word."
# 	return set.union(known([word]), known(edits1(word)), set(word))
# 	# return (known([word]) and known(edits1(word)) and known(edits2(word)) and [word])

# def known(words): 
# 	"The subset of `words` that appear in the dictionary of WORDS."
# 	return set(w for w in words if w in WORDS)

# def edits1(word):
# 	"All edits that are one edit away from `word`."
# 	letters    = 'abcdefghijklmnopqrstuvwxyz'
# 	splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
# 	deletes    = [L + R[1:]               for L, R in splits if R]
# 	transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
# 	replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
# 	inserts    = [L + c + R               for L, R in splits for c in letters]
# 	return set(deletes + transposes + replaces + inserts)

# def edits2(word): 
# 	"All edits that are two edits away from `word`."
# 	return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# def edits3(word): 
# 	"All edits that are three edits away from `word`."
# 	return (e3 for e1 in edits1(word) for e2 in edits1(e1) for e3 in edits1(e2))

# def build(predictions):
# 	print("Current Guess =", predictions[-1])

# def splitPairs(word, maxLen=20):
#    return [(word[:i+1], word[i+1:]) for i in range(max(len(word), maxLen))]

import re
from collections import Counter

thres = 0

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open("./LM/frequency_dictionary_en_82_765.txt").read()))

with open("./LM/frequency_dictionary_en_82_765.txt") as f:
		content = f.readlines()

only_words = []
only_probs = []
for i in range(len(content)):
	entry = content[i].rstrip().split(' ')
	entry[0] = entry[0].replace(u'\ufeff', '')
	only_words.append(entry[0])
	only_probs.append(int(entry[1]))

# def P(word, N=sum(WORDS.values())): 
# 	"Probability of `word`."
# 	return WORDS[word] / N

def P(word): 
	"Probability of `word`."
	try:
		return only_probs[only_words.index(word)]
	except:
		return 1.0

def correction(word): 
	"Most probable spelling correction for word."
	# return max(candidates(word), key=P)
	if word.isalpha():
		candys1 = []
		candys2 = []
		candits = candidates(word)
		if len(candits) != 0:
			for cand in candits:
				candys1.append(P(cand) / levenshteinDistance(word, cand))
				candys2.append(cand)
			if max(candys1) > thres:
				return candys2[np.argmax(candys1)]
			else:
				return word
		else:
			return word
	else:
		return word
	

def candidates(word): 
	"Generate possible spelling corrections for word."
	# return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

	# if len(known(word)) != 0:
	# 	return known([word])#return set.union(known([word]), known(edits1(word)))
	# else:
	return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
	"The subset of `words` that appear in the dictionary of WORDS."
	return set(w for w in words if w in WORDS)

def edits1(word):
	"All edits that are one edit away from `word`."
	letters    = 'abcdefghijklmnopqrstuvwxyz'
	splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
	deletes    = [L + R[1:]               for L, R in splits if R]
	transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
	replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
	inserts    = [L + c + R               for L, R in splits for c in letters]
	return set(replaces + inserts)

def edits2(word): 
	"All edits that are two edits away from `word`."
	return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def levenshteinDistance(s1, s2):
	if s1 == s2:
		return 0.5
	if len(s1) > len(s2):
		s1, s2 = s2, s1

	distances = range(len(s1) + 1)
	for i2, c2 in enumerate(s2):
		distances_ = [i2+1]
		for i1, c1 in enumerate(s1):
			if c1 == c2:
				distances_.append(distances[i1])
			else:
				distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
		distances = distances_
	return distances[-1]

# import segment
import numpy as np
import time
if __name__ == "__main__":
	# slist = 'Theonlydifferencesareminor.Thisreturnsalistratherthanastr,itworksinpython3,itincludesthewordlistandproperlysplitseveniftherearenon-alphachars(likeunderscores,dashes,etc).'
	# slist = ['andbu', 'Exceptfor', 'FUI', 'AShion', 'InGerie', 'ickets', 'mayEAIR', 'Flook', 'OSaeco', 'riser', 'StudentAccoun', 'Knowanyone', 'guniversity', 'Insura', 'Centre,']
	# slist = ['insura']

	# with open("words-by-frequency.txt").read().split()

	# print(candidates('th'))
	w = 'swarovsk'
	# print(segment.segment(w))
	print(correction(w))
	# for i in candidates(w):
	# 	print(i, P(i), levenshteinDistance(w, i))

	


	# start = time.time()
	# print(correction("th"))
	# print("took:", time.time()-start)
	# start = time.time()
	# print(correction("pizzo"))
	# print("took:", time.time()-start)
	# print(P("pizza"))
	# word = "eports"
	# start = time.time()
	# print(correction(word))
	# print("took:", start-time.time())
	# print(segment.segment(word))
	# print(segment.wordSeqFitness(["homebuilt", "airplanes"]))
	# print(segment.wordSeqFitness(["home", "builtairplanes"]))
	# print(segment.wordSeqFitness(["homebuiltair", "planes"]))
	# print(segment.wordSeqFitness(["homebuilt", "air", "planes"]))
	# print(segment.wordSeqFitness(["home", "built", "air", "planes"]))
	# print(segment.wordSeqFitness(["h", "omebuiltairplanes"]))
	# print(infer_spaces([slist]))
	# print("\n".join(open("one-grams.txt").read().split()[0::2]))
	# print(open("words-by-frequency.txt").read())
	# for s in slist:
	# 	print(segment.segment(s))
	# 	for i in segment.segment(s):
	# 		print(correction(i))
	# print(correction(slist))
	# for s in slist:
	# 	possible = []

	# 	# pre-processing
	# 	s = s.lower()
	# 	s = re.sub(r'[^\w\s]','',s)
	# 	s = ''.join([i for i in s if not i.isdigit()])

	# 	for i in s:
	# 		if i.isdigit():
	# 			print("Contains number")

	# 	# ============ BRANCH 1 ===================
	# 	possible.append(candidates(s))

	# 	# # ============ BRANCH 2 ===================
	# 	# split_words = infer_spaces(s).split(" ")
	# 	# print(split_words)
	# 	# for word in split_words:
	# 	# 	possible.append(candidates(word))

	# 	# ============ BRANCH 3 ===================
	# 	for i in range(len(s) + 1):
	# 		possible.append(candidates(s[:i]))
	# 		possible.append(candidates(s[i:]))

	# 	# ============ Combine ===================
	# 	possible = [item for sublist in possible for item in sublist]
	# 	for i in possible:
	# 		print(i, "\t\t\t", P(i))

		# for pair in splits:
		# 	word1 = pair[0]
		# 	word2 = pair[1]
		# 	splits1 = [(word1[:i], word1[i:])    for i in range(len(word1) + 1)]
		# 	splits2 = [(word2[:i], word2[i:])    for i in range(len(word2) + 1)]
		# 	all.append([item for sublist in splits1 for item in sublist])
		# 	all.append([item for sublist in splits2 for item in sublist])
		# all = [item for sublist in all for item in sublist]
		# all = list(set(all))
		# # print(splits)
		# # print("-"*100)
		# # print(all)
		# fixed = []
		# for wrd in all:
		# 	fixed.append(correction(wrd))
		# fixed = list(set(fixed))
		# print(fixed)
		# print("-"*100)
		# print(known(fixed))
		# print("-"*100)
		# print(candidates("accoun"))


		# s = s.lower()
		# s = re.sub(r'[^\w\s]','',s)
		# s = ''.join([i for i in s if not i.isdigit()])
		# for i in edits2(s):
		# 	sw = infer_spaces(i)
		# 	ss = sw.split(" ")
		# 	for w in ss:
		# 		print(correction(w))
		
		# s = s.split(" ")
		# fixed = []
		# for w in s:	
		# 	fixed.append(correction(w))
		# s = ' '.join(fixed)
		# print(correction(s))
		# print(spell(s))


	'''

	Try multiple different NLP strategies, to get many different possible texts.
	Then find the most likely and use that.

	'''

