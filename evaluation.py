from __future__ import print_function

import json
from key_word_extraction import extract_keywords, load_model, strip_accents_and_uppercase
import spacy
import nltk
from Levenshtein import distance as levenshtein_distance
from greek_stemmer import GreekStemmer


def evaluate():
	with open("preprocess/test_articles_dataset.json", "r") as testF:
		test_articles = json.load(testF)

	nltk.download('stopwords')
	pos_el = spacy.load("el_core_news_md")
	model, tokenizer = load_model("greekBERT")
	stemmer = GreekStemmer()
	num_predictions = 0
	num_golds = 0
	num_relevant = 0

	for article in test_articles:
		doc = article['title'] + " " + article["abstract"]
		gold_keywords = article["keywords"]
		pred_keywords = extract_keywords(doc, model, tokenizer, pos_el, top_n=5)

		gold_keywords_prep = []
		for word in gold_keywords:
			_tmp = []
			for token in strip_accents_and_uppercase(word).split():
				_tmp.append(stemmer.stem(token))
			gold_keywords_prep.append(" ".join(_tmp))

		pred_keywords_prep = []

		for word in pred_keywords:
			_tmp = []
			for token in strip_accents_and_uppercase(word).split():
				_tmp.append(stemmer.stem(token))
			pred_keywords_prep.append(" ".join(_tmp))

		num_predictions += len(pred_keywords_prep)
		num_golds += len(gold_keywords_prep)

		rel = 0
		for pred_word in pred_keywords_prep:
			broken = False
			for gold_word in gold_keywords_prep:
				for token in pred_word.split():
					if token in gold_word:
						rel += 1
						broken = True
						gold_keywords_prep.remove(gold_word)
						break
				if broken:
					break

		print("gold: ", gold_keywords_prep)
		print("pred: ", pred_keywords_prep)
		print(rel)



if __name__ == '__main__':
	evaluate()
