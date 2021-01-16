import json
from key_word_extraction import extract_keywords, load_model, strip_accents_and_uppercase
import spacy
import nltk
from __future__ import print_function
from Levenshtein import distance as levenshtein_distance


def evaluate():
	with open("preprocess/test_articles_dataset.json", "r") as testF:
		test_articles = json.load(testF)

	nltk.download('stopwords')
	pos_el = spacy.load("el_core_news_md")
	model, tokenizer = load_model("greekBERT")
	for article in test_articles:
		doc = article['title'] + " " + article["abstract"]
		gold_keywords = article["keywords"]
		pred_keywords = extract_keywords(doc, model, tokenizer, pos_el, top_n=5)

		gold_keywords = [strip_accents_and_uppercase(word) for word in gold_keywords]
		pred_keywords = [strip_accents_and_uppercase(word) for word in pred_keywords]

		print("gold: ", gold_keywords)
		print("pred: ", pred_keywords)
		break


if __name__ == '__main__':
	evaluate()
