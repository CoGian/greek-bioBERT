from __future__ import print_function

import json
from key_word_extraction import extract_keywords, load_model, strip_accents_and_uppercase, extract_keywords_RAKE, extract_keywords_YAKE, extract_keywords_TEXTRANK
import spacy
import nltk
from greek_stemmer import GreekStemmer
import argparse
from tqdm import tqdm
from multi_rake import Rake
import yake
import pytextrank


def evaluate(model_name, k=5):
	with open("preprocess/test_articles_dataset.json", "r") as testF:
		test_articles = json.load(testF)

	if model_name == "rake":
		model = Rake(language_code="el")
	elif model_name == "yake":
		model = yake.KeywordExtractor(lan="el", top=k)
	elif model_name == "textrank":
		pos_el = spacy.load("el_core_news_md")
		tr = pytextrank.TextRank()
		pos_el.add_pipe(tr.PipelineComponent, name="textrank", last=True)
	else:
		nltk.download('stopwords')
		pos_el = spacy.load("el_core_news_md")
		model, tokenizer = load_model(model_name)
	stemmer = GreekStemmer()
	num_predictions = 0
	num_golds = 0
	num_relevant = 0

	for article in tqdm(test_articles):
		doc = article['title'] + " " + article["abstract"]
		gold_keywords = article["keywords"]

		if model_name == "rake":
			pred_keywords = extract_keywords_RAKE(model, doc, top_n=k)
		elif model_name == "yake":
			pred_keywords = extract_keywords_YAKE(model, doc)
		elif model_name == "textrank":
			pred_keywords = extract_keywords_TEXTRANK(pos_el, doc, top_n=10)
		else:
			pred_keywords = extract_keywords(doc, model, tokenizer, pos_el, top_n=k)

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
		matched_gold_keywords = []
		for pred_word in sorted(pred_keywords_prep):
			broken = False
			for gold_word in sorted(gold_keywords_prep):
				for token in pred_word.split():
					if token in gold_word and gold_word not in matched_gold_keywords:
						rel += 1
						broken = True
						matched_gold_keywords.append(gold_word)
						break
				if broken:
					break

		num_relevant += rel

	precision_at_k = num_relevant / num_predictions
	recall_at_k = num_relevant / num_golds
	f1_at_k = (2 * precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
	print("Precision@{:d}: {:.3f}".format(k, precision_at_k))
	print("Recall@{:d}: {:.3f}".format(k, recall_at_k))
	print("F1@{:d}: {:.3f}".format(k, f1_at_k))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--model",
		"-m",
		help="model name/path",
	)
	parser.add_argument(
		"--k",
		"-k",
		help="top k keywords",
	)
	args = parser.parse_args()

	model_name = args.model
	k = int(args.k)

	evaluate(model_name, k)
