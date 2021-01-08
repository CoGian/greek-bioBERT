import json
import unicodedata

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, TFBertModel


def load_model(model_name):
	model = TFBertModel.from_pretrained(model_name)
	tokenizer = BertTokenizer.from_pretrained(model_name)
	# print(model.summary())
	return model, tokenizer


def produce_doc_embeddings(model, tokenizer, text):
	input_ids = tokenizer.encode(text,
	                             return_tensors="tf",
	                             padding="max_length",
	                             max_length=512,
	                             truncation=True)
	doc_embedding = model(input_ids)[1].numpy()
	return doc_embedding


def produce_candidates_embeddings(model, tokenizer, candidates):
	input_ids = tokenizer.batch_encode_plus(candidates,
	                                        return_tensors="tf",
	                                        padding="max_length",
	                                        max_length=32,
	                                        truncation=True)["input_ids"]
	candidates_embeddings = model(input_ids)[1].numpy()
	return candidates_embeddings


def prepare_stopwords_list():
	stop_words = nltk.corpus.stopwords.words('greek')
	with open("stopwords-el.json", "r", encoding="utf-8") as fin:
		stop_words2 = json.load(fin)

	stop_words.extend(stop_words2)
	stop_words = set(stop_words)

	return stop_words


def strip_accents_and_lowercase(s):
	return ''.join(c for c in unicodedata.normalize('NFD', s)
	               if unicodedata.category(c) != 'Mn').lower()


def produce_candidates(doc, n_gram_range, stop_words):
	# unaccented_doc = strip_accents_and_lowercase(doc)

	# Extract candidate words/phrases
	count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
	candidates = count.get_feature_names()

	return candidates


def extract_keywords(doc, model):
	stop_words = prepare_stopwords_list()
	stop_words = [strip_accents_and_lowercase(word) for word in stop_words]

	candidates = produce_candidates(doc, (3, 4), stop_words)

	doc_embedding = produce_doc_embeddings(model, tokenizer, doc)
	candidate_embeddings = produce_candidates_embeddings(model, tokenizer, candidates)

	top_n = 5
	similarities = cosine_similarity(doc_embedding, candidate_embeddings)
	keywords = [candidates[index] for index in similarities.argsort()[0][-top_n:]]
	print(keywords[::-1])


if __name__ == '__main__':
	nltk.download('stopwords')
	model, tokenizer = load_model("greekBERT")
	while True:
		doc = input()
		if doc == "end":
			break
		extract_keywords(doc, model)
