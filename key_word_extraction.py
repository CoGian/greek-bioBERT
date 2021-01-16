import json
import unicodedata
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, TFBertModel
import itertools
import spacy


def load_model(model_name):
	"""
	Loads the model and the tokenizer given the model name
	:model_name: the bane of the file tha the model is
	"""
	model = TFBertModel.from_pretrained(model_name)
	tokenizer = BertTokenizer.from_pretrained(model_name)
	# print(model.summary())
	return model, tokenizer


def produce_doc_embeddings(model, tokenizer, text):
	input_ids = tokenizer.encode(
		text,
		return_tensors="tf",
		padding="max_length",
		max_length=512,
		truncation=True)
	doc_embedding = model(input_ids)[1].numpy()
	return doc_embedding


def produce_candidates_embeddings(model, tokenizer, candidates):
	input_ids = tokenizer.batch_encode_plus(
		candidates,
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


def strip_accents_and_uppercase(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn').upper()


def produce_candidates(doc, n_gram_range, pos_model, stop_words):
	pos = {'NOUN', 'ADJ'}
	doc = pos_model(doc)
	doc = " ".join(
		[
			w.text for w in doc
			if w.pos_ in pos
			   and w.text not in stop_words
			   and 2 < len(w.text) < 36])

	# Extract candidate words/phrases
	count = CountVectorizer(ngram_range=n_gram_range).fit([doc])
	candidates = count.get_feature_names()

	return candidates


def extract_keywords(doc, model, tokenizer, pos_model, top_n=5):
	stop_words = prepare_stopwords_list()

	candidates = produce_candidates(doc, (1, 3), pos_model, stop_words)

	doc_embedding = produce_doc_embeddings(model, tokenizer, doc)
	candidate_embeddings = produce_candidates_embeddings(model, tokenizer, candidates)

	# similarities = cosine_similarity(doc_embedding, candidate_embeddings)
	# keywords = [candidates[index] for index in similarities.argsort()[0][-top_n:]]
	keywords = max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n, 20)

	return keywords


def max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates):
	# Calculate distances and extract keywords
	distances = cosine_similarity(doc_embedding, candidate_embeddings)
	distances_candidates = cosine_similarity(candidate_embeddings, candidate_embeddings)

	# Get top_n words as candidates based on cosine similarity
	words_idx = list(distances.argsort()[0][-nr_candidates:])
	words_vals = [candidates[index] for index in words_idx]
	distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

	# Calculate the combination of words that are the least similar to each other
	min_sim = np.inf
	candidate = None
	for combination in itertools.combinations(range(len(words_idx)), top_n):
		sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
		if sim < min_sim:
			candidate = combination
			min_sim = sim

	return [words_vals[idx] for idx in candidate]


if __name__ == '__main__':
	nltk.download('stopwords')
	pos_el = spacy.load("el_core_news_md")
	input_model, input_tokenizer = load_model("greekBERT")

	while True:
		input_doc = input()
		if input_doc == "end":
			break
		output = extract_keywords(input_doc, input_model, input_tokenizer, pos_el)
		print(output)
