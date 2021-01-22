import json
import unicodedata
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, TFBertModel
import itertools
import spacy
from multi_rake import Rake
import yake
import pytextrank
import tensorflow as tf

def load_model(model_name):
	"""
	Loads the model and the tokenizer given the model name
	:param model_name: the bane of the file tha the model is
	:return: model: huggingface model using TF, tokenizer
	"""
	model = TFBertModel.from_pretrained(model_name)
	tokenizer = BertTokenizer.from_pretrained(model_name)
	# print(model.summary())
	return model, tokenizer


def produce_doc_embeddings(model, tokenizer, text):
	"""
	Produces embeddings for a doc.
	:param model: huggingface model
	:param tokenizer: huggingface tokenizer
	:param text: the that the embeddings are produced
	:return: doc_embeddings: a tf tensor with shape (1,768)
	"""
	input_ids = tokenizer.encode(
		text,
		return_tensors="tf",
		padding="max_length",
		max_length=512,
		truncation=True)
	doc_embedding = model(input_ids)[0]
	doc_embedding = tf.reduce_mean(doc_embedding, axis=[1, 2]).numpy()
	# print(doc_embedding.shape)
	return doc_embedding


def produce_candidates_embeddings(model, tokenizer, candidates):
	"""
	Produces embeddings for candidates in a batch.
	:param model: huggingface model
	:param tokenizer: huggingface tokenizer
	:param candidates: a list of candidates that the embeddings are produced
	:return: doc_embeddings: a tf tensor with shape (len(candidates),768)
	"""
	input_ids = tokenizer.batch_encode_plus(
		candidates,
		return_tensors="tf",
		padding="max_length",
		max_length=32,
		truncation=True)["input_ids"]
	candidates_embeddings = model(input_ids)[0]
	candidates_embeddings = tf.reduce_mean(candidates_embeddings, axis=[1, 2]).numpy()
	# print(candidates_embeddings.shape)
	return candidates_embeddings


def prepare_stopwords_list():
	"""
	Prepares stopwords list
	"""
	stop_words = nltk.corpus.stopwords.words('greek')
	with open("stopwords-el.json", "r", encoding="utf-8") as fin:
		stop_words2 = json.load(fin)

	stop_words.extend(stop_words2)
	stop_words = set(stop_words)

	return stop_words


def strip_accents_and_uppercase(s):
	"""
	Strip accent and uppercase a string
	:param s: a string
	:return: the string without accent and uppercased
	"""
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

	print(doc_embedding.shape)
	print(candidate_embeddings.shape)
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


def test_extraction_with_embeddings():
	nltk.download('stopwords')
	pos_el = spacy.load("el_core_news_md")
	input_model, input_tokenizer = load_model("greekBERT")
	while True:
		input_doc = input()
		if input_doc == "end":
			break
		output = extract_keywords(input_doc, input_model, input_tokenizer, pos_el)
		print(output)


def extract_keywords_RAKE(rake, text, top_n=5):
	keywords = rake.apply(text)
	return [word[0] for word in keywords][:top_n]


def test_extraction_with_RAKE():
	rake = Rake(language_code="el")
	while True:
		input_doc = input()
		if input_doc == "end":
			break
		output = extract_keywords_RAKE(rake, input_doc)
		print(output)


def extract_keywords_YAKE(yake_extractor, text):
	keywords = yake_extractor.extract_keywords(text)
	return [word[1] for word in keywords]


def test_extraction_with_YAKE():
	yake_extractor = yake.KeywordExtractor(lan="el", top=5)
	while True:
		input_doc = input()
		if input_doc == "end":
			break
		output = extract_keywords_YAKE(yake_extractor, input_doc)
		print(output)


def extract_keywords_TEXTRANK(model, doc, top_n=5):
	doc = model(doc)

	return [p.text for p in doc._.phrases if 1 <= len(p.text.split()) <= 3][:top_n]


def test_extraction_with_TEXTRANK():
	tr = pytextrank.TextRank()
	pos_el = spacy.load("el_core_news_md")

	pos_el.add_pipe(tr.PipelineComponent, name="textrank", last=True)

	while True:
		input_doc = input()
		if input_doc == "end":
			break
		output = extract_keywords_TEXTRANK(pos_el, input_doc, 5)
		print(output)


if __name__ == '__main__':
	test_extraction_with_embeddings()
