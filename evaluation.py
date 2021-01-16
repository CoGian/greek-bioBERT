import json
from key_word_extraction import extract_keywords, load_model
import spacy


def evaluate():
	with open("preprocess/test_articles_dataset.json", "r") as testF:
		test_articles = json.load(testF)

	pos_el = spacy.load("el_core_news_md")
	model, tokenizer = load_model("greekBERT")
	for article in test_articles:
		doc = article['title'] + " " + article["abstract"]
		gold_keywords = article["keywords"]
		pred_keywords = extract_keywords(doc, model, tokenizer, pos_el, top_n=5)
		print(gold_keywords)
		print(pred_keywords)


if __name__ == '__main__':
	evaluate()
