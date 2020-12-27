import os
import re
import json
import spacy
import random


def process_text(text):
    proc_text = text.lower()

    proc_text = re.sub("\n", " ", proc_text)
    proc_text = re.sub("\t", " ", proc_text)
    proc_text = re.sub('\r', '', proc_text)
    proc_text = re.sub('\$.*?\$', '', proc_text)
    proc_text = re.sub('<.*?>', '', proc_text)
    proc_text = re.sub('\s+', ' ', proc_text)

    return proc_text


def main():
    articles = []

    with open("iatrolexi_abstracts_with_keywords.json", 'r', encoding="utf-8") as fin:
        data = json.load(fin)
    articles.extend(data)

    with open("../bioscrape/mendet.json", 'r', encoding="utf-8") as fin:
        data = json.load(fin)
    articles.extend(data)

    with open("../bioscrape/sebe.json", 'r', encoding="utf-8") as fin:
        data = json.load(fin)
    articles.extend(data)

    nlp = spacy.load('el')

    sentences_dataset = []
    articles_dataset = []
    test_articles_dataset = []

    random.seed(44)
    random.shuffle(articles)
    test_articles = articles[:150]
    articles = articles[150:]

    for article in articles:
        abstract = process_text(article['abstract'])
        keywords = process_text(article['keywords'])

        articles_dataset.append({
            "abstract": abstract,
            "keywords": keywords
        })

        doc = nlp(abstract)
        sentences = list(doc.sents)
        sentences = [str(sent) for sent in sentences]
        sentences_dataset.extend(sentences)

    with open('articles_dataset.json', 'w', encoding='utf8') as fout:
        json.dump(articles_dataset, fout, ensure_ascii=False, indent=2)

    with open('sentences_dataset.json', 'w', encoding='utf8') as fout:
        json.dump(sentences_dataset, fout, ensure_ascii=False, indent=2)

    for article in test_articles:
        abstract = process_text(article['abstract'])
        keywords = process_text(article['keywords'])

        test_articles_dataset.append({
            "abstract": abstract,
            "keywords": keywords
        })
    with open('test_articles_dataset.json', 'w', encoding='utf8') as fout:
        json.dump(test_articles_dataset, fout, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
