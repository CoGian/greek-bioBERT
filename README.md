# keyword-extraction-with-greekBERT
# ABSTRACT
BERT has recently emerged as a very effective language representation model. BERT is conceptually simple and empirically powerful. In this paper, we gather a novel dataset from biomedical greek websites and produce word embeddings using the fine tuned BERT model for the keyword extraction from the specific domain of greek biomedical texts. Experiments and evaluation conducted on already existing unsupervised keyword extraction methods compared to our approach shows that BERT can learn from greek biomedical texts. Code is publicly available at: https://github.com/CoGian/keyword-extraction-with-greekBERT  and our fine tuned model is available at: https://drive.google.com/drive/folders/1xjzB9e7e-sZT7Qy3BnRyACqRgo7YXR8g?usp=sharing.

## Scraping biomedical sites
To scrape biomedical sites 

`cd bioscrape`

`scrapy crawl name_of_the_spider -o name.json`

e.g. for mednet: 
`scrapy crawl mednet -o mednet.json`

## Extraction 
Useful info can be found at keyword_extraction.ipynb
    
     
