import scrapy
import string


class MednetSpider(scrapy.Spider):
	name = "mednet"
	start_urls = ["http://www.mednet.gr/archives/older-gr.html"]

	def parse(self, response, **kwargs):

		all_hrefs = response.xpath('//a/@href').extract()
		issues_hrefs = []
		for href in all_hrefs:
			if "contents" in href:
				issues_hrefs.append(href)

		for href in issues_hrefs:
			yield scrapy.Request("http://www.mednet.gr/archives/" + href, callback=self.parse_issues)

	def parse_issues(self, response):
		all_hrefs = response.xpath('//a/@href').extract()
		abstract_hrefs = []
		for href in all_hrefs:
			if "per" in href:
				abstract_hrefs.append(href)

		for href in abstract_hrefs:
			yield scrapy.Request("http://www.mednet.gr/archives/" + href, callback=self.parse_abstracts)

	def parse_abstracts(self, response):
		title = response.css('span.HeadTitle::text').getall()
		abstract = response.css('p.AbsText::text').getall()
		labels = response.css('span.AbsText::text').getall()

		if len(title) == 0:
			title = response.css('font.HeadTitle::text').getall()

		if len(title) == 0:
			title = response.css('p.HeadTitle::text').getall()

		if len(abstract) == 0 or len(abstract[0].split()) < 10:
			if len(labels) >= 2 :
				abstract = labels[0:-1]
				labels = labels[-1:]


		title = " ".join(title)
		abstract = " ".join(abstract)
		labels = " ".join(labels)

		title = title.replace("\r", " ")
		title = title.replace("\n", " ")

		abstract = abstract.replace("\r", " ")
		abstract = abstract.replace("\n", " ")

		labels = labels.replace("\r", " ")
		labels = labels.replace("\n", " ")

		if "�" not in abstract and "�" not in title and "�" not in title:
			yield {
				'title': title,
				'abstract': abstract,
				'labels': labels
			}
