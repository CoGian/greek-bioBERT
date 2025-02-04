import scrapy
import string


class MednetSpider(scrapy.Spider):
    name = "sebe"
    start_urls = [
        "http://www.sebe.gr/%cf%80%ce%b5%cf%81%ce%b9%ce%bf%ce%b4%ce%b9%ce%ba%ce%bf-%cf%83%cf%84%ce%bf%ce%bc%ce%b1/%cf%84%ce%b5%cf%85%cf%87%ce%b7-%cf%80%ce%b5%cf%81%ce%b9%ce%bf%ce%b4%ce%b9%ce%ba%ce%bf%cf%85/"]

    def parse(self, response, **kwargs):

        all_hrefs = response.xpath('//a/@href').extract()
        issues_hrefs = []
        for href in all_hrefs:
            if "%cf%84%ce%b5%cf%8d%cf%87%ce%bf%cf%82" in href:
                issues_hrefs.append(href)

        for href in issues_hrefs:
            yield scrapy.Request(href, callback=self.parse_issues)

    def parse_issues(self, response):
        abstract_hrefs = response.css('.mpcth-read-more').xpath('@href').extract()

        for href in abstract_hrefs:
            yield scrapy.Request(href, callback=self.parse_abstracts)

    def parse_abstracts(self, response):
        title = response.css('span.mpcth-color-main-border::text').getall()

        title = [item for item in title if '\t' in item]
        p_list = response.css('p::text').getall()

        abstract = p_list[:-1]
        keywords = p_list[-1]

        title = " ".join(title)
        abstract = " ".join(abstract)

        title = title.replace("\r", " ")
        title = title.replace("\n", " ")
        title = title.replace("\t", "")
        title = title.replace("\xad", "")
        title = title.replace("\xa0", "")

        abstract = abstract.replace("\r", " ")
        abstract = abstract.replace("\n", " ")
        abstract = abstract.replace("\xad", "")
        abstract = abstract.replace("\xa0", "")

        keywords = keywords.replace("\r", " ")
        keywords = keywords.replace("\n", " ")
        keywords = keywords.replace("\xad", "")
        keywords = keywords.replace("\xa0", "")

        if abstract != '' and keywords != '':
            yield {
                'title': title,
                'abstract': abstract,
                'keywords': keywords
            }
