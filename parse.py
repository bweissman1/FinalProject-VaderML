import newspaper
import requests
from newspaper import Article
##
##input=str(raw_input('Enter your url here and wait for awesomeness'))
##article = Article(url)
##article.download()
##article.html
##article.parse()
##article.authors
##article.publish_date
##article.text
##article.top_image
##article.nlp()
##article.keywords
##article.summary
##
##news_dict = {'cnn':'http://cnn.com','alternet' :'https://www.alternet.org/',
## 'msnbc':'http://www.msnbc.com', 'fox':'http://www.foxnews.com',
## 'breitbart':'http://www.breitbart.com/news','bbc':'http://www.bbc.com/news',
## 'aljazeera':'https://www.aljazeera.com/','wsj':'https://www.wsj.com/',
## 'newyorktimes':'https://www.nytimes.com/','washingtonpost':'https://www.washingtonpost.com/'}
##
##artic = str(input('Enter the source: '))
##news = newspaper.build(news_dict[artic])
##for category in news.category_urls():
##    print (category)

url = str(input('Enter your url here for awesomeness'))
article = Article(url)
article.download()
article.parse()
print (article.text)
print (article.nlp())

#article.nlp()
##article.summary
