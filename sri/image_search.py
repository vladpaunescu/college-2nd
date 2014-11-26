#!/usr/bin/python

import urllib

USER_AGENT = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 " \
             "(KHTML, like Gecko) Chrome/33.0.1750.154 Safari/537.36"

URL_BASE = "https://ajax.googleapis.com/ajax/services/search/images"

import urllib2
import simplejson

def get_page(url):
  req = urllib2.Request(url, None)
  response = urllib2.urlopen(req)
  results = simplejson.load(response)
  print results
  image = results['responseData']['results'][0]['url']
  response = urllib2.urlopen(image)
  html = response.read()
  with open('img.jpg', 'wb') as f:
    f.write(html)



def search_image(words):
  query = urllib.urlencode({'v' : '1.0',
                            "q" : " ".join(words),
                            'rsz' :'8',
                            'start' : '0'})
  print query
  url = URL_BASE + '?' + query
  print url
  get_page(url)

if __name__ == "__main__":
  search_image(['water', 'park'])

