#!/usr/bin/python

import urllib

USER_AGENT = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 " \
             "(KHTML, like Gecko) Chrome/33.0.1750.154 Safari/537.36"

URL_BASE = "https://ajax.googleapis.com/ajax/services/search/images"

import urllib2
import simplejson

from cache import *
import shutil


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

    return html


def search_image(words):

    print '>>>search image for: ', words
    img = getImage(words)
    if img is not None:
        print 'found image in cache: ', img
        shutil.copyfile(img, 'img.jpg')
        return img

    query = urllib.urlencode({'v': '1.0',
                              "q": " ".join(words),
                              'rsz': '8',
                              'start': '0'})
    print query
    url = URL_BASE + '?' + query
    print url

    ret = get_page(url)
    name = "_".join(words) + '.jpg'
    addToCache(" ".join(words), name)
    shutil.copyfile('img.jpg', './images/' + name)
    return ret

if __name__ == "__main__":
    search_image(['juice', 'lemon'])

