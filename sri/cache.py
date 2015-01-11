import nltk
from nltk.corpus import wordnet
from pickle import dump, load
from os import listdir
from os.path import isfile, join


def buildCache():
    d = {}
    
    mypath = './images/'
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

    for f in onlyfiles:
        lst = f.split(".")
        d[lst[0]] = f
    dump(d, open( "cache.p", "wb" ))


def addToCache(word, image):
    d = load( open( "cache.p", "rb" ) )

    d[word] = image

    dump(d, open( "cache.p", "wb" ))


def displayCache():
    d = load( open( "cache.p", "rb" ) )

    print d


def getSynonims(word):
    syns = wordnet.synsets(word)
    
    if syns is None or syns == []:
        syns.append(word)
        return syns


    synonims = [syn.name().split('.')[0] for syn in syns if syn.pos() == 'n']
    synonims = set(synonims)
    synonims.discard(word)
   
    synonims = list(synonims)
    synonims.insert(0, word)

    return synonims

def getImage(words):
    d = load( open( "cache.p", "rb" ) )

    for word in words:
        if word in d:
            return './images/' + d[word]

    for word in words:
        syns = getSynonims(word)
        for syn in syns:
            if syn in d:
                return './images/' + d[syn]
        

#syns = getSynonims('sleep')
#print syns

addToCache('water', 'water.jpg')
#buildCache()
displayCache()
#addToCache('bed', 'bed.jpg')
#displayCache()