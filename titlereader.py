import urllib2, re

with open('/home/mogren/sync/code/mogren/rnn-gan/links.txt', 'r') as f:
   for line in f.readlines():
     response = urllib2.urlopen(line)
     data = response.read()
     title = re.findall('<TITLE>(.*)</TITLE>')
     for t in title:
       print t

