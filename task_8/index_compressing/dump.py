#!/usr/bin/env python3

import sys

from file_storage import FileStorage
from bs4 import BeautifulSoup

def get_text(html):
    soup = BeautifulSoup(html)
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    words = [word.strip().lower() for word in text.split(' ') if word.isalpha()]
    return words

for url, data in FileStorage(sys.argv[1]).items():
    if url.startswith('https://simple.wikipedia.org/wiki/Category:'):
        continue
    text = get_text(data)
    print(url)
    print(len(text))
    for line in text:
        print(line)

