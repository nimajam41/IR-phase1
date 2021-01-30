import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from Phase1.ir_system import IRSystem


ir_system = IRSystem(None, None, None)


class News:
    def __init__(self, title, summary, link, tags, id):
        self.title = title
        self.summary = summary
        self.link = link
        self.tags = tags
        self.id = id


def rebuild_doc(doc):
    s = ""
    for arr in doc:
        for word in arr:
            s += (word + " ")
    return s[:len(s) - 1]


with open('data/hamshahri.json', encoding="utf8") as f:
    dataset = json.load(f)

titles = []
descriptions = []
news_sets = []
id = 1
for data in dataset:
    news = News(data['title'], data['summary'], data['link'], data['tags'], id)
    news_sets += [news]
    new_title = re.sub("[\{].*?[\}]", "", data['title'])
    new_title = re.sub(r'[0-9]', "", new_title)
    new_new_title = re.sub(r'[۰-۹]', "", new_title)
    titles += [new_title]
    new_summary = re.sub("[\{].*?[\}]", "", data['summary'])
    new_summary = re.sub(r'[0-9]', "", new_summary)
    new_summary = re.sub(r'[۰-۹]', "", new_summary)
    descriptions += [new_summary]
    id += 1

stopwords = []
# print(titles)
ir_system.collections["persian"].extend([titles, descriptions])
_, processed_documents, remaining, stopwords = ir_system.prepare_text(ir_system.collections["persian"], "persian",
                                                                      stopwords,
                                                                      False)
restructured_documents = []
for doc in processed_documents:
    restructured_documents += [rebuild_doc(doc)]
vectorizer = TfidfVectorizer()
tf_idf_matrix = vectorizer.fit_transform(restructured_documents)

corpus = ["car cat dog", "dog lion"]

