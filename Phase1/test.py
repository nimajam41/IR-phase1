import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Phase1.ir_system import IRSystem
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn import metrics


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


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


def kmeans(data, k):
    model = KMeans(n_clusters=k, random_state=0)
    return model.fit_predict(data)


def gmm(data, n):
    model = GaussianMixture(n)
    return model.fit_predict(data)


def find_real_clusters(news_sets):
    real_clusters = dict()
    for news in news_sets:
        main_subject = news.tags[0].split(">")[0].split(" ")[0]
        if main_subject not in real_clusters.keys():
            real_clusters[main_subject] = [news.id]
        else:
            real_clusters[main_subject] += [news.id]
    return real_clusters


def enumerate_clusters(news_sets, keys):
    arr = []
    keys_dict = dict()
    for i in range(len(keys)):
        keys_dict[keys[i]] = i
    for news in news_sets:
        main_subject = news.tags[0].split(">")[0].split(" ")[0]
        arr += [keys_dict[main_subject]]
    return np.array(arr)


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
# print(restructured_documents)
vectorizer = TfidfVectorizer()
tf_idf_matrix = vectorizer.fit_transform(restructured_documents)
tf_idf_matrix = tf_idf_matrix.toarray()
real_clusters = find_real_clusters(news_sets)
keys = list(real_clusters.keys())
numbered_clusters = enumerate_clusters(news_sets, keys)
kmeans_result = gmm(tf_idf_matrix, 4)
