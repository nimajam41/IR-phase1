import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Phase1.ir_system import IRSystem
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv

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


def hierarchical_clustering(data, n):
    model = AgglomerativeClustering(n_clusters=n, linkage="average")
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


def initialize_data(path):
    with open(path, encoding="utf8") as f:
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
    ir_system.collections["persian"].extend([titles, descriptions])
    _, processed_documents, remaining, stopwords = ir_system.prepare_text(ir_system.collections["persian"], "persian",
                                                                          stopwords,
                                                                          False)

    restructured_documents = []
    for doc in processed_documents:
        restructured_documents += [rebuild_doc(doc)]
    return news_sets, restructured_documents


def tf_idf_initializer(restructured_documents):
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(restructured_documents)
    tf_idf_matrix = tf_idf_matrix.toarray()
    return tf_idf_matrix


def show_plot(n_values, ari_values, nmi_values, type, cluster, param):
    plt.plot(n_values, ari_values)
    plt.title("ARI values for different " + param + " values (" + type + " & " + cluster + ")")
    plt.xlabel(param)
    plt.ylabel("ARI")
    plt.show()
    plt.plot(n_values, nmi_values)
    plt.title("NMI values for different " + param + " values (" + type + " & " + cluster + ")")
    plt.xlabel(param)
    plt.ylabel("NMI")
    plt.show()


def write_to_csv(path, news_sets, result):
    with open(path, mode='w', newline='', encoding='utf8') as csv_file:
        fieldnames = ['link', 'cluster']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(news_sets)):
            writer.writerow({'link': str(news_sets[i].link), 'cluster': str(result[i])})


def find_kmeans_metrics(tf_idf_matrix, numbered_clusters):
    kmeans_result = kmeans(tf_idf_matrix, 1)
    best_result = kmeans_result
    best_k = 1
    best_ari = adjusted_rand_score(kmeans_result, numbered_clusters)
    best_nmi = normalized_mutual_info_score(kmeans_result, numbered_clusters)
    k_values = [best_k]
    ari_values = [best_ari]
    nmi_values = [best_nmi]
    for k in range(2, 20):
        kmeans_result = kmeans(tf_idf_matrix, k)
        ari = adjusted_rand_score(kmeans_result, numbered_clusters)
        nmi = normalized_mutual_info_score(kmeans_result, numbered_clusters)
        k_values += [k]
        ari_values += [ari]
        nmi_values += [nmi]
        if nmi > best_nmi:
            best_nmi = nmi
        if ari > best_ari:
            best_k = k
            best_ari = ari
            best_result = kmeans_result
    best_result.dump("data/phase3_outputs/tf-idf-kmeans.dat")
    show_plot(k_values, ari_values, nmi_values, "tf-idf", "K-means", "k")

    print("best k value:", best_k, ",best ARI value:", best_ari, ",best NMI value:", best_nmi)


def tf_idf_kmeans(tf_idf_matrix, numbered_clusters, find_metric):
    if find_metric:
        find_kmeans_metrics(tf_idf_matrix, numbered_clusters)
    else:
        kmeans_result = np.load("data/phase3_outputs/tf-idf-kmeans.dat", allow_pickle=True)
        print("ARI for K-means (best k):", adjusted_rand_score(kmeans_result, numbered_clusters))
        write_to_csv('data/phase3_outputs/tf-idf-kmeans.csv', news_sets, kmeans_result)


def find_gmm_metrics(reduced_matrix, numbered_clusters):
    gmm_result = gmm(reduced_matrix, 1)
    best_result = gmm_result
    best_n = 1
    best_ari = adjusted_rand_score(gmm_result, numbered_clusters)
    best_nmi = normalized_mutual_info_score(gmm_result, numbered_clusters)
    n_values = [best_n]
    ari_values = [best_ari]
    nmi_values = [best_nmi]
    for n in range(2, 20):
        gmm_result = gmm(reduced_matrix, n)
        ari = adjusted_rand_score(gmm_result, numbered_clusters)
        nmi = normalized_mutual_info_score(gmm_result, numbered_clusters)
        n_values += [n]
        ari_values += [ari]
        nmi_values += [nmi]
        if nmi > best_nmi:
            best_nmi = nmi
        if ari > best_ari:
            best_n = n
            best_ari = ari
            best_result = gmm_result
    show_plot(n_values, ari_values, nmi_values, "tf-idf", "GMM", "n")
    best_result.dump("data/phase3_outputs/tf-idf-gmm.dat")

    print("best n value:", best_n, ",best ARI value:", best_ari, ",best NMI value:", best_nmi)


def tf_idf_gmm(tf_idf_matrix, numbered_clusters, find_metric):
    if find_metric:
        reduced_matrix = PCA(n_components=1000).fit_transform(tf_idf_matrix)
        find_gmm_metrics(reduced_matrix, numbered_clusters)
    else:
        gmm_result = np.load("data/phase3_outputs/tf-idf-gmm.dat", allow_pickle=True)
        print("ARI for GMM (best n):", adjusted_rand_score(gmm_result, numbered_clusters))
        write_to_csv('data/phase3_outputs/tf-idf-gmm.csv', news_sets, gmm_result)


def find_hierarchical_clustering_metrics(tf_idf_matrix, numbered_clusters):
    hierarchical_result = hierarchical_clustering(tf_idf_matrix, 1)
    best_result = hierarchical_result
    best_n = 1
    best_ari = adjusted_rand_score(hierarchical_result, numbered_clusters)
    best_nmi = normalized_mutual_info_score(hierarchical_result, numbered_clusters)
    n_values = [best_n]
    ari_values = [best_ari]
    nmi_values = [best_nmi]
    for n in range(2, 31, 2):
        hierarchical_result = hierarchical_clustering(tf_idf_matrix, n)
        ari = adjusted_rand_score(hierarchical_result, numbered_clusters)
        nmi = normalized_mutual_info_score(hierarchical_result, numbered_clusters)
        n_values += [n]
        ari_values += [ari]
        nmi_values += [nmi]
        if nmi > best_nmi:
            best_nmi = nmi
        if ari > best_ari:
            best_n = n
            best_ari = ari
            best_result = hierarchical_result
    show_plot(n_values, ari_values, nmi_values, "tf-idf", "Hierarchical Clustering", "n")
    best_result.dump("data/phase3_outputs/tf-idf-agglomerative.dat")

    print("best n value:", best_n, ",best ARI value:", best_ari, ",best NMI value:", best_nmi)


def tf_idf_hierarchical_clustering(tf_idf_matrix, numbered_clusters, find_metric):
    if find_metric:
        find_hierarchical_clustering_metrics(tf_idf_matrix, numbered_clusters)
    else:
        hierarchical_result = np.load("data/phase3_outputs/tf-idf-agglomerative.dat", allow_pickle=True)
        print("ARI for Hierarchical Clustering (best n):", adjusted_rand_score(hierarchical_result, numbered_clusters))
        write_to_csv('data/phase3_outputs/tf-idf-agglomerative.csv', news_sets, hierarchical_result)


news_sets, restructured_documents = initialize_data("data/hamshahri.json")
real_clusters = find_real_clusters(news_sets)
keys = list(real_clusters.keys())
numbered_clusters = enumerate_clusters(news_sets, keys)
tf_idf_matrix = tf_idf_initializer(restructured_documents)
tf_idf_hierarchical_clustering(tf_idf_matrix, numbered_clusters, False)
