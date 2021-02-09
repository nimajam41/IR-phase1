import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Phase1.ir_system import IRSystem
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import csv
from scipy.cluster.hierarchy import dendrogram
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

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


def hierarchical_clustering(data, n, linkage):
    model = AgglomerativeClustering(n_clusters=n, linkage=linkage)
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
    docs_terms = []
    news_sets = []
    id = 1
    for data in dataset:
        news = News(data['title'], data['summary'], data['link'], data['tags'], id)
        news_sets += [news]
        new_title = re.sub("[\{].*?[\}]", "", data['title'])
        new_title = re.sub(r'[0-9]', "", new_title)
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
        l = []
        for arr in doc:
            for word in arr:
                l += [word]
        docs_terms += [l]
    return news_sets, restructured_documents, docs_terms


def visualize(matrix, label, title):
    plt.figure(figsize=(8,8))
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(matrix)
    a = principalComponents.T
    a = pd.DataFrame({'x': a[0], 'y': a[1], 'z': label})
    sns.scatterplot(data=a, x="x", y="y", hue='z', palette="deep")
    plt.title(title)
    plt.show()


def tf_idf_initializer(restructured_documents):
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(restructured_documents)
    tf_idf_matrix = tf_idf_matrix.toarray()
    return tf_idf_matrix


def w2v_initializer(docs_terms):
    model = Word2Vec(docs_terms, workers=8, iter=100)
    w2v = []
    for doc in docs_terms:
        res = np.concatenate(
            [np.expand_dims(np.array(model.wv[term]), axis=1) for term in doc if term in model.wv.vocab],
            axis=1)
        mean_vector = res.mean(axis=1)
        w2v += [mean_vector]
    return np.array(w2v)


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


def show_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child in merge:
            if child < n_samples:
                current_count += 1
            else:
                current_count += counts[child - n_samples]
        counts[i] = current_count

    linkage = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    dendrogram(linkage, **kwargs)


def write_to_csv(path, news_sets, result):
    with open(path, mode='w', newline='', encoding='utf8') as csv_file:
        fieldnames = ['link', 'cluster']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(news_sets)):
            writer.writerow({'link': str(news_sets[i].link), 'cluster': str(result[i])})


def find_kmeans_metrics(matrix, numbered_clusters, type):
    kmeans_result = kmeans(matrix, 1)
    best_result = kmeans_result
    best_k = 1
    best_ari = adjusted_rand_score(kmeans_result, numbered_clusters)
    best_nmi = normalized_mutual_info_score(kmeans_result, numbered_clusters)
    k_values = [best_k]
    ari_values = [best_ari]
    nmi_values = [best_nmi]
    for k in range(2, 20):
        kmeans_result = kmeans(matrix, k)
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

    best_result.dump("data/phase3_outputs/" + type + "-kmeans.dat")
    show_plot(k_values, ari_values, nmi_values, type, "K-means", "k")

    print("best k value:", best_k, ",best ARI value:", best_ari, ",best NMI value:", best_nmi)


def tf_idf_kmeans(tf_idf_matrix, numbered_clusters, find_metric):
    if find_metric:
        find_kmeans_metrics(tf_idf_matrix, numbered_clusters, "tf-idf")
    else:
        kmeans_result = np.load("data/phase3_outputs/tf-idf-kmeans.dat", allow_pickle=True)
        print("ARI for K-means (best k) for tf-idf:", adjusted_rand_score(kmeans_result, numbered_clusters))
        write_to_csv('data/phase3_outputs/tf-idf-kmeans.csv', news_sets, kmeans_result)
        visualize(tf_idf_matrix, kmeans_result, "tf_idf_kmeans_result")
        visualize(tf_idf_matrix, numbered_clusters, "tf_idf_real_clustered_data")


def w2v_kmeans(w2v_matrix, numbered_clusters, find_metric):
    if find_metric:
        find_kmeans_metrics(w2v_matrix, numbered_clusters, "w2v")
    else:
        kmeans_result = np.load("data/phase3_outputs/w2v-kmeans.dat", allow_pickle=True)
        print("ARI for K-means (best k) for w2v:", adjusted_rand_score(kmeans_result, numbered_clusters))
        write_to_csv('data/phase3_outputs/w2v-kmeans.csv', news_sets, kmeans_result)
        visualize(w2v_matrix, kmeans_result, "word2vec_kmeans_result")
        visualize(w2v_matrix, numbered_clusters, "word2vec_real_clustered_data")

def find_gmm_metrics(reduced_matrix, numbered_clusters, type):
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
    show_plot(n_values, ari_values, nmi_values, type, "GMM", "n")
    best_result.dump("data/phase3_outputs/" + type + "-gmm.dat")

    print("best n value:", best_n, ",best ARI value:", best_ari, ",best NMI value:", best_nmi)


def tf_idf_gmm(tf_idf_matrix, numbered_clusters, find_metric):
    if find_metric:
        reduced_matrix = PCA(n_components=1000).fit_transform(tf_idf_matrix)
        find_gmm_metrics(reduced_matrix, numbered_clusters, "tf-idf")
    else:
        gmm_result = np.load("data/phase3_outputs/tf-idf-gmm.dat", allow_pickle=True)
        print("ARI for GMM (best n) for tf-idf:", adjusted_rand_score(gmm_result, numbered_clusters))
        write_to_csv('data/phase3_outputs/tf-idf-gmm.csv', news_sets, gmm_result)
        visualize(tf_idf_matrix, gmm_result, "tf_idf_gmm_result")
        visualize(tf_idf_matrix, numbered_clusters, "tf_idf_real_clustered_data")

def w2v_gmm(w2v_matrix, numbered_clusters, find_metric):
    if find_metric:
        find_gmm_metrics(w2v_matrix, numbered_clusters, "w2v")
    else:
        gmm_result = np.load("data/phase3_outputs/w2v-gmm.dat", allow_pickle=True)
        print("ARI for GMM (best n) for w2v:", adjusted_rand_score(gmm_result, numbered_clusters))
        write_to_csv('data/phase3_outputs/w2v-gmm.csv', news_sets, gmm_result)
        visualize(w2v_matrix, gmm_result, "word2vec_gmm_result")
        visualize(w2v_matrix, numbered_clusters, "word2vec_real_clustered_data")

def find_hierarchical_clustering_metrics(matrix, numbered_clusters, type, linkage):
    hierarchical_result = hierarchical_clustering(matrix, 1, linkage)
    best_result = hierarchical_result
    best_n = 1
    best_ari = adjusted_rand_score(hierarchical_result, numbered_clusters)
    best_nmi = normalized_mutual_info_score(hierarchical_result, numbered_clusters)
    n_values = [best_n]
    ari_values = [best_ari]
    nmi_values = [best_nmi]
    for n in range(2, 31, 2):
        hierarchical_result = hierarchical_clustering(matrix, n, linkage)
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
    show_plot(n_values, ari_values, nmi_values, type, "Hierarchical Clustering", "n")
    best_result.dump("data/phase3_outputs/" + type + "-agglomerative.dat")

    print("best n value:", best_n, ",best ARI value:", best_ari, ",best NMI value:", best_nmi)


def tf_idf_hierarchical_clustering(tf_idf_matrix, numbered_clusters, find_metric):
    if find_metric:
        find_hierarchical_clustering_metrics(tf_idf_matrix, numbered_clusters, "tf-idf", "average")
    else:
        hierarchical_result = np.load("data/phase3_outputs/tf-idf-agglomerative.dat", allow_pickle=True)
        print("ARI for Hierarchical Clustering (best n) for tf-idf:",
              adjusted_rand_score(hierarchical_result, numbered_clusters))
        write_to_csv('data/phase3_outputs/tf-idf-agglomerative.csv', news_sets, hierarchical_result)
        visualize(tf_idf_matrix, hierarchical_result, "tf_idf_hierarchical_result")
        visualize(tf_idf_matrix, numbered_clusters, "tf_idf_real_clustered_data")


def w2v_hierarchical_clustering(w2v_matrix, numbered_clusters, find_metric):
    if find_metric:
        find_hierarchical_clustering_metrics(w2v_matrix, numbered_clusters, "w2v", "ward")
    else:
        hierarchical_result = np.load("data/phase3_outputs/w2v-agglomerative.dat", allow_pickle=True)
        print("ARI for Hierarchical Clustering (best n) for w2v:",
              adjusted_rand_score(hierarchical_result, numbered_clusters))
        write_to_csv('data/phase3_outputs/w2v-agglomerative.csv', news_sets, hierarchical_result)
        visualize(w2v_matrix, hierarchical_result, "word2vec_hierarchical_result")
        visualize(w2v_matrix, numbered_clusters, "word2vec_real_clustered_data")

news_sets, restructured_documents, docs_terms = initialize_data("data/hamshahri.json")
real_clusters = find_real_clusters(news_sets)
keys = list(real_clusters.keys())
numbered_clusters = enumerate_clusters(news_sets, keys)
tf_idf_matrix = tf_idf_initializer(restructured_documents)
w2v_matrix = w2v_initializer(docs_terms)
w2v_gmm(w2v_matrix, numbered_clusters, False)
