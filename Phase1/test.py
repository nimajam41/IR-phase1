from Phase1.ir_system import IRSystem
import numpy as np
import pandas as pd
import random
from sklearn.svm import SVC
import math


class Classifier:
    def __init__(self, path):
        self.train_ir_sys = IRSystem(["description", "title"], "data/train.csv", None)
        self.train_ir_sys.call_prepare("english", False)
        self.train_ir_sys.call_create_positional("english")
        self.train_size = len(self.train_ir_sys.structured_documents["english"])
        self.train_ir_sys.csv_insert(path, "english")
        # self.train_vector_space = self.create_vector_matrix(self.train_ir_sys, "english")
        #        self.vector_space = self.train_ir_sys.use_ntn("english")
        #        print("vector space created successfully")
        self.y_train = self.csv_views("data/train.csv")
        self.y_test = None
        if path == "data/test.csv":
            self.y_test = self.csv_views("data/test.csv")
        # self.dists = self.documents_distances(self.vector_space[:self.train_size], self.vector_space[self.train_size + 1:])
        # print("distances computed successfully")
        #        print(self.vector_space[0])
        #        print(self.knn(self.vector_space[:self.train_size], self.y_train, self.vector_space[self.train_size:], 1))
        #        self.find_best_k([1, 5, 9], True)

        self.naive_bayes("english")

    def csv_views(self, path):
        df = pd.read_csv(path, usecols=["views"])
        result = []
        for i in range(len(df)):
            result += [df.iloc[i]["views"]]
        return result

    def token_to_number(self, ir_sys, lang):
        return {token: ind for ind, token in enumerate(ir_sys.positional_index[lang].keys())}

    def create_vector_matrix(self, ir_sys, lang):
        vector = np.zeros([len(ir_sys.structured_documents[lang]), len(ir_sys.positional_index[lang].keys())])
        tokenized_vector = ir_sys.use_ntn(lang)
        for doc_id in range(len(tokenized_vector)):
            for term in tokenized_vector[doc_id].keys():
                vector[doc_id][self.token_to_number(ir_sys, lang)[term]] = tokenized_vector[doc_id][term]
        return vector

    def naive_bayes(self, lang):
        flag_counter = {"positive_docs": 0, "negative_docs": 0, "positive_terms": 0, "negative_terms": 0}
        words = dict()
        self.naive_bayes_train(flag_counter, words, lang)
        y_predicted = self.naive_bayes_test(flag_counter, words, lang)
        print(self.find_metric(self.y_test, y_predicted, "precision"))
        print(self.find_metric(self.y_test, y_predicted, "recall"))
        print(self.find_metric(self.y_test, y_predicted, "accuracy"))
        print(self.find_metric(self.y_test, y_predicted, "f1"))
        return y_predicted

    def naive_bayes_train(self, flag_counter, words, lang):
        for docID in range(self.train_size):

            if self.y_train[docID] == 1:
                flag = "positive"
                flag_counter["positive_docs"] += 1
            else:
                flag = "negative"
                flag_counter["negative_docs"] += 1

            for col in range(2):
                for word in self.train_ir_sys.structured_documents[lang][docID][col]:
                    if word not in words.keys():
                        words[word] = {"positive": 0, "negative": 0}
                    words[word][flag] += 1
                    flag_counter[str(flag + "_terms")] += 1

    def naive_bayes_test(self, flag_counter, words, lang):
        y_predicted = []
        p_positive_doc = flag_counter["positive_docs"] / self.train_size
        p_negative_doc = 1 - p_positive_doc
        for docID in range(len(self.y_test)):
            p_positive = 0
            p_negative = 0
            for col in range(2):
                for word in self.train_ir_sys.structured_documents[lang][self.train_size + docID][col]:
                    if word in words.keys():
                        p_positive += math.log(
                            (words[word]["positive"] + 1) / (flag_counter["positive_terms"] + len(words)))
                        p_negative += math.log(
                            (words[word]["negative"] + 1) / (flag_counter["negative_terms"] + len(words)))
                    else:  # new word in test Doc
                        p_positive += math.log(1 / (flag_counter["positive_terms"] + len(words)))
                        p_negative += math.log(1 / (flag_counter["negative_terms"] + len(words)))
            if math.log(p_positive_doc) + p_positive >= math.log(p_negative_doc) + p_negative:
                y_predicted.append(1)
            else:
                y_predicted.append(-1)
        return y_predicted

    def knn(self, x_train, y_train, x_test, k):
        dists = np.array(self.documents_distances(x_train, x_test))
        y_pred = []
        for doc_id in range(len(x_test)):
            test_dist = dists[:, doc_id]
            sorted_nearest_docs = [s[0] for s in sorted(enumerate(test_dist), key=lambda a: a[1])]
            top_k = sorted_nearest_docs[:k]
            pos_views, neg_views = 0, 0
            for train_id in top_k:
                if y_train[train_id] == 1:
                    pos_views += 1
                else:
                    neg_views += 1
            if pos_views > neg_views:
                y_pred += [1]
            else:
                y_pred += [-1]
        return y_pred

    def docs_distances(self, first, second):
        result = []
        for i in range(len(first)):
            result += [[]]
            for j in range(len(second)):
                dist = np.linalg.norm(first[i] - second[j])
                result[i] += [dist]
        return result

    def svm(self, x_train, y_train, x_test, c_parameter):
        svclassifier = SVC(kernel='linear', C=c_parameter)
        classifier = svclassifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        return y_pred

    def random_forrest(self):
        pass

    def two_doc_distance(self, first, second, doc1, doc2):
        list1 = first[doc1].keys()
        list2 = second[doc2].keys()
        intersect = set(list1) & set(list2)
        delta = set(list1) ^ set(list2)
        result = 0
        for term in intersect:
            result += ((first[doc1][term] - second[doc2][term]) ** 2)
        for term in delta:
            if term in first[doc1].keys():
                result += ((first[doc1][term]) ** 2)
            else:
                result += ((second[doc2][term]) ** 2)
        return math.sqrt(result)

    def documents_distances(self, first, second):
        result = []
        for i in range(len(first)):
            result += [[]]
            for j in range(len(second)):
                result[i] += [self.two_doc_distance(first, second, i, j)]
        return result

    def find_metric(self, y_test, y_pred, metric):
        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(len(y_test)):
            if y_test[i] == 1 and y_pred[i] == 1:
                tp += 1
            elif y_test[i] == -1 and y_pred[i] == 1:
                fp += 1
            elif y_test[i] == 1 and y_pred[i] == -1:
                fn += 1
            elif y_test[i] == -1 and y_pred[i] == -1:
                tn += 1
        if metric == "precision":
            return tp / (tp + fp)
        if metric == "recall":
            return tp / (tp + fn)
        if metric == "accuracy":
            return (tp + tn) / (tp + fp + tn + fn)
        if metric == "f1":
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * precision * recall / (precision + recall)
        else:
            return None

    def make_validation_set(self):
        x_train_set = []
        x_validation_set = []
        y_train_set = []
        y_validation_set = []
        for i in range(self.train_size):
            m = random.uniform(0, 1)
            if m < 0.9:
                x_train_set += [self.vector_space[i]]
                y_train_set += [self.y_train[i]]
            else:
                x_validation_set += [self.vector_space[i]]
                y_validation_set += [self.y_train[i]]
        return x_train_set, y_train_set, x_validation_set, y_validation_set

    def find_best_k(self, arr, print_flag):
        max_f1 = -1
        best_k = None
        x_train_set, y_train_set, x_validation_set, y_validation_set = self.make_validation_set()
        for k in arr:
            y_pred = self.knn(x_train_set, y_train_set, x_validation_set, k)
            f1 = self.find_metric(y_validation_set, y_pred, "f1")
            precision = self.find_metric(y_validation_set, y_pred, "precision")
            recall = self.find_metric(y_validation_set, y_pred, "recall")
            accuracy = self.find_metric(y_validation_set, y_pred, "accuracy")
            if print_flag:
                print("metrics for k = ", k)
                print("f1 score = ", f1)
                print("precision = ", precision)
                print("recall = ", recall)
                print("accuracy = ", accuracy)
                print()
            if f1 > max_f1:
                max_f1 = f1
                best_k = k
        if print_flag:
            print("best k is: ", best_k)
            print("best f1 score is: ", max_f1)
        return best_k


test = Classifier("data/test.csv")
y_pred = test.naive_bayes("english")
print(len(test.train_ir_sys.positional_index["english"].keys()))
print(len(test.train_ir_sys.structured_documents["english"]))
print(len(test.y_test))
print(test.find_metric(test.y_test, y_pred, "accuracy"))
print(y_pred)