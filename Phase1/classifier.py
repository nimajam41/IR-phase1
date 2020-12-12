from Phase1.ir_system import IRSystem
import numpy as np
import pandas as pd
import random


class Classifier:
    def __init__(self):
        self.train_ir_sys = IRSystem(["description", "title"], "data/train.csv", None)
        self.train_ir_sys.call_prepare("english", False)
        self.train_ir_sys.call_create_positional("english")
        self.train_size = len(self.train_ir_sys.structured_documents["english"])
        self.train_ir_sys.csv_insert("data/test.csv", "english")
        self.train_vector_space = self.create_vector_matrix(self.train_ir_sys, "english")
        self.y_train = self.csv_views("data/train.csv")
        self.y_test = self.csv_views("data/test.csv")
        self.find_best_k([1, 5, 9])

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

    def naive_bayes(self):
        pass

    def knn(self, x_train, y_train, x_test, k):
        dists = np.array(self.docs_distances(x_train, x_test))
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

    def svm(self):
        pass

    def random_forrest(self):
        pass

    def find_metric(self, y_test, y_pred, metric):
        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(len(y_test)):
            if y_test[i] == 1 and y_pred[i] == 1:
                tp += 1
            elif y_test[i] == -1 and y_pred[i] == 1:
                fp += 1
            elif y_test[i] == 1 and y_pred[i] == -1:
                fn += 1
            elif y_test[i] == 1 and y_pred[i] == -1:
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
                x_train_set += [self.train_vector_space[i]]
                y_train_set += [self.y_train[i]]
            else:
                x_validation_set += [self.train_vector_space[i]]
                y_validation_set += [self.y_train[i]]
        return x_train_set, y_train_set, x_validation_set, y_validation_set

    def find_best_k(self, arr):
        max_f1 = -1
        best_k = None
        x_train_set, y_train_set, x_validation_set, y_validation_set = self.make_validation_set()
        for k in arr:
            y_pred = self.knn(x_train_set, y_train_set, x_validation_set, k)
            f1 = self.find_metric(y_validation_set, y_pred, "f1")
            if f1 > max_f1:
                max_f1 = f1
                best_k = k
        print("best k is: ", best_k)
        print("best F1 score is: ", max_f1)
        return k



classifier = Classifier()
