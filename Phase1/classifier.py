from Phase1.ir_system import IRSystem
import numpy as np
import pandas as pd


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
        print(
            self.knn(self.train_vector_space[:self.train_size], self.y_train, self.train_vector_space[self.train_size:],
                     1)
            )

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


classifier = Classifier()
