from Phase1.ir_system import IRSystem


class Classifier:
    def __init__(self):
        train_ir_sys = IRSystem(["description", "title"], "data/train.csv", None)
        train_ir_sys.call_prepare("english", False)
        train_ir_sys.call_create_positional("english")
        print(train_ir_sys.use_ntn("english")[0])

    def naive_bayes(self):
        pass

    def knn(self):
        pass

    def svm(self):
        pass

    def random_forrest(self):
        pass


classifier = Classifier()
