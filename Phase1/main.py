import pickle

from Phase1.ir_system import IRSystem
from Phase1.classifier import Classifier
import numpy as np

ir_sys = IRSystem(["description", "title"], "data/ted_talks.csv", 'data/Persian.xml')
# phase1_classifier = Classifier("data/ted_talks.csv")
# test_classifier = Classifier("data/test.csv")
best_k = 1
best_c = 0.5
phase1_classifier = None
test_classifier = None


def check_language(lang):
    if (not lang == "english") and (not lang == "persian"):
        print("this language " + lang + " is not supported")
        return False
    return True


def check_index(index):
    return index == "positional" or index == "bigram" or index == "stop_words" or index == "structured_documents"


def save_predicted_y_for_docs(docs_y_prediction, method):
    file_name = method + "_" + "y_" + "prediction"
    with open(file_name, 'wb') as pickle_file:
        pickle.dump(docs_y_prediction, pickle_file)
        pickle_file.close()
        print("y's saved for " + str(method) + " method")


while True:
    split_text = input().split()
    if len(split_text) == 0:
        print("not a valid command!")
        continue
    if split_text[0] == "prepare":
        if len(split_text) != 2:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if check_language(lang):
            ir_sys.call_prepare(lang, True)
    elif split_text[0] == "create":
        if split_text[1] == "bigram":
            lang = split_text[2]
            if check_language(lang):
                ir_sys.call_create_bigram(lang)
        elif split_text[1] == "positional":
            lang = split_text[2]
            if check_language(lang):
                ir_sys.call_create_positional(lang)
        elif split_text[1] == "test_classifier":
            test_classifier = Classifier("data/test.csv")
            print("test_classifier created")
        elif split_text[1] == "phase1_classifier":
            phase1_classifier = Classifier("data/ted_talks.csv")
            print("phase1_classifier created")
        else:
            print("not a valid command!")
    elif split_text[0] == "bigram":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if check_language(lang):
            biword = split_text[2]
            if len(biword) != 2:
                print(biword + " is not a biword!")
            else:
                ir_sys.call_bigram(lang, biword)
    elif split_text[0] == "positional":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if check_language(lang):
            term = split_text[2]
            ir_sys.call_positional(lang, term)
    elif split_text[0] == "compress":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        if split_text[1] == "variable_byte":
            lang = split_text[2]
            if check_language(lang):
                if len(ir_sys.positional_index[lang]) != 0:
                    ir_sys.call_compress_variable_byte(lang)
                else:
                    print("Positional Index Is Empty!")

        elif split_text[1] == "gamma_code":
            lang = split_text[2]
            if check_language(lang):
                if len(ir_sys.positional_index[lang]) != 0:
                    ir_sys.call_compress_gamma_code(lang)
                else:
                    print("Positional Index Is Empty!")
        else:
            print("not a valid command!")
    elif split_text[0] == "decompress":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        if split_text[1] == "variable_byte":
            lang = split_text[2]
            if check_language(lang):
                ir_sys.call_decompress_variable_byte(lang)
        elif split_text[1] == "gamma_code":
            lang = split_text[2]
            if check_language(lang):
                ir_sys.call_decompress_gamma_code(lang)
        else:
            print("not a valid command!")
    elif split_text[0] == "tokens":
        if len(split_text) != 2:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if check_language(lang):
            print(ir_sys.document_tokens[lang])
    elif split_text[0] == "stopwords":
        if len(split_text) != 2:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if check_language(lang):
            print(ir_sys.stop_words_dic[lang])
    elif split_text[0] == "terms":
        if len(split_text) != 2:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if check_language(lang):
            print(len(ir_sys.terms[lang]))
    elif split_text[0] == "delete":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if check_language(lang):
            doc_id = int(split_text[2])
            ir_sys.call_delete(lang, doc_id)
    elif split_text[0] == "insert":
        if len(split_text) != 4:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if check_language(lang):
            doc_number = int(split_text[2])
            part_number = int(split_text[3])
            ir_sys.call_insert(lang, doc_number, part_number)
    elif split_text[0] == "save":
        if len(split_text) == 2:
            if split_text[1] == "test_classifier":
                with open('test_classifier_data', 'wb') as pickle_file:
                    pickle.dump(test_classifier, pickle_file)
                    pickle_file.close()
                    print("test classifier create successfully")
            elif split_text[1] == "phase1_classifier":
                with open('phase1_classifier_data', 'wb') as pickle_file:
                    pickle.dump(phase1_classifier, pickle_file)
                    pickle_file.close()
                    print("phase1 classifier create successfully")
        elif len(split_text) == 3:
            type_of_indexing = split_text[1]
            lang = split_text[2]
            if check_language(lang) and check_index(type_of_indexing):
                ir_sys.call_save_index(type_of_indexing, lang)
        else:
            print("not a valid command!")
            continue
    elif split_text[0] == "load":
        if len(split_text) == 2:
            if split_text[1] == "test_classifier":
                try:
                    with open('test_classifier_data', 'rb') as pickle_file:
                        test_classifier = pickle.load(pickle_file)
                        pickle_file.close()
                        print("test classifier loaded successfully")
                except IOError:
                    print("File Not Found!!")
            elif split_text[1] == "phase1_classifier":
                try:
                    with open('phase1_classifier_data', 'rb') as pickle_file:
                        phase1_classifier = pickle.load(pickle_file)
                        pickle_file.close()
                        print("phase1 classifier loaded successfully")
                except IOError:
                    print("File Not Found!!")
        elif len(split_text) == 3:
            type_of_indexing = split_text[1]
            lang = split_text[2]
            if check_language(lang) and check_index(type_of_indexing):
                ir_sys.call_load_index(type_of_indexing, lang)
        else:
            print("not a valid command!")
            continue

    elif split_text[0] == "jaccard":
        if len(split_text) != 4:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if check_language(lang):
            print(ir_sys.jaccard_similarity(
                split_text[2], split_text[3], lang))
    elif split_text[0] == "correction_list":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if check_language(lang):
            term = split_text[2]
            print(ir_sys.call_correction_list(lang, term))
    elif split_text[0] == "edit_distance":
        if len(split_text) != 4:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if check_language(lang):
            print(ir_sys.edit_distance(split_text[2], split_text[3]))
    elif split_text[0] == "query":
        if len(split_text) != 2:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if check_language(lang):
            query = input("Enter Your Query: ")
            correction = ir_sys.query_spell_correction(lang, query)
            ir_sys.process_usual_query(lang, correction)
    elif split_text[0] == "proximity" and split_text[1] == "query":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        lang = split_text[2]
        if check_language(lang):
            proximity_len_of_window = int(
                input("Please Enter Size Of Window: "))
            query = input("Enter Your Query: ")
            correction = ir_sys.query_spell_correction(lang, query)
            ir_sys.process_proximity_query(
                lang, correction, proximity_len_of_window)
    elif split_text[0] == "phase1" and split_text[1] == "query":
        if len(split_text) != 2:
            print("not a valid command!")
            continue
        zone_of_search = int(input("Please Enter Zone Of Search(1 -> Most View ,-1 -> Less View): "))
        query = input("Enter Your Query: ")
        correction = ir_sys.query_spell_correction("english", query)
        ir_sys.process_phase1_query(correction, zone_of_search)
    elif split_text[0] == "exit":
        exit()
    elif split_text[0] == "csv":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if check_language(lang):
            try:
                ir_sys.csv_insert(split_text[2], lang)
            except:
                print("No such csv file found in the path!")
    elif split_text[0] == "xml":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if check_language(lang):
            try:
                ir_sys.xml_insert(split_text[2], lang)
            except:
                print("No such xml file found in the path!")
    elif split_text[0] == "naive_bayes":
        if len(split_text) != 2:
            print("not a valid command!")
            continue
        if split_text[1] == "test":
            print(test_classifier.naive_bayes("english"))
        elif split_text[1] == "phase1":
            docs_y_prediction = phase1_classifier.naive_bayes("english")
            save_predicted_y_for_docs(docs_y_prediction, "naive_bayes")
        else:
            print("not a valid command!")
    elif split_text[0] == "knn":
        if len(split_text) != 3 and len(split_text) != 2:
            print("not a valid command!")
            continue
        if split_text[1] == "test":
            try:
                test_classifier.knn(test_classifier.train_vector_space[:test_classifier.train_size],
                                    test_classifier.y_train,
                                    test_classifier.train_vector_space[test_classifier.train_size:],
                                    int(split_text[2]))
            except:
                print("enter an integer number with a value greater than zero!")
        elif split_text[1] == "phase1":
            try:
                docs_y_prediction = phase1_classifier.knn(
                    phase1_classifier.train_vector_space[:phase1_classifier.train_size],
                    phase1_classifier.y_train,
                    phase1_classifier.train_vector_space[phase1_classifier.train_size:],
                    best_k)
                save_predicted_y_for_docs(docs_y_prediction, "knn")
            except:
                print("enter an integer number with a value greater than zero!")
        else:
            print("not a valid command!")
    elif split_text[0] == "svm":
        if len(split_text) != 3 and len(split_text) != 2:
            print("not a valid command!")
            continue
        if split_text[1] == "test":
            try:
                test_classifier.svm(test_classifier.train_vector_space[:test_classifier.train_size],
                                    test_classifier.y_train,
                                    test_classifier.train_vector_space[test_classifier.train_size:],
                                    float(split_text[2]))
            except:
                print("enter an integer number with a value greater than zero!")
        elif split_text[1] == "phase1":
            try:
                docs_y_prediction = phase1_classifier.svm(
                    phase1_classifier.train_vector_space[:phase1_classifier.train_size],
                    phase1_classifier.y_train,
                    phase1_classifier.train_vector_space[phase1_classifier.train_size:],
                    best_c)
                save_predicted_y_for_docs(docs_y_prediction, "svm")
            except:
                print("enter an integer number with a value greater than zero!")
        else:
            print("not a valid command!")
    elif split_text[0] == "random_forrest":
        if len(split_text) != 2:
            print("not a valid command!")
            continue
        if split_text[1] == "test":
            test_classifier.random_forrest(test_classifier.train_vector_space[:test_classifier.train_size],
                                           test_classifier.y_train,
                                           test_classifier.train_vector_space[test_classifier.train_size:])
        elif split_text[1] == "phase1":
            docs_y_prediction = phase1_classifier.random_forrest(
                phase1_classifier.train_vector_space[:phase1_classifier.train_size],
                phase1_classifier.y_train,
                phase1_classifier.train_vector_space[phase1_classifier.train_size:])
            save_predicted_y_for_docs(docs_y_prediction, "random_forrest")
        else:
            print("not a valid command!")
    elif split_text[0] == "best":
        if len(split_text) != 2:
            print("not a valid command!")
            continue
        if split_text[1] == "k":
            best_k = test_classifier.find_best_k([1, 5, 9], True)
        elif split_text[1] == "c":
            best_c = test_classifier.find_best_c([0.5, 1, 1.5, 2], True)
        else:
            print("not a valid command!")
    elif split_text[0] == "accuracy" or split_text[0] == "precision" or split_text[0] == "recall" or \
            split_text[0] == "f1":
        if len(split_text) != 3 and len(split_text) != 2:
            print("not a valid command!")
            continue
        print_str = split_text[0] + " is: "
        if split_text[1] == "svm":
            y_prediction = test_classifier.svm(test_classifier.train_vector_space[:test_classifier.train_size],
                                               test_classifier.y_train,
                                               test_classifier.train_vector_space[test_classifier.train_size:],
                                               float(split_text[2]))
            print(print_str, test_classifier.find_metric(test_classifier.y_test, y_prediction, split_text[0]))
        elif split_text[1] == "knn":
            y_prediction = test_classifier.knn(test_classifier.train_vector_space[:test_classifier.train_size],
                                               test_classifier.y_train,
                                               test_classifier.train_vector_space[test_classifier.train_size:],
                                               int(split_text[2]))
            print(print_str, test_classifier.find_metric(test_classifier.y_test, y_prediction, split_text[0]))
        elif split_text[1] == "random_forrest":
            y_prediction = test_classifier.random_forrest(
                test_classifier.train_vector_space[:test_classifier.train_size],
                test_classifier.y_train,
                test_classifier.train_vector_space[test_classifier.train_size:])
            print(print_str, test_classifier.find_metric(test_classifier.y_test, y_prediction, split_text[0]))
        elif split_text[1] == "naive_bayes":
            y_prediction = test_classifier.naive_bayes("english")
            print(len(test_classifier.train_vector_space), len(test_classifier.train_vector_space[1]))
            print(len(test_classifier.train_ir_sys.positional_index["english"].keys()))
            print(len(test_classifier.train_ir_sys.structured_documents["english"]))
            print(len(test_classifier.y_test))
            print(print_str, test_classifier.find_metric(test_classifier.y_test, y_prediction, split_text[0]))
    else:
        print("not a valid command!")
        with open('random_forrest_y_prediction', 'rb') as pickle_file:
            kos = pickle.load(pickle_file)
            pickle_file.close()
            for x in kos:
                print(x, end=" ")
        print("kos")
        print(ir_sys.docs_size["english"])
