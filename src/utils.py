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
