import math
import pickle
import sys
import pandas as pd
import xml.etree.ElementTree as ET
from nltk import word_tokenize
from collections import Counter
from nltk.stem import SnowballStemmer
from hazm import *
import matplotlib.pyplot as plt
import re


class IRSystem:
    def __init__(self):
        self.collections = {"english": [], "persian": []}
        self.document_tokens = {"english": [], "persian": []}
        self.structured_documents = {"english": [], "persian": []}
        self.terms = {"english": [], "persian": []}
        self.stop_words_dic = {"english": [], "persian": []}
        self.bigram_index = {"english": dict(), "persian": dict()}
        self.positional_index = {"english": dict(), "persian": dict()}
        self.vb_positional_index = {"english": dict(), "persian": dict()}
        self.gamma_positional_index = {"english": dict(), "persian": dict()}
        self.docs_size = {"english": 0, "persian": 0}
        self.deleted_documents = {"english": [], "persian": []}
        self.initialize_english()
        self.initialize_persian()

    def initialize_english(self):
        english_columns = ["description", "title"]
        english_df = pd.read_csv(
            "data/ted_talks.csv", usecols=english_columns)
        x = len(english_df)
        for i in range(x):
            title = english_df.iloc[i]["title"]
            description = english_df.iloc[i]["description"]
            self.collections["english"] += [[title, description]]

    def initialize_persian(self):
        tree = ET.parse('data/Persian.xml')
        root = tree.getroot()
        titles = []
        descriptions = []
        for child in root:
            for sub_child in child:
                if sub_child.tag == '{http://www.mediawiki.org/xml/export-0.10/}title':
                    titles.append(sub_child.text)
                if sub_child.tag == '{http://www.mediawiki.org/xml/export-0.10/}revision':
                    revision = sub_child
                    for x in revision:
                        if x.tag == '{http://www.mediawiki.org/xml/export-0.10/}text':
                            new_text = re.sub("[\{].*?[\}]", "", x.text)
                            new_text = re.sub(r'[0-9]', "", new_text)
                            descriptions.append(new_text)
        self.collections["persian"].extend([titles, descriptions])

    @staticmethod
    def word_correction(token, punc):
        new_word = ""
        for i in range(len(token)):
            if token[i] in punc:
                new_word += " "
            else:
                new_word += token[i]
        return new_word

    def prepare_text(self, documents, lang, stop_words):
        if lang == "english":
            return self.prepare_english(documents, stop_words)
        elif lang == "persian":
            return self.prepare_persian(documents, stop_words)

    @staticmethod
    def process_stop_words(size, all_tokens):
        frequency_counter = Counter(all_tokens)
        tokens_size = len(all_tokens)
        sorted_token_counter = frequency_counter.most_common(
            len(frequency_counter))
        sorted_token_ratio = [(c[0], c[1] / tokens_size)
                              for c in sorted_token_counter]
        stop_words = [sorted_token_counter[i][0] for i in range(size)]
        remaining_terms = [(sorted_token_counter[i][0], sorted_token_counter[i][1]) for i in
                           range(size, len(frequency_counter))]
        r = range(size)
        y = [sorted_token_counter[i][1] for i in range(size)]
        plt.bar(r, y, color="red", align="center")
        plt.title("Stopwords Frequencies")
        plt.xticks(r, stop_words, rotation="vertical")
        plt.show()
        return remaining_terms, stop_words

    def prepare_english(self, documents, stop_words):
        processed_documents = []
        all_tokens = []
        for i in range(len(documents)):
            document = documents[i]
            parts = []
            for part in document:
                tokenized_part = word_tokenize(part)
                case_folded_part = [word.lower() for word in tokenized_part]
                for j in range(len(case_folded_part)):
                    if case_folded_part[j].__contains__(",") or case_folded_part[j].__contains__("'") or \
                            case_folded_part[j].__contains__("-") or case_folded_part[j].__contains__("?"):
                        case_folded_part[j] = case_folded_part[j].replace(",", "").replace("'", "").replace("-",
                                                                                                            "").replace(
                            "?", "")
                removed_punctuation_part = [
                    word for word in case_folded_part if word.isalpha()]
                parts += [removed_punctuation_part]
                all_tokens += [word for word in removed_punctuation_part]
            processed_documents += [parts]
        if len(stop_words) == 0:
            remaining_terms, stop_words = self.process_stop_words(40, all_tokens)
        else:
            remaining_terms = []
        final_tokens = []
        stemmer = SnowballStemmer("english")
        for i in range(len(documents)):
            parts = processed_documents[i]
            for j in range(len(parts)):
                parts[j] = [word for word in parts[j]
                            if word not in stop_words]
                parts[j] = [stemmer.stem(word) for word in parts[j]]
                final_tokens += [word for word in parts[j]]
        return final_tokens, processed_documents, remaining_terms, stop_words

    def prepare_persian(self, documents, stop_words):
        punctuation = ['!', '"', "'", '#', '(', ')', '*', '-', ',', '.', '/', ':', '[', ']', '|', ';', '?', '،',
                       '...',
                       '$',
                       '{',
                       '}', '=', '==', '===', '>', '<', '>>', '<<', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹',
                       '۰',
                       '«', '||',
                       '""', "''", "&", "'''", '"""', '»', '', '–', "؛", "^", "--", "<--", "-->", "_", "--", "٬",
                       "؟"]
        normalizer = Normalizer()
        titles = documents[0]
        descriptions = []
        if len(documents) == 2:
            descriptions = documents[1]
        for i in range(len(titles)):

            titles_array = []
            descriptions_array = []

            titles[i] = self.word_correction(titles[i], punctuation)
            if len(descriptions) != 0:
                descriptions[i] = self.word_correction(
                    descriptions[i], punctuation)
            titles[i] = normalizer.normalize(titles[i])
            titles[i] = word_tokenize(titles[i])
            if len(descriptions) != 0:
                descriptions[i] = normalizer.normalize(descriptions[i])
                descriptions[i] = word_tokenize(descriptions[i])
            for persian_word in titles[i]:
                persian_word = persian_word.replace("\u200c", "")
                if len(re.findall(r'([a-zA-Z]+)', persian_word)) == 0:
                    titles_array.append(persian_word)
            if len(descriptions) != 0:
                for persian_word in descriptions[i]:
                    persian_word = re.sub(r'[^\s\w]', "", persian_word)
                    if len(re.findall(r'([a-zA-Z]+)', persian_word)) == 0:
                        descriptions_array.append(persian_word)
            titles[i] = titles_array
            if len(descriptions) != 0:
                descriptions[i] = descriptions_array
            # for x in titles[i]:
            #     for word in x.split('|'):
            #         if len(re.findall(r'([a-zA-Z]+)', word)) == 0:
            #             titles_array.append(word)
            # if len(descriptions) != 0:
            #     for x in descriptions[i]:
            #         for word in x.split('|'):
            #             if len(re.findall(r'([a-zA-Z]+)', word)) == 0:
            #                 descriptions_array.append(word)
        stemmer = Stemmer()

        all_tokens = []
        dictionary = []
        for i in range(len(titles)):
            title_arr = []
            description_arr = []
            for x in titles[i]:
                if x not in punctuation and len(x) > 0:
                    all_tokens.append(stemmer.stem(x))
                    title_arr.append(stemmer.stem(x))
            if len(descriptions) != 0:
                for x in descriptions[i]:
                    if x not in punctuation and len(x) > 0:
                        all_tokens.append(stemmer.stem(x))
                        description_arr.append(stemmer.stem(x))
            dictionary.append([title_arr, description_arr])

        if len(stop_words) == 0:
            remaining_terms, stop_words = self.process_stop_words(40, all_tokens)
        else:
            remaining_terms = []
        final_tokens = []
        for word in all_tokens:
            if word not in stop_words:
                final_tokens.append(word)
        processed_documents = []
        for doc in dictionary:
            processed_title_document = []
            processed_description_document = []
            for word in doc[0]:
                if word not in stop_words:
                    processed_title_document.append(word)
            for word in doc[1]:
                if word not in stop_words:
                    processed_description_document.append(word)
            processed_documents.append(
                [processed_title_document, processed_description_document])
        return final_tokens, processed_documents, remaining_terms, stop_words

    @staticmethod
    def positional(input_list, positional_index_creation, start, end):
        for docID in range(start - 1, end):
            for col in range(2):
                for ind in range(len(input_list[docID - start + 1][col])):
                    term = input_list[docID - start + 1][col][ind]
                    if term not in positional_index_creation.keys():  # new term
                        positional_index_creation[term] = dict()
                        positional_index_creation[term]["cf"] = dict()
                        positional_index_creation[term]["cf"] = 0
                    # our term is found in new docID
                    if docID not in positional_index_creation[term].keys():
                        positional_index_creation[term][docID] = dict()
                    positional_index_creation[term]["cf"] += 1
                    if col == 0:
                        if "title" not in positional_index_creation[term][docID].keys():
                            positional_index_creation[term][docID]["title"] = [
                                ind + 1]
                        else:
                            positional_index_creation[term][docID]["title"] += [ind + 1]

                    elif col == 1:
                        if "description" not in positional_index_creation[term][docID].keys():
                            positional_index_creation[term][docID]["description"] = [
                                ind + 1]
                        else:
                            positional_index_creation[term][docID]["description"] += [
                                ind + 1]

    @staticmethod
    def bigram(input_list, bigram_creation, start, end):
        for docID in range(start - 1, end):
            for col in range(2):
                for ind in range(len(input_list[docID - start + 1][col])):
                    term = input_list[docID - start + 1][col][ind]
                    if len(term) != 0:
                        for i in range(-1, len(term), 1):
                            if i == -1:
                                sub_term = "$" + term[0]

                            elif i == len(term) - 1:
                                sub_term = term[-1] + "$"
                            else:
                                sub_term = term[i:i + 2]

                            if sub_term not in bigram_creation.keys():
                                bigram_creation[sub_term] = [term]
                            elif term not in bigram_creation[sub_term]:
                                bigram_creation[sub_term] += [term]

    def insert(self, documents, lang, bigram_index, positional_index):
        doc_tokens, docs_structured, doc_terms, doc_stops = self.prepare_text(documents, lang,
                                                                              self.stop_words_dic[lang])
        self.document_tokens[lang] += [word for word in doc_tokens]
        self.structured_documents[lang] += [doc for doc in docs_structured]
        self.terms[lang] += [term for term in doc_terms if term not in self.terms[lang]]
        self.bigram(docs_structured, bigram_index,
                    self.docs_size[lang] + 1, self.docs_size[lang] + len(documents))
        self.positional(docs_structured, positional_index, self.docs_size[lang] + 1,
                        self.docs_size[lang] + len(documents))
        self.docs_size[lang] += len(documents)
        for _ in range(len(documents)):
            self.deleted_documents[lang] += [False]
        return bigram_index, positional_index

    def csv_insert(self, csv_path, lang):
        columns = ["description", "title"]
        df = pd.read_csv(csv_path, usecols=columns)
        x = len(df)
        documents = []
        for i in range(x):
            title = df.iloc[i]["title"]
            description = df.iloc[i]["description"]
            documents += [[title, description]]
        self.insert(documents, lang,
                    self.bigram_index[lang], self.positional_index[lang])

    def xml_insert(self, xml_path, lang):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        titles = []
        descriptions = []
        documents = []
        for child in root:
            for sub_child in child:
                if sub_child.tag == '{http://www.mediawiki.org/xml/export-0.10/}title':
                    titles.append(sub_child.text)
                if sub_child.tag == '{http://www.mediawiki.org/xml/export-0.10/}revision':
                    revision = sub_child
                    for x in revision:
                        if x.tag == '{http://www.mediawiki.org/xml/export-0.10/}text':
                            s = x.text
                            new_text = re.sub("[\{].*?[\}]", "", s)
                            descriptions.append(new_text)
        documents.extend([titles, descriptions])
        self.insert(documents, lang,
                    self.bigram_index[lang], self.positional_index[lang])

    def delete(self, documents, doc_id, bigram_index, positional_index, deleted_list, lang):
        if doc_id >= len(deleted_list):
            print("docID (" + str(doc_id + 1) + ") doesn't exist!")
            return
        if not deleted_list[doc_id]:
            document = documents[doc_id]
            for part in document:
                for term in part:
                    positional_index[term]["cf"] -= 1
                    if doc_id in positional_index[term].keys():
                        del positional_index[term][doc_id]
                    if positional_index[term]["cf"] == 0:
                        first = '$' + term[0]
                        last = term[len(term) - 1] + '$'
                        if term in bigram_index[first]:
                            bigram_index[first].remove(term)
                        if len(bigram_index[first]) == 0:
                            del bigram_index[first]
                        if term in bigram_index[last]:
                            bigram_index[last].remove(term)
                        if len(bigram_index[last]) == 0:
                            del bigram_index[last]
                        for i in range(0, len(term) - 1):
                            s = term[i:i + 2]
                            if term in bigram_index[s]:
                                bigram_index[s].remove(term)
                            if len(bigram_index[s]) == 0:
                                del bigram_index[s]
                        del positional_index[term]
            deleted_list[doc_id] = True
            self.docs_size[lang] -= 1
        else:
            print("this docID (" + str(doc_id + 1) +
                  ") does not exist in the documents set!")

    @staticmethod
    def create_gamma_code(number, col):  # col is "title" or "description"
        gamma_code = ""
        binary_of_number = bin(number)[2:]
        if number == 0:
            gamma_code += "0"
        else:
            right_section = binary_of_number[1:]
            for i in range(len(right_section)):
                gamma_code += "1"
            gamma_code += "0"
            gamma_code += right_section
        if col == "title":
            gamma_code += "0"
        elif col == "description":
            gamma_code += "1"
        return gamma_code

    @staticmethod
    def decode_gamma_code(number):
        string_number = number
        col_bit = string_number[-1]
        if col_bit == "0":
            col = "title"
        else:
            col = "description"
        gamma_code = string_number[:len(string_number) - 1]
        count_of_one = 0
        for i in range(len(gamma_code)):
            if gamma_code[i] == "1":
                count_of_one += 1
            else:
                break
        decoded_gamma_code_to_number = "1"
        decoded_gamma_code_to_number += gamma_code[count_of_one + 1:]
        return int(decoded_gamma_code_to_number, 2), col

    def positional_index_to_gamma_code(self, positional_index, gamma_positional_index):
        for term in positional_index.keys():
            for doc_id in positional_index[term].keys():
                if term not in gamma_positional_index.keys():
                    gamma_positional_index[term] = dict()
                if doc_id not in gamma_positional_index[term].keys():
                    gamma_positional_index[term][doc_id] = "1"
                if doc_id == "cf":
                    gamma_positional_index[term]["cf"] = positional_index[term]["cf"]
                    continue
                flag = False
                for col in positional_index[term][doc_id].keys():
                    for i in range(len(positional_index[term][doc_id][col])):
                        if not flag:
                            gamma_positional_index[term][doc_id] += self.create_gamma_code(
                                positional_index[term][doc_id][col][i], col)
                            flag = True
                        elif i == 0:
                            gamma_positional_index[term][doc_id] += self.create_gamma_code(
                                positional_index[term][doc_id][col][i], col)
                        else:
                            gamma_positional_index[term][doc_id] += self.create_gamma_code(
                                positional_index[term][doc_id][col][i] - positional_index[term][doc_id][col][i - 1],
                                col)
                gamma_positional_index[term][doc_id] = int(gamma_positional_index[term][doc_id], 2).to_bytes(
                    math.ceil(len(gamma_positional_index[term][doc_id]) / 8), sys.byteorder)

    def gamma_code_to_positional_index(self, gamma_positional_index, positional_index):
        dict(positional_index).clear()
        for term in gamma_positional_index.keys():
            for doc_id in gamma_positional_index[term].keys():
                if term not in positional_index.keys():
                    positional_index[term] = dict()
                if doc_id not in positional_index[term].keys():
                    positional_index[term][doc_id] = dict()
                if doc_id == "cf":
                    positional_index[term]["cf"] = gamma_positional_index[term]["cf"]
                    continue
                gamma_code = str(format(int.from_bytes(gamma_positional_index[term][doc_id], sys.byteorder), 'b'))
                gamma_code = gamma_code[1:]
                i = 0
                j = 0
                len_of_gamma_code = 0
                while True:
                    if i == len(gamma_code):
                        break
                    if gamma_code[j] == "1":
                        len_of_gamma_code += 1
                        j += 1
                        continue
                    else:
                        len_of_gamma_code = 2 * (len_of_gamma_code + 1)
                        gap, col = self.decode_gamma_code(
                            gamma_code[i:i + len_of_gamma_code])
                        if col not in positional_index[term][doc_id].keys():
                            positional_index[term][doc_id][col] = [gap]
                        else:
                            last_value = positional_index[term][doc_id][col][-1]
                            positional_index[term][doc_id][col] += [last_value + gap]
                        i += len_of_gamma_code
                        j = i
                        len_of_gamma_code = 0

    @staticmethod
    def create_variable_byte(number, col):  # col is "title" or "description"
        number = bin(number).replace("0b", "")

        while len(number) % 6 != 0:
            number = "0" + number
        result = ""
        byte_size = len(number) // 6
        for i in range(byte_size):
            if i == byte_size - 1:
                result += "1"
            else:
                result += "0"
            result += number[6 * i:6 * (i + 1)]
            if col == "title":
                result += "0"
            elif col == "description":
                result += "1"
        # returns bytes of data
        return int(result, 2).to_bytes(byte_size, sys.byteorder)

    @staticmethod
    def decode_variable_byte(number):
        my_bytes = []
        for byte in number:
            my_bytes += [byte.to_bytes(1, sys.byteorder)]
        my_bytes.reverse()
        number = ""
        for byte in my_bytes:
            number += format(int.from_bytes(byte, sys.byteorder), 'b')
        while len(number) % 8 != 0:
            number = "0" + number
        byte_size = len(number) // 8
        result = ""
        for i in range(byte_size):
            result += number[8 * i + 1:8 * i + 7]
        col = (number[-1] == "0") * "title" + \
              (number[-1] == "1") * "description"
        return int(result, 2), col

    def positional_index_to_variable_byte(self, positional_index, vb_positional_index):
        for term in positional_index.keys():
            for doc_id in positional_index[term].keys():
                if term not in vb_positional_index.keys():
                    vb_positional_index[term] = dict()
                if doc_id not in vb_positional_index[term].keys():
                    vb_positional_index[term][doc_id] = dict()
                if doc_id == "cf":
                    vb_positional_index[term]["cf"] = positional_index[term]["cf"]
                    continue
                flag = False
                for col in positional_index[term][doc_id].keys():
                    for i in range(len(positional_index[term][doc_id][col])):
                        if not flag:
                            vb_positional_index[term][doc_id] = [
                                self.create_variable_byte(positional_index[term][doc_id][col][i], col)]
                            flag = True
                        elif i == 0:
                            vb_positional_index[term][doc_id] += [
                                self.create_variable_byte(positional_index[term][doc_id][col][i], col)]
                        else:
                            vb_positional_index[term][doc_id] += [
                                self.create_variable_byte(positional_index[term][doc_id][col][i]
                                                          - positional_index[term][doc_id][col][i - 1], col)]

    def variable_byte_to_positional_index(self, vb_positional_index, positional_index):
        dict(positional_index).clear()
        for term in vb_positional_index.keys():
            for doc_id in vb_positional_index[term].keys():
                if term not in positional_index.keys():
                    positional_index[term] = dict()
                if doc_id not in positional_index[term].keys():
                    positional_index[term][doc_id] = dict()
                if doc_id == "cf":
                    positional_index[term]["cf"] = vb_positional_index[term]["cf"]
                    continue
                for i in range(len(vb_positional_index[term][doc_id])):
                    gap, col = self.decode_variable_byte(
                        vb_positional_index[term][doc_id][i])
                    if col not in positional_index[term][doc_id].keys():
                        positional_index[term][doc_id][col] = [gap]
                    else:
                        last_value = positional_index[term][doc_id][col][-1]
                        positional_index[term][doc_id][col] += [last_value + gap]

    def jaccard_similarity(self, query, term, lang):
        query_bigrams = []
        for i in range(0, len(query) - 1):
            query_bigrams += [query[i:i + 2]]
        intersect_counter = 0
        for bichar in query_bigrams:
            if bichar in self.bigram_index[lang].keys() and term in self.bigram_index[lang][bichar]:
                intersect_counter += 1
        return intersect_counter / (len(query_bigrams) + len(term) - 1 - intersect_counter)

    def correction_list(self, word, lang, threshold):
        word_bigrams = []
        for i in range(0, len(word) - 1):
            word_bigrams += [word[i:i + 2]]
        suggested_terms = []
        for bichar in word_bigrams:
            if bichar in self.bigram_index[lang].keys():
                for term in self.bigram_index[lang][bichar]:
                    if term not in suggested_terms:
                        if self.jaccard_similarity(word, term, lang) > threshold:
                            suggested_terms += [term]
        return suggested_terms

    @staticmethod
    def edit_distance(query, term):
        dp = [[0 for _ in range(len(term) + 1)] for _ in range(len(query) + 1)]
        for i in range(len(query) + 1):
            dp[i][0] = i
        for j in range(len(term) + 1):
            dp[0][j] = j
        for i in range(1, len(query) + 1):
            for j in range(1, len(term) + 1):
                if query[i - 1] == term[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i]
                    [j - 1] + 1, dp[i - 1][j - 1] + 1)
        return dp[len(query)][len(term)]

    def doc_length(self, doc_id, lang):
        doc_terms = []
        document = self.structured_documents[lang][doc_id]
        for part in document:
            for word in part:
                doc_terms += [word]
        length = 0
        counted_terms = Counter(doc_terms)
        for word in counted_terms.keys():
            df_word = len(self.positional_index[lang][word].keys()) - 1
            idf_word = math.log10(
                len(self.structured_documents[lang]) / df_word)
            tf_idf_word = (1 + math.log10(counted_terms[word])) * idf_word
            length += (tf_idf_word ** 2)
        return math.sqrt(length)

    def tf_idf(self, query, doc_id, lang, q_length):
        result = 0
        for term in query.keys():
            q_tf = 1 + math.log10(query[term])
            p = self.positional_index[lang][term]
            df = len(p.keys()) - 1
            idf = math.log10(len(self.structured_documents[lang]) / df)
            if doc_id - 1 in p.keys():
                tf = 0
                if "title" in p[doc_id - 1].keys():
                    tf += len(p[doc_id - 1]["title"])
                if "description" in p[doc_id - 1].keys():
                    tf += len(p[doc_id - 1]["description"])
                result += (((1 + math.log10(tf)) * idf) / self.doc_length(doc_id -
                                                                          1, lang) * (q_tf / q_length))
        return result

    def call_prepare(self, lang):
        self.document_tokens[lang], self.structured_documents[lang], self.terms[lang], self.stop_words_dic[
            lang] = self.prepare_text(
            self.collections[lang], lang, [])
        self.docs_size[lang] = len(self.structured_documents[lang])
        self.deleted_documents[lang] = [
            False for _ in range(len(self.structured_documents[lang]))]

    def call_create_bigram(self, lang):
        self.bigram(
            self.structured_documents[lang], self.bigram_index[lang], 1, self.docs_size[lang])
        print("creation was successful")

    def call_create_positional(self, lang):
        self.positional(
            self.structured_documents[lang], self.positional_index[lang], 1, self.docs_size[lang])
        print("creation was successful")

    def call_bigram(self, lang, biword):
        if biword in self.bigram_index[lang].keys():
            print(self.bigram_index[lang][biword])
        else:
            print("biword (" + biword + ") doesn't match any word in " +
                  lang + " documents.")

    def call_positional(self, lang, term):
        if term in self.positional_index[lang].keys():
            print(self.positional_index[lang][term])
        else:
            print("term (" + term + ") doesn't match any term in " +
                  lang + " documents.")

    def call_compress_variable_byte(self, lang):
        self.positional_index_to_variable_byte(
            self.positional_index[lang], self.vb_positional_index[lang])
        if lang == "english":
            with open('variable_byte_english', 'wb') as pickle_file:
                pickle.dump(
                    self.vb_positional_index["english"], pickle_file, pickle.HIGHEST_PROTOCOL)
                pickle_file.close()
        elif lang == "persian":
            with open('variable_byte_persian', 'wb') as pickle_file:
                pickle.dump(
                    self.vb_positional_index["persian"], pickle_file, pickle.HIGHEST_PROTOCOL)
                pickle_file.close()
        print("Positional Index Saved Successfully With Variable Byte Method")

    def call_compress_gamma_code(self, lang):
        self.positional_index_to_gamma_code(
            self.positional_index[lang], self.gamma_positional_index[lang])
        if lang == "english":
            with open('gamma_code_english', 'wb') as pickle_file:
                pickle.dump(
                    self.gamma_positional_index["english"], pickle_file, pickle.HIGHEST_PROTOCOL)
                pickle_file.close()
        elif lang == "persian":
            with open('gamma_code_persian', 'wb') as pickle_file:
                pickle.dump(
                    self.gamma_positional_index["persian"], pickle_file, pickle.HIGHEST_PROTOCOL)
                pickle_file.close()
        print("Positional Index Saved Successfully With Gamma Method")

    def call_decompress_variable_byte(self, lang):
        if lang == "english":
            with open('variable_byte_english', 'rb') as pickle_file:
                self.vb_positional_index["english"] = pickle.load(pickle_file)
                pickle_file.close()
        elif lang == "persian":
            with open('variable_byte_persian', 'rb') as pickle_file:
                self.vb_positional_index["persian"] = pickle.load(pickle_file)
                pickle_file.close()
        self.positional_index[lang].clear()
        self.variable_byte_to_positional_index(
            self.vb_positional_index[lang], self.positional_index[lang])
        print("Decompressed Done Successfully With Variable Byte Method")

    def call_decompress_gamma_code(self, lang):
        if lang == "english":
            with open('gamma_code_english', 'rb') as pickle_file:
                self.gamma_positional_index["english"] = pickle.load(
                    pickle_file)
                pickle_file.close()
        elif lang == "persian":
            with open('gamma_code_persian', 'rb') as pickle_file:
                self.gamma_positional_index["persian"] = pickle.load(
                    pickle_file)
                pickle_file.close()
        self.positional_index[lang].clear()
        self.gamma_code_to_positional_index(
            self.gamma_positional_index[lang], self.positional_index[lang])
        print("Decompressed Done Successfully With Gamma Code Method")

    def call_delete(self, lang, doc_id):
        self.delete(self.structured_documents[lang], doc_id - 1, self.bigram_index[lang], self.positional_index[lang],
                    self.deleted_documents[lang], lang)

    def call_insert(self, lang, doc_number, part_number):
        new_docs = []
        for _ in range(doc_number):
            new_docs += [[]]
            for i in range(part_number):
                t = input()
                new_docs[-1] += [t]
        self.insert(new_docs, lang,
                    self.bigram_index[lang], self.positional_index[lang])

    def call_save_index(self, type_of_indexing, lang):
        if lang == "english" and type_of_indexing == "positional":
            with open('positional_english_indexing', 'wb') as pickle_file:
                pickle.dump(
                    self.positional_index["english"], pickle_file, pickle.HIGHEST_PROTOCOL)
                pickle_file.close()
        elif lang == "english" and type_of_indexing == "bigram":
            with open('bigram_english_indexing', 'wb') as pickle_file:
                pickle.dump(
                    self.bigram_index["english"], pickle_file, pickle.HIGHEST_PROTOCOL)
                pickle_file.close()
        elif lang == "persian" and type_of_indexing == "positional":
            with open('positional_persian_indexing', 'wb') as pickle_file:
                pickle.dump(
                    self.positional_index["persian"], pickle_file, pickle.HIGHEST_PROTOCOL)
                pickle_file.close()
        elif lang == "persian" and type_of_indexing == "bigram":
            with open('bigram_persian_indexing', 'wb') as pickle_file:
                pickle.dump(
                    self.bigram_index["persian"], pickle_file, pickle.HIGHEST_PROTOCOL)
                pickle_file.close()
        elif lang == "english" and type_of_indexing == "stop_words":
            with open('stop_words_english_indexing', 'wb') as pickle_file:
                pickle.dump(
                    self.stop_words_dic["english"], pickle_file, pickle.HIGHEST_PROTOCOL)
                pickle_file.close()
        elif lang == "persian" and type_of_indexing == "stop_words":
            with open('stop_words_persian_indexing', 'wb') as pickle_file:
                pickle.dump(
                    self.stop_words_dic["persian"], pickle_file, pickle.HIGHEST_PROTOCOL)
                pickle_file.close()
        elif lang == "english" and type_of_indexing == "structured_documents":
            with open('structured_documents_english_indexing', 'wb') as pickle_file:
                pickle.dump(
                    self.structured_documents["english"], pickle_file, pickle.HIGHEST_PROTOCOL)
                pickle_file.close()
        elif lang == "persian" and type_of_indexing == "structured_documents":
            with open('structured_documents_persian_indexing', 'wb') as pickle_file:
                pickle.dump(
                    self.structured_documents["persian"], pickle_file, pickle.HIGHEST_PROTOCOL)
                pickle_file.close()
        print(type_of_indexing + " " + lang + " indexing saved successfully")

    def call_load_index(self, type_of_indexing, lang):
        is_found = True
        if lang == "english" and type_of_indexing == "positional":
            try:
                with open('positional_english_indexing', 'rb') as pickle_file:
                    self.positional_index["english"] = pickle.load(pickle_file)
                    pickle_file.close()
            except IOError:
                is_found = False
                print("File Not Found!!")
        elif lang == "english" and type_of_indexing == "bigram":
            try:
                with open('bigram_english_indexing', 'rb') as pickle_file:
                    self.bigram_index["english"] = pickle.load(pickle_file)
                    pickle_file.close()
            except IOError:
                is_found = False
                print("File Not Found!!")
        elif lang == "persian" and type_of_indexing == "positional":
            try:
                with open('positional_persian_indexing', 'rb') as pickle_file:
                    self.positional_index["persian"] = pickle.load(pickle_file)
                    pickle_file.close()
            except IOError:
                is_found = False
                print("File Not Found!!")
        elif lang == "persian" and type_of_indexing == "bigram":
            try:
                with open('bigram_persian_indexing', 'rb') as pickle_file:
                    self.bigram_index["persian"] = pickle.load(pickle_file)
                    pickle_file.close()
            except IOError:
                is_found = False
                print("File Not Found!!")
        elif lang == "english" and type_of_indexing == "stop_words":
            try:
                with open('stop_words_english_indexing', 'rb') as pickle_file:
                    self.stop_words_dic["english"] = pickle.load(pickle_file)
                    pickle_file.close()
            except IOError:
                is_found = False
                print("File Not Found!!")
        elif lang == "persian" and type_of_indexing == "stop_words":
            try:
                with open('stop_words_persian_indexing', 'rb') as pickle_file:
                    self.stop_words_dic["persian"] = pickle.load(pickle_file)
                    pickle_file.close()
            except IOError:
                is_found = False
                print("File Not Found!!")
        elif lang == "english" and type_of_indexing == "structured_documents":
            try:
                with open('structured_documents_english_indexing', 'rb') as pickle_file:
                    self.structured_documents["english"] = pickle.load(pickle_file)
                    pickle_file.close()
            except IOError:
                is_found = False
                print("File Not Found!!")
        elif lang == "persian" and type_of_indexing == "structured_documents":
            try:
                with open('structured_documents_persian_indexing', 'rb') as pickle_file:
                    self.structured_documents["persian"] = pickle.load(pickle_file)
                    pickle_file.close()
            except IOError:
                is_found = False
                print("File Not Found!!")
        if is_found:
            print(type_of_indexing + " " + lang + " indexing loaded successfully")

    def call_correction_list(self, lang, term):
        corrected_list = []
        threshold = 0.4
        while len(corrected_list) == 0:
            corrected_list = self.correction_list(term, lang, threshold)
            threshold -= 0.1
        return corrected_list

    def query_spell_correction(self, lang, query):
        document = [[query]]
        query_tokens, _, _, _ = self.prepare_text(
            document, lang, self.stop_words_dic[lang])
        correct_query = True
        correction = []
        for token in query_tokens:
            if token not in self.positional_index[lang].keys():
                correct_query = False
                suggested_list = self.call_correction_list(lang, token)
                edit_distances = []
                for term in suggested_list:
                    edit_distances += [self.edit_distance(token, term)]
                ind = edit_distances.index(min(edit_distances))
                correction += [suggested_list[ind]]
            else:
                correction += [token]

        if correct_query:
            print("no spell correction needed!")
        else:
            new_str = "suggested correction for the query:"
            for word in correction:
                new_str += (" " + word)
            print(new_str)
        return correction

    def process_usual_query(self, lang, correction):
        query_dict = Counter(correction)
        q_length = sum((1 + math.log10(query_dict[t])) ** 2 for t in query_dict.keys())
        q_length = math.sqrt(q_length)
        scores = []
        for doc_id in range(len(self.structured_documents[lang])):
            scores += [self.tf_idf(query_dict, doc_id + 1, lang, q_length)]
        top_ten = [s[0] for s in sorted(
            enumerate(scores), key=lambda a: a[1], reverse=True)]
        for i in range(10):
            if not scores[top_ten[i]] == 0:
                print("document " + str(top_ten[i] + 1) + ":",
                      self.structured_documents[lang][top_ten[i]])
                print("ltc-lnc score:", (scores[top_ten[i]]))

    def process_proximity_query(self, lang, correction, proximity_len_of_window):
        list_of_document_contain_all_words = self.documents_contain_all_words(
            lang, correction)
        list_of_document_satisfying_proximity = self.documents_satisfying_proximity(lang, correction,
                                                                                    proximity_len_of_window,
                                                                                    list_of_document_contain_all_words)
        query_dict = Counter(correction)
        q_length = sum((1 + math.log10(query_dict[t])) ** 2 for t in query_dict.keys())
        q_length = math.sqrt(q_length)
        scores = []
        for doc_id in list_of_document_satisfying_proximity:
            scores += [(self.tf_idf(query_dict, doc_id +
                                    1, lang, q_length), doc_id + 1)]
        top_ten = sorted(scores, key=lambda x: x[0], reverse=True)
        if len(top_ten) == 0:
            print("No Result Found!!")
        else:
            for i in range(10):
                print("document " + str(top_ten[i][1]) + ":",
                      self.structured_documents[lang][top_ten[i][1] - 1])
                print("ltc-lnc score:", top_ten[i][0])
                if i == len(top_ten) - 1:
                    break

    def documents_contain_all_words(self, lang, correction):
        minimum_document_frequency_in_correction_query = correction[0]
        list_of_document_contain_all_words = []
        for term in correction:
            if len(self.positional_index[lang][term].keys()) < \
                    len(self.positional_index[lang][minimum_document_frequency_in_correction_query].keys()):
                minimum_document_frequency_in_correction_query = term
        document_contain_min_term = []
        for x in self.positional_index[lang][minimum_document_frequency_in_correction_query].keys():
            if x != "cf":
                document_contain_min_term.append(x)
        for x in document_contain_min_term:
            flag = True
            for term in correction:
                if x not in self.positional_index[lang][term].keys():
                    flag = False
                    break
            if flag:
                list_of_document_contain_all_words.append(x)
        return list_of_document_contain_all_words

    @staticmethod
    def check_word_is_near(arr1, arr2):
        for x1 in arr1:
            if not arr2.__contains__(x1):
                return False
        return True

    def documents_satisfying_proximity(self, lang, correction, proximity_len_of_window, suggested_docs):
        if len(correction) == 1:
            return suggested_docs
        list_of_document_contain_all_words_with_proximity = []
        for x in suggested_docs:
            checked = False
            for y in self.structured_documents[lang][x]:
                for i in range(len(y)):
                    if i + proximity_len_of_window > len(y):
                        break
                    else:
                        if self.check_word_is_near(correction, y[i:i + proximity_len_of_window]):
                            list_of_document_contain_all_words_with_proximity.append(
                                x)
                            checked = True
                            break
                if checked:
                    break
        return list_of_document_contain_all_words_with_proximity
