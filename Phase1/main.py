from Phase1.ir_system import IRSystem

ir_sys = IRSystem()


def check_language(lang):
    if (not lang == "english") and (not lang == "persian"):
        print("this language " + lang + " is not supported")
        return False
    return True


def check_index(index):
    return index == "positional" or index == "bigram" or index == "stop_words" or index == "structured_documents"


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
            ir_sys.call_prepare(lang)
    elif split_text[0] == "create":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        if split_text[1] == "bigram":
            lang = split_text[2]
            if check_language(lang):
                ir_sys.call_create_bigram(lang)
        elif split_text[1] == "positional":
            lang = split_text[2]
            if check_language(lang):
                ir_sys.call_create_positional(lang)
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
            print(ir_sys.terms[lang])
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
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        type_of_indexing = split_text[1]
        lang = split_text[2]
        if check_language(lang) and check_index(type_of_indexing):
            ir_sys.call_save_index(type_of_indexing, lang)
    elif split_text[0] == "load":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        type_of_indexing = split_text[1]
        lang = split_text[2]
        if check_language(lang) and check_index(type_of_indexing):
            ir_sys.call_load_index(type_of_indexing, lang)
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
    else:
        print("not a valid command!")
