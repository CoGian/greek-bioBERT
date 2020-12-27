import os
import json


def main():
    path = 'texts'
    abstract_possible_names = ["περιληψη", "περίληψη", "περιληψη:", "περίληψη:", ]
    keywords_possible_names = ["λέξεις-κλειδιά:", "λεξεις-κλειδια:", "κλειδιά:", "κλειδια:",
                               "λέξεις-κλειδιά", "λεξεις-κλειδια", "κλειδιά", "κλειδια",
                               "ευρετηριασμού:", "ευρετηρίου:", "ευρετηριασμoύ:", "ευρετηριασµού:"]
    stop_abstract_keyword_reading = ["εισαγωγή", "εισαγωγή:", "εισαγωγη", "εισαγωγη:", "ιστορική", "ιστορικη"
                                     "σκοπός", "σκοπος", "Σκοπός"]

    output = []
    output_with_keywords = []

    for journal in sorted(os.listdir(path)):
        journal_path = os.path.join(path, journal)
        for text in sorted(os.listdir(journal_path)):
            text_path = os.path.join(journal_path, text)
            if "src" in text_path:
                continue
            with open(text_path, "r") as text_file:
                text_content = text_file.read()
                rest_text = []
                abstract = []
                keywords = []
                save_abstract = False
                save_keywords = False

                output_dict = {}
                for word in text_content.encode('utf-8').decode("utf-8").split():

                    if word.lower() in abstract_possible_names:
                        save_abstract = True
                        continue

                    if word.lower() in stop_abstract_keyword_reading:
                        break

                    if (word.istitle() or word.isupper()) and len(keywords) >= 7:
                        break

                    if word.lower() in keywords_possible_names:
                        save_keywords = True
                        save_abstract = False
                        continue

                    if save_abstract:
                        if word.lower() not in ['λέξεις', 'λεξεις', 'όροι', 'οροι'] and word not in stop_abstract_keyword_reading:
                            abstract.append(word)
                    elif save_keywords:
                        keywords.append(word)
                    else:
                        rest_text.append(word)

            if len(abstract) > 0:
                output_dict["abstract"] = " ".join(abstract)
                output_dict["keywords"] = " ".join(keywords)
                if len(keywords) > 0:
                    output_with_keywords.append(output_dict)
                else:
                    output.append(output_dict)

    with open('iatrolexi_corpus.json', 'w',  encoding='utf8') as fout:
        json.dump(output, fout, ensure_ascii=False)

    with open('iatrolexi_abstracts_with_keywords.json', 'w', encoding='utf8') as fout:
        json.dump(output_with_keywords, fout, ensure_ascii=False)


if __name__ == '__main__':
    main()
