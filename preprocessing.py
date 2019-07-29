#!/usr/bin/python
import message
import prepare_infrastructure
import query_database
import os
import json
import argparse


# removing duplicates from data
def remove_duplicates(data):
    print("INFO: Removing duplicates")
    cleaned_data = []
    previous_line = data[0]
    num = 0
    for line in data:
        if (line != previous_line) or num == 0:
            cleaned_data.append(line)
            previous_line = data[num]
            num += 1
        else:
            previous_line = data[num]
            num += 1
    return cleaned_data


if __name__ == '__main__':

    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("-f", "--messages_from_file", required=False, type=str, default="data/test_topics.csv",
                    help="Path to local message file")
    ap.add_argument("-d", "--database", required=False, type=bool, default=False,
                    help="Access database?")
    # general output folder
    ap.add_argument("-o", "--output_folder", required=False, default="output", type=str, help="Name of the output folder, not the path.")

    args = vars(ap.parse_args())

    # setting up directory structure
    prepare_infrastructure.create_environment(args["output_folder"])

    # prepare output file names
    clusterable_words, clusterable_words_with_hashtags, clusterable_words_only_verbs, clusterable_words_only_adjectives, clusterable_words_only_nouns, preprocessed = prepare_infrastructure.prepare_file_name_preprocessing(args["messages_from_file"], args["output_folder"])

    # enter query words to obtain the data from messages_from_file, will be overwritten by file input
    query_words = []

    # adapt from until and output folder to fetch the data from a database
    if args["database"]:
        data = query_database.query_db_by_timespan_with_filewrite('2018-04-07 00:00:00', '2018-05-12 23:59:59','data/database_lndmhh_2018.csv')
        cleaned_data = remove_duplicates(data)
    else:
        # Read messages line by line from a text file and store it with an index as key into a json file:
        f = open(args["messages_from_file"], 'r', encoding='utf8')
        data = f.readlines()
        f.close()
        if data[0] == "\n":
            data = data[1:]
        cleaned_data = remove_duplicates(data)

    if os.path.isfile(preprocessed):
        os.remove(preprocessed)
    file = open(preprocessed, 'a', encoding='utf8')
    parent = {'Messages': []}
    place_holder_list = parent.get('Messages')

    if os.path.isfile(clusterable_words):
        os.remove(clusterable_words)
    f2 = open(clusterable_words, 'a', encoding='utf8')

    if os.path.isfile(clusterable_words_with_hashtags):
        os.remove(clusterable_words_with_hashtags)
    f3 = open(clusterable_words_with_hashtags, 'a', encoding='utf8')

    if os.path.isfile(clusterable_words_only_verbs):
        os.remove(clusterable_words_only_verbs)
    f4 = open(clusterable_words_only_verbs, 'a', encoding='utf8')

    if os.path.isfile(clusterable_words_only_adjectives):
        os.remove(clusterable_words_only_adjectives)
    f5 = open(clusterable_words_only_adjectives, 'a', encoding='utf8')

    if os.path.isfile(clusterable_words_only_nouns):
        os.remove(clusterable_words_only_nouns)
    f6 = open(clusterable_words_only_nouns, 'a', encoding='utf8')

    count = -1
    # The pipeline starts here - the order of calling the functions is important
    for line in cleaned_data:
        count += 1
        print(count)
        line_elements = line.split(';')
        date = line_elements[0]
        username = line_elements[1]
        query = line_elements[2].split()
        query_words = query
        lang = line_elements[3]
        text = line_elements[4]
        if len(line_elements) == 6:
            source = line_elements[5]
        elif (len(line_elements)) == 5:
            source = 'twitter'
        newMessage = message.Message(text, username, query, date, source)
        exception = newMessage.lang_detection()
        if newMessage.language_code not in ['en', 'de'] or exception == 'Missing Input':
            continue
        newMessage.get_emoticons()
        newMessage.tokenisation()
        newMessage.get_hashtags()
        newMessage.get_urls()
        newMessage.get_tags()
        newMessage.filter_tokens()
        newMessage.stem()
        newMessage.lemmatize_and_pos_and_activities()
        newMessage.seg()
        newMessage.ner()
        newMessage.get_clusterable_words()
        newMessage.get_moods()
        newMessage.get_sentiment()
        clusterable_words_string = ' '.join(map(str, newMessage.clusterable_words))
        f2.write(clusterable_words_string + '\n')
        clusterable_words_with_hashtags_string = ' '.join(map(str, newMessage.clusterable_words_with_hashtags))
        f3.write(clusterable_words_with_hashtags_string + '\n')
        clusterable_words_only_verbs_string = ' '.join(map(str, newMessage.clusterable_words_only_verbs))
        f4.write(clusterable_words_only_verbs_string + '\n')
        clusterable_words_only_adjectives_string = ' '.join(map(str, newMessage.clusterable_words_only_adjectives))
        f5.write(clusterable_words_only_adjectives_string + '\n')
        clusterable_words_only_nouns_string = ' '.join(map(str, newMessage.clusterable_words_only_nouns))
        f6.write(clusterable_words_only_nouns_string + '\n')
        place_holder_list.append(newMessage.__dict__)

    f2.close()  # Close the file after storing the clusterable words in it
    f3.close()
    f4.close()
    f5.close()
    f6.close()

    json_string = json.dumps(parent)
    datastore = json.loads(json_string)
    json_content = json.dumps(datastore, ensure_ascii=False, indent=4, sort_keys=True)
    file.write(json_content + '\n\n')
    file.close()