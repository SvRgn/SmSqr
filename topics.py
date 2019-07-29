#!/usr/bin/python
import prepare_infrastructure
import statistics
import os
import json
import argparse
import lda_tm
from itertools import groupby
from operator import itemgetter


def load_expert_terms(filename):
   f = open(filename, 'r+', encoding='utf8')
   data = f.readlines()
   expert_terms = ""
   for line in data:
      expert_terms = expert_terms+" "+line.rstrip('\n')
   print(expert_terms)
   f.close()
   return expert_terms

if __name__ == '__main__':

    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()

    # data file for training lda on
    ap.add_argument("-f", "--messages_from_file", required=False, type=str, default="data/test_topics.csv",
                    help="Path to local message file")

    # data file for testing
    ap.add_argument("-t", "--test_data", required=False, type=str, default="data/test.csv",
                    help="Path to local message file for testing")

    # cluster mode
    ap.add_argument("-c", "--cluster_mode", required=False, default="train", type=str,
                    help="Cluster mode can be test or train")

    # number of topics
    ap.add_argument("-to", "--topics", required=False, default="4", type=int,
                    help="Number of topics")

    # use tfidf for lda
    ap.add_argument("-i", "--tfidf", required=False, default=False, type=bool,
                    help="Setting to true if you want to use tfidf")

    # train on only nound, verbs, adjectives or every word type
    ap.add_argument("-w", "--word_type", required=False, type=str, default="",
                    help="Considering hashtags_, verbs_, nouns_ or adjectives_ in topics; leaving it emtpty takes all the word types")

    # general output folder
    ap.add_argument("-o", "--output_folder", required=False, default="output", type=str, help="Name of the output folder, not the path.")

    # path to expert term file
    ap.add_argument("-e", "--expert_term", required=False, default="tools/expert_terms.txt", type=str,
                    help="Path to file with expert terms")

    # statistics
    ap.add_argument("-st", "--statistics_period", required=False, default="complete", type=str,
                    help="Get statistics on a 'daily', 'monthly' (2018-05), 'weekly' (calender week starting with 0) basis, 'weekday' (monday etc), 'byhour' (only hour, no date) or for the 'complete' dataset.")
    # requery
    #ap.add_argument("-q", "--re_query", required=False, default="", type=str,
    #                help="Analyse collected messages with information of created topics, e.g. domplatz")

    args = vars(ap.parse_args())

    # prepare file names required for topic modelling using training data
    clusterable_words, preprocessed, model, wordcloud, statistics_period = prepare_infrastructure.prepare_file_name_topics(args["messages_from_file"], args["output_folder"], args['word_type'])

    # enter query words to obtain the data from messages_from_file, will be overwritten by file input
    query_words = ['hamburg']

    expert_terms = load_expert_terms(args["expert_term"])

    if args['cluster_mode'] == "train":
        print("INFO: Training a new topic model with " + str(args['topics']) + " topics.")
        lda_tm.train_topic_model(wordcloud, args['topics'], model, preprocessed, clusterable_words,
                             query_words, args["messages_from_file"], args["output_folder"], args["tfidf"], expert_terms)

    else:
        print("INFO: Testing an existing topic model")
        # prepare file names required for assigning clusters to test data
        clusterable_words_test_file, preprocessed, statistics_period = prepare_infrastructure.prepare_file_name_topics_test(args["test_data"], args["output_folder"], args['word_type'])
        lda_tm.test_topic_model(model, query_words, clusterable_words_test_file, args["output_folder"], preprocessed, args['test_data'], expert_terms)

    print("INFO: Creating statistics per period of time (" + args[
    'statistics_period'] + "), see " + statistics_period + ". Run display_period_statistics.py to plot statistics.")
    if os.path.isfile(statistics_period):
        os.remove(statistics_period)
    stat_file = open(statistics_period, 'a', encoding='utf8')
    stat_file_parent = {'Statistics': []}
    stat_place_holder_list = stat_file_parent.get('Statistics')

    with open(preprocessed, 'r', encoding='utf8') as data_file:
        preprocessed_file = json.load(data_file)


    newStatistic = statistics.Statistics(expert_terms, args['statistics_period'])
    preprocessed_parent = newStatistic.prepare_period(preprocessed_file)

    for dt, k in groupby(sorted(preprocessed_parent.get('Messages'), key=itemgetter('period')), key=itemgetter('period')):
        sub_parent = {'Messages': []}
        for d in k:
            place_holder_list = sub_parent.get('Messages')
            place_holder_list.append(d)
        num = newStatistic.get_number_of_messages(sub_parent)
        moods = newStatistic.get_no_messages_with_mood(sub_parent)
        senti = newStatistic.get_sentiments_of_text(sub_parent)
        acc, type = newStatistic.get_activities(sub_parent)
        no = newStatistic.get_no_activities(sub_parent)
        noUser, userWithTweets, mostCommon = newStatistic.get_users(sub_parent)
        expert, occurrence = newStatistic.get_expert_term_usage(sub_parent)
        service_use = newStatistic.get_service_usage(sub_parent)
        length = newStatistic.get_length_of_messages(sub_parent)
        language = newStatistic.get_language_of_messages(sub_parent)
        phrases = newStatistic.get_phrases(sub_parent)
        pos_tags = newStatistic.get_pos(sub_parent)
        topics = newStatistic.get_topic(sub_parent)

        periodDict = {
            'date': str(dt),
            'statistics_period': args['statistics_period'],
            'number_of_messages': num,
            'number_of_messages_with_mood': dict(moods),
            'sentiments_in_messages': dict(senti),
            'number_of_activities': no,
            'activities_by_occurence': acc,
            'types_of_activities': type,
            'number_of_active_users_in_period': noUser,
            'number_of_active_users': userWithTweets,
            'top_ten_twitterari': dict(mostCommon),
            'number_of_expert_terms_mentioned': expert,
            'user_mentioned_expert_term_n_times': occurrence,
            'service_usage': service_use,
            'text_length': length,
            'languages_in_period': language,
            'phrases_in_period': phrases,
            'most common_part_of_speech': pos_tags,
            'topics': topics
        }
        stat_place_holder_list.append(periodDict)

    stat_json = json.dumps(stat_file_parent)
    stat_datastore = json.loads(stat_json)
    stat_json_content = json.dumps(stat_datastore, ensure_ascii=False, indent=4, sort_keys=False)
    stat_file.write(stat_json_content + '\n\n')
    stat_file.close()

    # if os.path.isfile('output/clusterable_requeried.txt'):
    #     os.remove('output/clusterable_requeried.txt')
    # f3 = open('output/clusterable_requeried.txt', 'a', encoding='utf8')
    #
    # parent_re_queried = {'Re_queried_messages': []}
    # place_holder_list_re_queried = parent_re_queried.get('Re_queried_messages')
    # if args['re_query']:
    #     for a in parent.get("Messages"):
    #         if a.get("text"):
    #             # a = 'hello there tobi'
    #             b = args['re_query'].split(' ')
    #             if any(x in a.get('text') for x in b):
    #                 place_holder_list_re_queried.append(a)
    #
    #     mes = parent_re_queried.get('Re_queried_messages')
    #     for m in parent_re_queried.get('Re_queried_messages'):
    #         if m.get("username") == "josch73":
    #             print(str(m.get("clusterable_words")))
    #             out = ' '.join(m.get("clusterable_words"))
    #             f3.write(out + '\n')
    #     f3.close()
    #     json_string = json.dumps(parent_re_queried)
    #     datastore = json.loads(json_string)
    #     json_content = json.dumps(datastore, ensure_ascii=False, indent=4, sort_keys=True)
    #     if os.path.isfile('output/re_queried.json'):
    #         os.remove('output/re_queried.json')
    #     re_queried_data = open('output/re_queried.json', 'a', encoding='utf8')
    #     re_queried_data.write(json_content + '\n\n')
    #     re_queried_data.close()


    # --> call training with josch

    # queried_data = get_data.requery_database_by_user('11201847')
    # for message in queried_data:
    #     print(str(message[0]) + '; ' + str(message[1]) + '; ' + 'hamburg;' + ' ;"' + str(message[2]) + '"; ' + str(message[3]) + '\n')
    #
    # queried_data = get_data.requery_database_by_user_and_time('11201847', '2018-01-22 18:08:46', 1)
    # for message in queried_data:
    #     print(str(message[0]) + '; ' + str(message[1]) + '; ' + 'hamburg;' + ' ;"' + str(message[2]) + '"; ' + str(message[3]) + '\n')