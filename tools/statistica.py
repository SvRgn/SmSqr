from operator import itemgetter
from itertools import groupby
import statistics
import argparse
import json
import os


###
#
# This code can be used to create statistics based on the pre-processed.json
# that is an outcome of the preprocessing of the main.py
#
# Arguments needed are:
#  pre-processed file (json)
#  text file with expert terms (one single word per line)
#  period of statistics (daily, monthly, weekly, weekday, byhour, complete)
#  name of the output folder name
###

def get_statistic_output(filename, folder):
    # extract filename
    name = filename.split('.')[0]
    # just to be sure
    folder = folder.replace("/", "")
    statistics_folder = folder + "/" + "statistics_period_" + name + ".json"
    return statistics_folder


def get_expert_terms(filename):
    f = open(filename, 'r+', encoding='utf8')
    data = f.readlines()
    expert_terms = ""
    for line in data:
        expert_terms = expert_terms + " " +line.rstrip('\n')
    return expert_terms

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

# data file for training, thats been trained on
ap.add_argument("-p", "--preprocessed", required=False, type=str, default="pre-processed_database_lndmhh_2018.json",
                help="Path to local message file")

# general output folder
ap.add_argument("-o", "--output_folder", required=False, default="output", type=str, help="Name of the output folder, not the path.")

# expert term file
ap.add_argument("-ex", "--expert_term_file", required=False, default="expert_terms.txt", type=str, help="Name of file that contains expert terms.")

# statistics
ap.add_argument("-st", "--statistics_period", required=False, default="weekly", type=str,
                help="Get statistics on a 'daily', 'monthly' (2018-05), 'weekly' (calender week starting with 0) basis, 'weekday' (monday etc), 'byhour' (only hour, no date) or for the 'complete' dataset.")

args = vars(ap.parse_args())

expert_terms = get_expert_terms(args['expert_term_file'])
print(expert_terms)

statistic_results = get_statistic_output(args['preprocessed'], args['output_folder'])
print (statistic_results)


print("INFO: Creating statistics per period of time (" + args['statistics_period'] + "), see " + statistic_results + ". Run display_period_statistics.py to plot statistics.")

if os.path.isfile(statistic_results):
    os.remove(statistic_results)
stat_file = open(statistic_results, 'w+', encoding='utf8')
stat_file_parent = {'Statistics': []}
stat_place_holder_list = stat_file_parent.get('Statistics')

with open(args['preprocessed'], 'r', encoding='utf8') as data_file:
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
print("INFO: Done creating statistics over time (" + args['statistics_period'] + "), see " + statistic_results + ". Run display_period_statistics.py to plot statistics.")

