import os


def create_environment(output_folder):
    print("INFO: Preparing folder structure")
    path = os.getcwd()
    output_folder = output_folder.replace("/", "")
    folder_list = ['/data', '/db', '/phraser', '/phraser/data', '/sentiws', '/tools']
    folder_list.append('/'+output_folder)
    folder_list.append('/'+output_folder+'/plots/period')
    folder_list.append('/' + output_folder + '/plots/topic')
    for f in folder_list:
        if not os.path.isdir(f):
            try:
                os.makedirs(path+f)
            except OSError:
                print("INFO: The directory %s exists already" % f)


def prepare_file_name_topics(arg_file, folder, input_type):

    # extract training data filename
    name = arg_file.rsplit('/', 1)[1].split('.')[0]

    # name_pre = test_file.rsplit('/', 1)[1].split('.')[0]
    # just to be sure
    folder = folder.replace("/", "")

    # training data clusterable words
    clusterable_words = folder + "/" + "clusterable_words_" + input_type + name + ".txt"

    # training data pre-processed filename
    preprocessed = folder + "/" + "pre-processed_" + name + ".json"

    # model prepared using training data
    model = folder + "/" + "finalized_model_" + name + ".sav"

    # training data lda word cloud
    wordcloud = folder + "/" + "word_cloud_" + name + ".json"

    # statistics file for training data
    statistics = folder + "/" + "statistics_period_" + name + ".json"

    # if test_file:
    #     preprocessed_test_file = folder + "/" + "pre-processed_" + name_pre + ".json"
    # else:
    #     preprocessed_test_file = ""

    return clusterable_words, preprocessed, model, wordcloud, statistics


def prepare_file_name_topics_test(test_data, folder, input_type):

    # test data clusterable words
    name = test_data.rsplit('/', 1)[1].split('.')[0]
    clusterable_words_test_file = folder + "/" + "clusterable_words_" + input_type + name + ".txt"

    # test data pre-processed filename
    preprocessed_test_file = folder + "/" + "pre-processed_" + name + ".json"

    # statistics file for training data
    statistics_period = folder + "/" + "statistics_period_" + name + ".json"

    return clusterable_words_test_file, preprocessed_test_file, statistics_period


def prepare_file_name_preprocessing(arg_file, folder):

    # extract filename
    name = arg_file.rsplit('/', 1)[1].split('.')[0]

    # just to be sure
    folder = folder.replace("/", "")

    clusterable_words = folder + "/" + "clusterable_words_" + name + ".txt"

    clusterable_words_with_hashtags = folder + "/" + "clusterable_words_hashtags_" + name + ".txt"

    clusterable_words_only_verbs = folder + "/" + "clusterable_words_verbs_" + name + ".txt"

    clusterable_words_only_adjectives = folder + "/" + "clusterable_words_adjectives_" + name + ".txt"

    clusterable_words_only_nouns = folder + "/" + "clusterable_words_nouns_" + name + ".txt"

    preprocessed = folder + "/" + "pre-processed_" + name + ".json"

    return clusterable_words, clusterable_words_with_hashtags, clusterable_words_only_verbs, clusterable_words_only_adjectives, clusterable_words_only_nouns, preprocessed


def prepare_file_names_train(arg_file, folder):

    # extract filename
    name = arg_file.rsplit('/', 1)[1].split('.')[0]

    # just to be sure
    folder = folder.replace("/", "")

    outfile = folder + "/" + "clusterable_words_" + name + "_cleaned.txt"

    outfile_pos = folder + "/" + "clusterable_words_pos_" + name + "_cleaned.txt"

    wordcloud_file = folder + "/" + "word_cloud_" + name + ".pdf"

    wordcloud_json = folder + "/" + "word_cloud_" + name + ".json"

    statistic_topics_json = folder + "/" + "statistic_per_topic_" + name + ".json"

    return outfile, outfile_pos, wordcloud_file, wordcloud_json, statistic_topics_json


def prepare_file_names_test(arg_file, folder, test_data):

    # extract filename
    name = arg_file.rsplit('/', 1)[1].split('.')[0]
    stat_suffix = test_data.rsplit('/', 1)[1].split('.')[0]
    # just to be sure
    folder = folder.replace("/", "")

    outfile = folder + "/" + name + "_cleaned.txt"

    statistic_test_topics_json = folder + "/" + "statistic_per_topic_" + stat_suffix + ".json"

    return outfile, statistic_test_topics_json