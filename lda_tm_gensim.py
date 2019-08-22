# This program performs topic modelling of messages using PyPI's Latent Dirichlet Allocation(based on Gibb's sampling)
# Topic modelling associates a document (message in our case) to one or more topics with certain probability

# Library imports
import os
import json
import pickle
import statistics
import prepare_infrastructure
from wordcloud import WordCloud
from nltk.stem.snowball import SnowballStemmer
from message import stop_words_german
from collections import Counter
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
import matplotlib.pyplot as plt
import gensim
import gensim.corpora as corpora
from gensim import corpora, models
import pyLDAvis.gensim
import nltk
from nltk.corpus import stopwords
import string


def prepare_stopwords(query_words):
    nltk.download('stopwords', quiet=True)
    # Customizing the German stopwords list
    stop_words_german = stopwords.words('german')

    f = open("data/square_stopwords.txt", 'r', encoding='utf8')
    data = f.readlines()
    domain_stopwords = []
    for line in data:
        line = line.replace(" ", "")
        domain_stopwords.extend(line.split(','))
    stop_words_german.extend(domain_stopwords)
    stop_words_german.extend(query_words)

    cap_words_german = []
    all_upper_case_german = []
    for w in stop_words_german:
        cap_words_german.append(string.capwords(w))
        all_upper_case_german.append(w.upper())
    stop_words_german.extend(cap_words_german)
    stop_words_german.extend(all_upper_case_german)

    # Customizing the English stopwords list
    stop_words_english = stopwords.words('english')
    stop_words_english.extend(query_words)
    cap_words_english = []
    all_upper_case_english = []
    for w in stop_words_english:
        cap_words_english.append(string.capwords(w))
        all_upper_case_english.append(w.upper())
    stop_words_english.extend(cap_words_english)
    stop_words_english.extend(all_upper_case_english)

    return stop_words_german, stop_words_english



stemmer = SnowballStemmer('german')


def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit

    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = gensim.models.coherencemodel.CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())

    # Show graph
    x = range(1, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.savefig('output/topic_coherence.png')
    plt.show()
    plt.close()

    return lm_list, c_v


def stem_clusterable_words(clusterable_words):
    # Read training corpus of messages prepared for clustering purposes
    # (i.e. messages containing only clusterable words) for storing their stemmed version
    corpus = open(clusterable_words, 'r', encoding='utf8')
    data = corpus.readlines()
    # Create a 'stem:lemma' dictionary of clusterable words and create list 'new_data'
    # that contains the stems of clusterable words which will be used for clustering
    stem_lemma_dict = {}
    new_data = []
    for line in data:
        word_list = []
        for term in line.split():
            if "%%" in term:
                phrasecleaned = term.replace("%%", "").lower()
                stem_lemma_dict[phrasecleaned] = phrasecleaned
                word_list.append(phrasecleaned)
            else:
                stem_lemma_dict[stemmer.stem(term)] = term  # 'term' here is the clusterable word which is already in its lemmatized form
                # word_list.append(stemmer.stem(term))  # if clustering to be done using stemmed words
                word_list.append(term)  # if clustering to be done without stemming words
        new_line = ' '.join(word for word in word_list)
        new_data.append(new_line)
    return new_data, stem_lemma_dict


def train_topic_model(wordcloud_path, number_topics, model_path, preprocessed_path, clusterable_words, query_words, filename, out, use_tfidf, expert_terms):

    outfile, outfile_pos, wordcloud_file, wordcloud_json, statistic_topics_json = prepare_infrastructure.prepare_file_names_train(filename, out)
    stop_words_german, stop_words_english = prepare_stopwords(query_words)
    # Json file to store topics and their word distribution
    if os.path.isfile(wordcloud_path):
        os.remove(wordcloud_path)
    file = open(wordcloud_path, 'a', encoding='utf8')
    var1 = {'name': 'topics', 'children': []}
    place_holder_list = var1.get('children')

    # Uncomment one of the assignments to the variable 'infile' as per requirement
    # infile = clusterable_words     ## works well
    # infile = clusterable_words_with_hashtags      ## works well
    # infile = clusterable_words_only_verbs      ## throws error, very few words -> needs to be fixed
    # infile = clusterable_words_only_adjectives     ## throws error, very few words -> needs to be fixed
    # infile = clusterable_words_only_nouns     ## works well
    infile = clusterable_words

    # Removing stopwords from the clusterable words
    fin = open(infile, 'r', encoding='utf8')
    fout = open(outfile, "w+", encoding='utf8')
    for line in fin.readlines():
        for word in line.split():
            if not word in stop_words_german:
                fout.write(word+' ')
        fout.write('\n')
    fin.close()
    fout.close()

    # Learn the vocabulary dictionary and return term-document matrix (BOWs)
    # new_data, stem_lemma_dict = stem_clusterable_words(outfile_pos)
    new_data, stem_lemma_dict = stem_clusterable_words(outfile)
    # new_data_gensim = [n.split(' ') for n in new_data]

    empty_lines = []
    new_data_gensim = []
    for i, n in enumerate(new_data):
        if new_data[i] == '':
            empty_lines.append(i)
        else:
            new_data_gensim.append(n.split(' '))

    with open('output/new_data_gensim.sav', 'wb') as f:
        pickle.dump(new_data_gensim, f)

    topic_input = open('output/topic_input.txt', "w+", encoding='utf8')
    for k in new_data:
        topic_input.write(k + '\n')
    topic_input.close()


    # Create Dictionary
    id2word = corpora.Dictionary(new_data_gensim)  # dictionary of id and token
    # print(id2word)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in new_data_gensim]  # list of (token_id, token_count) tuples
    # list of (token_id, token_count) tuples

    # Prepare coherence graph, very time consuming
    # lmlist, c_v = evaluate_graph(dictionary=id2word, corpus=corpus, texts=new_data_gensim, limit=10)
    print(use_tfidf)

    if use_tfidf == False:
        print("INFO: no tfidf in use")
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=number_topics,chunksize=100, passes=10, eval_every = 1,
                                           random_state=1, alpha=0.2, eta=0.01, iterations=1000)
    else:
        print("INFO: tfidf in use")
        # TF-IDF LDA
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=number_topics, id2word=id2word, random_state=1, alpha=0.2, eta=0.01, iterations=1000)

    for i in range(0, number_topics):
        # print(lda_model.get_topic_terms(i))
        # print(type(lda_model.get_topic_terms(i)))
        pass
        for a, b in lda_model.get_topic_terms(i):
            # print(id2word.get(a) + ' ' + str(b))
            pass

    pickle.dump(lda_model, open(model_path, 'wb'))
    model = pickle.load(open(model_path, 'rb'))
    prepare_topic_distribution(model, number_topics, id2word, place_holder_list, stem_lemma_dict)
    save_word_cloud_json(var1, file)
    display_word_cloud(number_topics, wordcloud_file, wordcloud_json)
    save_train_topic_to_json(model, id2word, preprocessed_path, statistic_topics_json, expert_terms, wordcloud_path)
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, sort_topics=False)
    pyLDAvis.save_html(vis, 'output/LDA_Visualization.html')


def prepare_topic_distribution(model, number_topics, id2word, place_holder_list, stem_lemma_dict):
        for i in range(0, number_topics):
            print(model.get_topic_terms(i))
            word_cloud = {}
            key_word_name = "name"
            key_value_name = 'topic ' + str(i+1)
            key_word_children = "children"
            key_value_children = []
            word_cloud[key_word_name] = key_value_name
            word_cloud[key_word_children] = key_value_children
            for a, b in model.get_topic_terms(i):
                print(id2word.get(a) + ' ' + str(b))
                child_key_word_name = "name"
                try:
                    child_key_value_name = id2word.get(a)  # if words not stemmed for clustering
                    # child_key_value_name = (stem_lemma_dict.get(id2word.get(a))).lower()  # if words are stemmed before clustering
                except:
                    child_key_value_name = ''
                print('topic words...............: ', child_key_value_name)
                child_key_word_size = "size"
                child_key_value_size = int(round(b * 100))

                if child_key_value_size != 0:
                    key_value_children.append({child_key_word_name: child_key_value_name, child_key_word_size: child_key_value_size})
            place_holder_list.append(word_cloud)


def display_word_cloud(number_topics, wordcloud_file, wordcloud_json):
    # Display wordcloud for topics
    with open(wordcloud_json, 'r', encoding='utf8') as word_cloud_pos_json:
        data = json.load(word_cloud_pos_json)
        fig = plt.figure()
        i = 0
        for p in data['children']:
            ax = fig.add_subplot((number_topics / 2) + (number_topics % 2), 2, i + 1)
            ax.title.set_text('Topic: ' + str(i+1))
            word_count_dict = {}
            for q in p['children']:
                word_count_dict[q['name']] = q['size']

            wc = WordCloud(background_color="white", width=1000, height=1000, max_words=10, relative_scaling=0.5,
                           normalize_plurals=False).generate_from_frequencies(word_count_dict)
            i += 1
            ax.imshow(wc, interpolation="bilinear")
            ax.axis('off')
        fig.tight_layout()
        plt.show(fig)
        fig.savefig(wordcloud_file)


def save_word_cloud_json(var1, file):
    # Writing the word cloud to a json file
    json_string1 = json.dumps(var1)
    datastore1 = json.loads(json_string1)
    json_content1 = json.dumps(datastore1, ensure_ascii=False, indent=4, sort_keys=True)
    file.write(json_content1 + '\n\n')
    file.close()


def test_topic_model(model_path, query_words, filename, out, preprocessed, test_data, expert_terms, wordcloud_path):
    # Loading a pretrained model
    infile = filename
    outfile, statistic_test_topics_json = prepare_infrastructure.prepare_file_names_test(filename, out, test_data)
    stop_words_german, stop_words_english = prepare_stopwords(query_words)

    fin = open(infile, 'r', encoding='utf8')  # clusterable words from test data
    fout = open(outfile, "w+", encoding='utf8')  # will contain cleaned clusterable words from test data
    for line in fin.readlines():
        for word in line.split():
            if not word in stop_words_german:
                fout.write(word + ' ')
        fout.write('\n')
    fin.close()
    fout.close()
    new_texts, stem_lemma_dict = stem_clusterable_words(outfile)
    new_texts_gensim = [n.split(' ') for n in new_texts]
    model = pickle.load(open(model_path, 'rb'))
    with open('output/new_data_gensim.sav', 'rb') as f:
        new_data_gensim = pickle.load(f)

    id2word = corpora.Dictionary(new_data_gensim)

    test_corpus = [id2word.doc2bow(text) for text in new_texts_gensim]
    save_test_topic_to_json(test_corpus, model, preprocessed, statistic_test_topics_json, expert_terms, wordcloud_path)


def save_train_topic_to_json(model, id2word, preprocessed_path, statistic_topics_json, expert_terms, wordcloud_path):
    # Load json content obtained from Message class
    with open(preprocessed_path, 'r', encoding='utf8') as data_file:
        pre_processed_data = json.load(data_file)
    jsonFile = open(preprocessed_path, "w+", encoding='utf8')
    idx = 0
    topic_input = open('output/topic_input.txt', "r", encoding='utf8')
    topic_input_lines = topic_input.readlines()
    for json_tweet in pre_processed_data['Messages']:
        bow = id2word.doc2bow(topic_input_lines[idx].split())
        if not bow:
            json_tweet["topic"] = 'NO TOPIC'  # assigns topic '0' to tweet with no clusterable words left after cleaning
        else:
            t = model.get_document_topics(bow)
            topic_probabilities = {}
            for x, y in t:
                topic_probabilities[x] = y
            json_tweet["topic"] = str(max(topic_probabilities, key=topic_probabilities.get) + 1)
        idx += 1
    jsonFile.write(json.dumps(pre_processed_data, ensure_ascii=False, indent=4, sort_keys=True) + '\n\n')
    jsonFile.close()
    create_statistic_for_topic(preprocessed_path, statistic_topics_json, expert_terms, wordcloud_path)


def save_test_topic_to_json(test_corpus, model, preprocessed, statistic_test_topics_json, expert_terms, wordcloud_path):
    # Load json content obtained from Message class
    with open(preprocessed, 'r', encoding='utf8') as data_file:
        pre_processed_data = json.load(data_file)
    jsonFile = open(preprocessed, "w+", encoding='utf8')
    idx = 0
    for json_tweet in pre_processed_data['Messages']:
        unseen_doc = test_corpus[idx]
        # assigns topic '0' to tweet with no clusterable words left after cleaning or
        # if not a single word from tweet exists in trained model
        if len(unseen_doc) == 0:
            json_tweet["topic"] = 'NO TOPIC'
            idx += 1
        else:
            vector = model[unseen_doc]
            best_topic = max(vector, key=lambda x: x[1])
            json_tweet["topic"] = str(best_topic[0]+1)
            idx += 1

    jsonFile.write(json.dumps(pre_processed_data, ensure_ascii=False, indent=4, sort_keys=True) + '\n\n')
    jsonFile.close()
    create_statistic_for_topic(preprocessed, statistic_test_topics_json, expert_terms, wordcloud_path)


def create_statistic_for_topic(preprocessed_path, file, expert_terms, wordcloud_path):
    topic_file_parent = {'Statistics_for_Topic': []}
    topic_place_holder_list = topic_file_parent.get('Statistics_for_Topic')

    if os.path.isfile(file):
        os.remove(file)
    topic_file = open(file, 'a', encoding='utf8')

    with open(preprocessed_path, 'r', encoding='utf8') as data_file:
        pre_processed_data = json.load(data_file)
        newStatistic = statistics.Statistics(expert_terms, 'complete')

    with open(wordcloud_path, 'r', encoding='utf8') as wordcloud_file:
        wordcloud_json = json.load(wordcloud_file)

    for t, k in groupby(sorted(pre_processed_data.get('Messages'), key=itemgetter('topic')), key=itemgetter('topic')):
        sub_parent = {'Messages': []}
        topic_list = wordcloud_json['children']

        for item in topic_list:
            if item['name'] == str('topic ' + t):
                topic_sub_parent = item
                break

        for m in k:
            place_holder_list = sub_parent.get('Messages')
            place_holder_list.append(m)
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
        clusterable_words = newStatistic.get_clusterable_words(topic_sub_parent)

        topicDict = {
            'topic': str(t),
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
            'languages_in_topic': language,
            'phrases_in_period': phrases,
            'most_common_part_of_speech': pos_tags,
            'topic_keywords': clusterable_words
        }
        if topicDict['topic'] == 'NO TOPIC':
            topicDict['topic_keywords'] = None
        topic_place_holder_list.append(topicDict)

    topic_json = json.dumps(topic_file_parent)
    topic_datastore = json.loads(topic_json)
    topic_json_content = json.dumps(topic_datastore, ensure_ascii=False, indent=4, sort_keys=False)
    topic_file.write(topic_json_content + '\n\n')
    topic_file.close()