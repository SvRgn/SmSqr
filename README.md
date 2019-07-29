# NLP Pipeline for SmartSquare #



## General Remarks ##
This is code for python3, please ensure the installment of the imports accordingly.

On the first run, the stopwords are downloaded, which takes a while
You may load the stopswords from nltk manually.

Also very time consuming at the first run is the download of the needed spacy models:

    python3 -m spacy download en_core_web_sm

The needed models are:

    en_core_web_sm
    de_core_news_sm
    xx_ent_wiki_sm

They can be downloaded manually as well, see https://spacy.io/models/en

Also needed to run the code the models:

    wiki.multi.de.vec
    wiki.multi.en.vec

Download a german word2vec model in tools before running the expert term expansion
via wget.download('http://cloud.devmount.de/d2bc5672c523b086/german.model')

Implement the database service layer, schema and factory in the folder 'db', also adapt the query_database.py to 
your data aquision accordingly (local database settings and credentials).

## Fetching Data ##
To fetch data one can query a database[1] or fetch data directly from twitter[2]

[1] To use data from a database a database service has to run on the host with the database of the hcu.
In the folder 'db' in 'db_factory.py' the credentials and settings for the database have to be set.
For changes in the database schema change the 'schema.py' in the 'db' folder.

To adapt the query of a database and to find all queries used in the project see 'query_database.py'.
Omitted services right now are 'facebook', 'rss', and 'googlealerts'. To change that see query in line 48.

[2] To query data from twitter use the 'tweet_collector.py' in the 'tools' folder.
The code can be run on a terminal, examples for the arguments setting are at the beginning of the code.
One needs to specify an output folder, the query word and a time span with 'since' and 'until', additionally the
langage of the tweets can be selected.

### Data Structure ###
To add data to the NLP pipeline, the data has to be structured as follows:

    <date><username_id><query word><language><text><social media service>

Example message vom csv file:

    2018-04-02 00:00:05; 11188822; hamburg; de; "#Hamburg #Spritpreis Diesel 1.183"; twitter

To run the NLP pipeline for the project two major steps have to be performed, the preprocessing and the topic modeling.

### Merge Hammaburg book with social media ###
In the 'tools' directory the script 'merge_bool_with_social_media.py' enables the merge of the sentences of the HCU 
Hammburg book with a selected social media csv file. It reconstructs the sentences of the book to the format for the NLP
pipeline. Please adapt the date to the time period of the real data set, if statistics are supposed to be made 
afterwards.

## Preprocessing ##
To preprocess data with the NLP pipeline run preprocessing.py.

### Preprocessing a csv file ###
The input file (csv) has to be set in the argument '--message_from_file' or -f. We added all
the test data in the 'data' folder of the project. The general output folder can be set via the
argument '--output_folder'.

### Preprocessing data from database ###
To query the data the argument '--database' or -d in 'preprocessing.py' has to be set to 'True'.
Additionally the timespan (from, until) has to be set in the the function call
'query_db_by_timespan_with_filewrite' in line 54. This data will be saved in a csv file as well.

To see the results of the preprocessing check the output folder.
The following files will be created:
- clusterable_words_<adjectives/hashtags/nouns/verbs><input_file_name>.txt (4 files)
- clusterable_words_<input_file_name>.txt (1 file)
- pre-processed_<file_name>.json (1 file)

In the pre-processed_<file_name>.json is the result of each preprocessing step saved for each message.
In the clusterable_words<*>.txt the input to the training and testing is saved. Each line represents one
message and its clusterable words (input for topics).
Are no words left from the preprocessing (e.g. only stopwords or urls on original text) the line representing
this message is empty. Please do not eliminate this line. the topic training will handle this case.

### Domain-specific stopwords ###
Stop words are words which are filtered out before preprocessing and training to get rid of most common words
in a language which are no information carrier for the topics.
In the file 'square_stopwords.txt' in the folder 'data' are comma seperated words that extend the used stopword
list from nltk. If words should be excluded add them comma seperated to that list.

### Get activities from Emoticons ###
To adapt the list of emoticons that are considered as a activity expressions in social media messages,
adapt the 'emoticons.json' in the 'data' folder. And add the shortcode of the emoticon to the "activities"
section of the json file. To get the shortcode of an emoticon check https://emojipedia.org for example.
These emojis are appended to the list of verbs (from POS-tagging) to indicate activities.

### Phrases ###
To detect phrases in the text of social media messages a phraser can be trained. To retrain the phraser 
(to find new phrases) use the 'training_phrases.py' in the 'phraser' folder.
It takes the sentences from '/data/example_sentences.txt'.

__Tip:__ Use different sentence structures because otherwise the sentences "Die Lange Nacht der Museem war schön" and
"Die Lange Nacht der Museen hat lange Schlangen" is resulting in the phrase "Die Lange Nacht der Museen".

## Topic modeling ##
To train or test data for topics the file 'topics.py' has to be used. 
During the project two topic modeling libraries where in use, gensim and scikit-learn. The implementations for that
can be found in 'lda_tm_gensim.py' or 'lda_tm_sklearn.py' To use one of them, rename the corresponding file to __lda_tm.py__.

To run the topic modeling code the following arguments have to be set:
- '--messages_from_file' or -f path to the data file (raw not preprocessed data) that needs topic modelling.
- '--test_data' or -t path to the data file with messages to be tested to an existing topic model.
- '--cluster_mode' or -c do we want to 'train' a new model or 'test' new data to an existing model.
- '--topics' or -to the number of topics to be created while training.
- '--tfidf' or -i set to true if ftidf is supposed to be used
- '--word_type' or t to set if only verbs, nouns, adjectives, words with hashtags or complete (only words) are considered
in the topics. For complete leave the argument empty.
- '--output_folder' or -o the name of the output folder for the results.
- '--statistics_period' or -st to set the period of time for the statistics that will be created, daily, monthly, weekly etc.
- '--expert_term' or -e to set the path to the file with the expert terms

__Example settings:__

To train data of cw 14 with 4 topics and only nouns in topics without tfidf, with expert terms of the hcu excel
and make statistics on a daily basis the arguments have to be set to:

    -f "data/database_kw_14_2018.csv"
    -c "train"
    -to "4"
    -i False
    -w "nouns_"
    -o "output"
    -e "tools/expert_terms.txt"
    -st "daily"


To train data of cw 14 with 50 topics and all words but no hashtags in topics with tfidf, with expert terms of the hcu excel
and make statistics on a weekly basis the arguments have to be set to:

    -f "data/database_kw_14_2018.csv"
    -c "train"
    -to "50"
    -i True
    -w ""
    -o "output"
    -e "tools/expert_terms.txt"
    -st "weekly"


To test data of cw 15 against the model of the cw 14 with 4 topics and without tfidf, with expanded expert terms of the
hcu excel and make statistics on the complete dataset the arguments have to be set to:

    -f "data/database_kw_14_2018.csv"
    -t "data/database_kw_15_2018.csv"
    -c "test"
    -to "4"
    -i False
    -w ""
    -o "output"
    -e "tools/expanded_expert_terms.txt"
    -st "complete"

__Keep in mind:__ the pre-processing has to be done in advance for both datasets and the output of it needs to be
directly in the output folder.


### Expert Terms ###
Expert terms is a domain-specific vocabulary that is evaluated contemplated and in addition to the query term.
The expert term is a term (e.g Hammabot) that appears in the messages that have been queried with the
query term (Hamburg) to fetch the data. The expert term is not considered in the pre-processing step but will be
evaluated in the topic modelling step.

To broaden the expert terms from the .txt file use the 'expand_expert_terms.py' in the tools folder.
A pre-trained model (on news and wikipedia articles) is used to find the 10 most similar words to a term. These are
written to a file named expanded_expert_terms.txt. To fetch more than 10 similar words change 'number_of_similar_words'
to the desired number. Keep in mind hashtags are omitted and are passed on to the output file as they are.

If this file is supposed to be used as expert terms, please adapt the argument '--expert_term' accordingly.

### Coherence graph ###
To get a coherence graph uncomment the evealuate_graph function in the 'lda_tm<algorithm>.py' file in the 
train_topic_model function. Please adapt the argument 'limit' to the number of topics. After the training a coherence
graph will be plotted in the output folder, named 'output/topic_coherence.png'.

### Results of Training and Testing ###
After training or testing the following files in the output folder have been created or changed:

- clusterable_words_<input_file_name>_cleaned.txt (created - further cleaned input),
- finalized_model_<input_file_name>.sav (trained model)
- new_data_gensim.sav ()
- topic_input.txt ()
- pre-processed_<input_file_name>.json (adapted, add topic number to each message)
- statistic_per_topic_<input_file_name>.json (created, needed to plot statistics)
- statistics_period_<input_file_name>.json (created, needed to plot statistics))
- word_cloud_<input filename>.json (created, needed to visualize the topics via d3)

#### Visualisation ####
For the visualisation of the resulting topics three options are given:
- a wordcloud visualisation has been created, see word_cloud_<data filename>.pdf
- to see an interactive bubble representation of the topics, see 'index.html' in the 'output' folder (adapt the filename
to the corresponding word_cloud_<input filename>.json in line 39 of the file)
- pyVizLDA, open LDA_visualisation.html from the output folder in a browser

To view the results in more detail plots can be created of the statistic files.
To create the plots of the two statistic files the two 'display_period_statistics.py' and
'display_topic_statistics.py' have to be used.

__Creating plots from the period-based statistics__
Use the 'display_period_statistics.py' file and adapt the path in the first line to use
the 'statistics_period_<input_file_name>.json' (is created in the output folder).
The output folder can also be adapted in the first lines.
The period is set while topic modeling with the argument '--statistics_period'.

__Creating plots from the topic-based statistics__
Use the 'display_topic_statistics.py' file and adapt the path in the first line to use
the 'statistics_per_topic_<input_file_name>.json' (is created in the output folder).
The output folder can also be adapted in the first lines.

__Create statistics without topic modelling__
To create the statistics 'statistics_period_<input_file_name>.json' without the topic modeling,
the script 'statistics.py' in the folder 'tools' can be used.
In the arguments of the script, the path to the preprocessed file, the output folder, the expert term file
and the period of the statistic (daily, weekly, etc.) can be given. Afterwards the statistics can
be plotted via the 'display_period_statistics.py' script.
