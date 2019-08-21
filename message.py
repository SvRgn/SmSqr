import nltk
import uuid
import re
import emojis
import emoji
import json
import string
import spacy
import gensim
from nltk.corpus import stopwords
from spacy_sentiws import spaCySentiWS
from langdetect import detect
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
# pip install vadersentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import iso639
from nltk import sent_tokenize
import io
import numpy as np
from ttp import ttp  # pip install twitter-text-python


# Parser for fetching @tags
p = ttp.Parser()

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

cap_words_german = []
all_upper_case_german = []
for w in stop_words_german:
    cap_words_german.append(string.capwords(w))
    all_upper_case_german.append(w.upper())
stop_words_german.extend(cap_words_german)
stop_words_german.extend(all_upper_case_german)

# Customizing the English stopwords list
stop_words_english = stopwords.words('english')
cap_words_english = []
all_upper_case_english = []
for w in stop_words_english:
    cap_words_english.append(string.capwords(w))
    all_upper_case_english.append(w.upper())
stop_words_english.extend(cap_words_english)
stop_words_english.extend(all_upper_case_english)


# Named Entity Recognition models
nlp_english = spacy.load("en_core_web_sm")
nlp_german = spacy.load("de_core_news_sm")
nlp_multi = spacy.load("xx_ent_wiki_sm")
sentiws = spaCySentiWS(sentiws_path='sentiws')
nlp_german.add_pipe(sentiws)

# Word embeddings obtained from 'https://github.com/facebookresearch/MUSE'
src_path = 'wiki.multi.en.vec'
tgt_path = 'wiki.multi.de.vec'
nmax = 200000  # maximum number of word embeddings to load

# Stemmer for fetching clusterable words after all messages have been translated to German
stemmer = SnowballStemmer('german')

# Regex to ignore urls and numbers in filtered text and named entities
url_regex = re.compile(r'http[s]?://(?:[a-züäöß]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', re.IGNORECASE)
# url_regex = re.compile(r'_^(?:(?:https?|ftp)://)(?:\S+(?::\S*)?@)?(?:(?!10(?:\.\d{1,3}){3})(?!127(?:\.\d{1,3}){3})(?!169\.254(?:\.\d{1,3}){2})(?!192\.168(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\x{00a1}-\x{ffff}0-9]+-?)*[a-z\x{00a1}-\x{ffff}0-9]+)(?:\.(?:[a-z\x{00a1}-\x{ffff}0-9]+-?)*[a-z\x{00a1}-\x{ffff}0-9]+)*(?:\.(?:[a-z\x{00a1}-\x{ffff}]{2,})))(?::\d{2,5})?(?:/[^\s]*)?$_iuS', re.IGNORECASE)
number_regex = re.compile(r'(?:(?:\d+,?)+(?:\.?\d+)?)')

# ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '…', '“', '”', '„']
punctuation_list = list(string.punctuation)
punctuation_list.extend(('…', '“', '”', '„', '``', "''"))


# Fetch activities from message
def clean_activities(word):
    res = True
    if any(exp in word for exp in ['http', 'twitter.com', "@", "#"]):
        res = False
    elif hasNumbers(word):
        res = False
    return res


def filter_named_entities(doc_ents, lang, query):
    if lang == 'german':
        stop_words = stop_words_german
    else:  # if self.language_name.lower()=='english':
        stop_words = stop_words_english
    # stop_words = stopwords.words(lang)
    cleaned_named_entities = [w.text for w in doc_ents if w.text.casefold() not in stop_words and w.text.casefold() not in ['rt', 'via'] and w.text not in emoji.UNICODE_EMOJI and not any(ext in w.text for ext in punctuation_list) and not re.match(url_regex, w.text) and not any(exp in w.text for exp in ['…', ' ', '@', 'http', 'twitter.com', '\n', '"']) and not re.match(number_regex, w.text)]
    return cleaned_named_entities

def lemma_for_one_word(word, lang):
    model = modeldict.get(lang)
    doc = model(word)
    for w in doc:
        lemma = w.lemma_
    return lemma


def extract_emojis(text):
    res = emojis.get(text)
    return res


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


def clean_token(token):
    if '„' in token: token = token.replace('„', '')
    if '“' in token: token = token.replace('“', '')
    if '...' in token: token = token.replace('...', '')
    if '?' in token: token = token.replace('?', '')
    if '!' in token: token = token.replace('!', '')
    if '+' in token: token = token.replace('+', '')
    if '—' in token: token = token.replace('—', '')
    # if # or @ are supposed to stay in the tokens just remove these two follwoing lines
    if '#' in token: token = ''
    if '@' in token: token = ''
    if token == ':/': token = ''

    if len(token) == 1: token = ''
    return token


# Loads the word embeddings
def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


src_embeddings, src_id2word, src_word2id = load_vec(src_path, nmax)
tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path, nmax)

# Dictionary containing the language code and their associated NER models
modeldict = {
        "en": nlp_english,
        "de": nlp_german,
        "multi": nlp_multi
    }

# Loading the Named Entity Recognition model
model_ner_multi = modeldict.get('multi')
model_ner_english = modeldict.get('en')
model_ner_german = modeldict.get('de')

with open("data/emoticons.json", 'r', encoding='utf8') as emo_file:
    json_emotis = json.load(emo_file)


class Message:

    def __init__(self, text, username, query, date, source):
        self.id = str(uuid.uuid4())
        self.username = username.strip()
        self.query = query
        self.date = date
        self.text = text.strip()
        self.service = source
        self.language_name = None
        self.language_code = None
        self.tokens = None
        self.filtered = None
        self.stemmed = None
        self.lemma = None
        self.sentences = None
        self.tagged = None
        self.entities = None
        self.topic = None
        self.clusterable_words = None
        self.hashtags = None
        self.emoticons = None
        self.activities = None
        self.moods = None
        self.sentiment = None
        self.tags = None
        self.urls = None
        self.clusterable_words_pos = None
        self.phrases = None
        self.clusterable_words_with_hashtags = None
        self.clusterable_words_only_verbs = None
        self.clusterable_words_only_adjectives = None
        self.clusterable_words_only_nouns = None

    # Language Identification
    def lang_detection(self):
        try:
            self.language_code = detect(self.text)
            self.language_name = iso639.to_name(self.language_code)
        except:
            return 'Missing Input'

    def find_phrases(self,tokens):
        resultingphrases = []
        trigram_phraser = gensim.models.phrases.Phraser.load("phraser/phraser_trigram.model")
        bigram_phraser = gensim.models.phrases.Phraser.load("phraser/phraser_bigram.model")
        phrases = trigram_phraser[bigram_phraser[tokens]]
        tokens = phrases
        for n, p in enumerate(phrases):
            if "%%" in p:
                for r in resultingphrases:
                    if r in p:
                        resultingphrases.remove(r)
                resultingphrases.append(p)
                tokens[n] ="quadroquadro"
        self.phrases = resultingphrases
        return tokens


    # Tokenisation (includes words, numbers, htmls, urls, @tags)
    def tokenisation(self):
        tokens = []
        # before = ""
        # print(self.text)
        for word in TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(self.text):
            word = clean_token(word)
            if not word == '': tokens.append(word)
        # print(str(tokens))

        # for word in nltk.word_tokenize(self.text):
        #     # fix for wrong tokenization splits # from word
        #     if word == '#':
        #         before = '#'
        #     if word == '@':
        #         before = '@'
        #     if word not in punctuation_list:
        #         if before == '@':
        #             word = '@'+word
        #             before = ''
        #         if before == '#':
        #             word = '#'+word
        #             before = ''
        #         word = clean_token(word)
        #         if not word == '' : tokens.append(word)
        phrased_tokens = self.find_phrases(tokens)
        self.tokens = phrased_tokens

    # Filtering stopwords, punctuations, terms like RT (used for re-tweets) and via (used to mention the original author of an
    # article or a re-tweet), user tags, urls and numbers
    def filter_tokens(self):
        if self.language_name.lower() == 'german':
            stop_words = stop_words_german
        else:   # if self.language_name.lower()=='english':
            stop_words = stop_words_english
        lang = self.language_code
        # print(stop_words_stem)
        # self.filtered = [w for w in self.tokens if w.casefold() not in stop_words and w.casefold() not in ['rt', 'via']  and not any(exp in w for exp in ['…', ' ', 'http', 'twitter.com', '\n', '"']) and w not in self.emoticons and not any(ext in w for ext in punctuation_list) and not re.match(url_regex, w) and not re.match(number_regex, w)]

        self.filtered = [w for w in self.tokens if (lemma_for_one_word(w, lang)).casefold() not in stop_words and
                         w.casefold() not in stop_words and w.casefold() not in ['rt', 'via'] and not any(
                             exp in w for exp in
                             ['…', ' ', 'http', 'www', 'twitter.com', '\n', '"']) and w not in self.emoticons and not re.match(url_regex, w) and not re.match(
                             number_regex, w) and not any(ext in w for ext in punctuation_list)]

    # Stemming
    def stem(self):
        stemmer = SnowballStemmer(self.language_name.lower())
        self.stemmed = [stemmer.stem(w) for w in self.filtered]

    # Lemmatization and Part Of Speech Tagging and fetching activities
    def lemmatize_and_pos_and_activities(self):
        pos_tags = {}
        activities = []
        # sent = []
        lemma = []
        model = modeldict.get(self.language_code)
        #if self.language_name.lower() == 'german':
        #    stop_words = stop_words_german
        #else:   # if self.language_name.lower()=='english':
        #    stop_words = stop_words_english
        # doc = model(self.text)
        test = " ".join(self.filtered)
        doc = model(test)
        # spacy uses a tokenizer that seperates #
        for word in doc:
            if '@' in word.text: continue
            elif 'http' in word.text: continue
            elif 'quadroquadro' == word.text: continue
            else:
                # sent.append(word.lemma_)
                lemma.append(word.lemma_)
                pos_tags[word.text] = word.pos_
                if word.pos_ == 'VERB':
                    if clean_activities(word.text):
                        activities.append(word.lemma_)
        for p in self.phrases:
            pos_tags[p] = 'PHRASE'

        self.activities = activities
        self.tagged = pos_tags
        self.lemma = lemma

    # Segmentation
    def seg(self):
        self.sentences = sent_tokenize(self.text)

    # Named Entity Recognition
    def ner(self):  # does not identify english tokens in german text and vice versa as an entity (eg. 'hammaburg' is ignored in an english text)
        # named_entities = {}
        # for sent in self.sentences:
        #     doc = model_ner(sent)
        #     cleaned_named_entities = filter_named_entities(doc.ents, self.language_name.lower())
        #     for token in doc.ents:
        #         if token.text in cleaned_named_entities:
        #             # if not re.match(url_regex, token.text) and not re.match(number_regex, token.text):
        #             named_entities[token.text] = token.label_
        # self.entities = named_entities
        named_entities = {}
        if self.language_name.lower() == 'german':
            doc = model_ner_german(self.text)
        else:
            doc = model_ner_english(self.text)
        cleaned_named_entities = filter_named_entities(doc.ents, self.language_name.lower(), self.query)
        for token in doc.ents:
            if token.text in cleaned_named_entities:
                named_entities[token.text] = token.label_
        self.entities = named_entities

    # Find list of clusterable words from English and German messages
    def get_clusterable_words(self):
        clusterable_words = []
        clusterable_words_pos = {}
        clusterable_words_only_verbs = []
        clusterable_words_only_adjectives = []
        clusterable_words_only_nouns = []
        # clusterable_words_with_hashtags = []
        pos_tags = self.tagged
        phrases = self.phrases
        if self.language_code == 'en':
            # Fetch equivalent german translation using word embeddings
            self.clusterable_words, self.clusterable_words_pos, self.clusterable_words_only_verbs, self.clusterable_words_only_adjectives, self.clusterable_words_only_nouns = self.get_nn(src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, k=1)
            self.clusterable_words_with_hashtags = self.clusterable_words + self.hashtags
        elif self.language_code == 'de':
            [clusterable_words.append(lemma_for_one_word(word, 'de')) if word not in self.entities else clusterable_words.append(word) for word in self.filtered]
            if "quadroquadro" in clusterable_words:
                clusterable_words = list(filter(("quadroquadro").__ne__, clusterable_words))
                clusterable_words.extend(phrases)
            self.clusterable_words = list(set(clusterable_words))
            self.clusterable_words_with_hashtags = self.clusterable_words + self.hashtags
            for j in self.filtered:
                if j in self.tagged and pos_tags.get(j)=='VERB':
                    clusterable_words_only_verbs.append(lemma_for_one_word(j,'de'))
                if j in self.tagged and pos_tags.get(j)=='ADJ':
                    clusterable_words_only_adjectives.append(lemma_for_one_word(j,'de'))
                if j in self.tagged and (pos_tags.get(j)=='NOUN' or pos_tags.get(j)=="PROPN"):
                    clusterable_words_only_nouns.append(lemma_for_one_word(j,'de'))
            self.clusterable_words_only_verbs=list(set(clusterable_words_only_verbs))
            self.clusterable_words_only_adjectives=list(set(clusterable_words_only_adjectives))
            self.clusterable_words_only_nouns=list(set(clusterable_words_only_nouns))
            for word in self.filtered:
                if word not in self.entities:
                    clusterable_words_pos.update({lemma_for_one_word(word, 'de'): pos_tags.get(word)})
                else:
                    clusterable_words_pos.update({word: pos_tags.get(word)})

            if "quadroquadro" in clusterable_words_pos:
                clusterable_words_pos.pop("quadroquadro")
                for i in phrases:
                    clusterable_words_pos[i] = 'PHRASE'
            self.clusterable_words_pos = clusterable_words_pos

    # Returns German translations for filtered tokens of English messages
    def get_nn(self, src_emb, src_id2word, tgt_emb, tgt_id2word, k):
        clusterable_words = []
        clusterable_words_pos = {}
        clusterable_words_only_verbs = []
        clusterable_words_only_adjectives = []
        clusterable_words_only_nouns = []
        pos_tags = self.tagged
        phrases = self.phrases
        word2id = {v: k for k, v in src_id2word.items()}
        for j in self.filtered:
            if j in word2id:
                word_emb = src_emb[word2id[j]]
                scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
                k_best = scores.argsort()[-k:][::-1]
                for i, idx in enumerate(k_best):
                    clusterable_words.append(lemma_for_one_word(tgt_id2word[idx], 'de'))
                    clusterable_words_pos.update({lemma_for_one_word(tgt_id2word[idx], 'de'): pos_tags.get(j)})
                    if j in self.tagged and pos_tags.get(j)=='VERB':
                        clusterable_words_only_verbs.append(lemma_for_one_word(tgt_id2word[idx], 'de'))
                    if j in self.tagged and pos_tags.get(j)=='ADJ':
                        clusterable_words_only_adjectives.append(lemma_for_one_word(tgt_id2word[idx], 'de'))
                    if j in self.tagged and (pos_tags.get(j)=='NOUN' or pos_tags.get(j)=="PROPN"):
                        clusterable_words_only_nouns.append(lemma_for_one_word(tgt_id2word[idx], 'de'))
            elif lemma_for_one_word(j, 'en') in word2id:
                word_emb = src_emb[word2id[lemma_for_one_word(j, 'en')]]
                scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
                k_best = scores.argsort()[-k:][::-1]
                for i, idx in enumerate(k_best):
                    clusterable_words.append(lemma_for_one_word(tgt_id2word[idx], 'de'))
                    clusterable_words_pos.update({lemma_for_one_word(tgt_id2word[idx], 'de'): pos_tags.get(j)})
                    if j in self.tagged and pos_tags.get(j)=='VERB':
                        clusterable_words_only_verbs.append(lemma_for_one_word(tgt_id2word[idx], 'de'))
                    if j in self.tagged and pos_tags.get(j)=='ADJ':
                        clusterable_words_only_adjectives.append(lemma_for_one_word(tgt_id2word[idx], 'de'))
                    if j in self.tagged and (pos_tags.get(j)=='NOUN' or pos_tags.get(j)=="PROPN"):
                        clusterable_words_only_nouns.append(lemma_for_one_word(tgt_id2word[idx], 'de'))
            elif j in self.entities:
                clusterable_words.append(j)
                clusterable_words_pos.update({j: pos_tags.get(j)})
                clusterable_words_only_nouns.append(j)
            elif '#' in j:
                clusterable_words.append(j)
                clusterable_words_pos.update({j: pos_tags.get(j)})
            else:
                continue
        if "quadroquadro" in clusterable_words:
            clusterable_words = list(filter(("quadroquadro").__ne__, clusterable_words))
            clusterable_words.extend(phrases)

        if "quadroquadro" in clusterable_words_pos:
            clusterable_words_pos.pop("quadroquadro")
            for i in phrases:
                clusterable_words_pos[i] = 'PHRASE'

        return list(set(clusterable_words)), clusterable_words_pos, clusterable_words_only_verbs, clusterable_words_only_adjectives, clusterable_words_only_nouns

    # Fetch Hashtags from message
    def get_hashtags(self):
        ######### PLEASE DO NOT DELETE THE COMMENTS BELOW ###################
        hashtags_str = r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"  # hash-tags
        regex_str = [
            hashtags_str
        ]
        hashtags_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
        self.hashtags = hashtags_re.findall(self.text)
        # result = p.parse(self.text)
        # self.hashtags = result.tags

    def get_urls(self):
        result = p.parse(self.text)
        self.urls = result.urls

    # Fetch @tags from message
    def get_tags(self):
        result = p.parse(self.text)
        self.tags = result.users

    # Fetch emoticons from message
    def get_emoticons(self):
        emo = extract_emojis(self.text)
        if emo:
            self.emoticons = emo
        elif not emo:
            emoticons_str = r"""
                (?:
                    [:=;] # Eyes
                    [oO\-]? # Nose (optional)
                    [DOPpCc8\)\]\(\[\\\|\<\>\{\}] # Mouth
                )"""
            omg_str = r"""(?:\\o\/)"""

            regex_str = [
                emoticons_str,
                omg_str
            ]
            emoticons_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
            self.emoticons = emoticons_re.findall(self.text)

    # Fetch moods from message (emoji or string-based)
    def get_moods(self):
        self.moods = None
        analyser = SentimentIntensityAnalyzer()
        score = analyser.polarity_scores(" ".join(self.emoticons))
        comp = score.get('compound')
        if comp >= 0.05:
            self.moods = "positive"
        elif (comp > -0.05 and comp < 0.05):
            self.moods = "neutral"
        elif comp <= -0.05:
            self.moods = "negative"

        for e in self.emoticons:
            if emoji.demojize(e):
                check = emoji.demojize(e)
                if check in json_emotis['Emojis']['activities']:
                    if not self.activities:
                        self.activities = []
                    self.activities.append(str(check))

        # if self.moods: print("Detected mood: ", self.moods)
        self.emoticons = " ".join(self.emoticons)

    # Fetch sentiment from message text
    def get_sentiment(self):
        text = " ".join(self.filtered)
        if self.language_code == "de":
            doc = nlp_german(text)
            sentiment_score=0
            for token in doc:
                # print('{}, {}, {}'.format(token.text, token._.sentiws, token.pos_))
                if token._.sentiws:
                    sentiment_score = sentiment_score + token._.sentiws
            f = float(sentiment_score)
            if f < 0:
                self.sentiment= "negative"
            elif f > 0:
                self.sentiment = "positive"
            else:
                self.sentiment = "neutral"
        elif self.language_code == "en":
            # Quelle: https://github.com/cjhutto/vaderSentiment
            # positive sentiment: compound score >= 0.05
            # neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
            # negative sentiment: compound score <= -0.05

            analyser = SentimentIntensityAnalyzer()
            score = analyser.polarity_scores(text)
            # print("{:-<40} {}".format(text, str(score)))
            comp= score.get('compound')
            if comp >= 0.05:
                self.sentiment = "positive"
            elif (comp > -0.05 and comp < 0.05):
                self.sentiment = "neutral"
            elif comp <= -0.05:
                self.sentiment = "negative"
        # print("Textual sentiment: " + self.sentiment)