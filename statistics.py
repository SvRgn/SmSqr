import uuid
from collections import Counter
from datetime import datetime, date


def get_date(datestring):
    # database date comes with seconds
    if (len(datestring) > 17) : datestring = datestring[:-3]
    date_time_obj = datetime.strptime(datestring, '%Y-%m-%d %H:%M')
    return (date_time_obj)

class Statistics:

    def __init__(self, expert_term, period):
        self.id = str(uuid.uuid4())
        self.timespan = None
        self.number_of_messages = None
        self.no_messages_with_mood = None
        self.no_sentiments = None
        self.activities_by_occurrences = None
        self.activities = None
        self.type_of_activities = None
        self.no_users = None
        self.users_with_number_of_messages = None
        self.expert_term = expert_term.split()
        self.occurrence_expert_term = None
        self.expert_term_by_user = None
        self.period = period
        self.service_usage = None
        self.text_length = None
        self.text_language = None
        self.topics = None
        self.phrases = None
        self.pos = None

    def prepare_period(self, parent):
        for a in parent.get("Messages"):
            if a.get("date").startswith("0"):
                dd = {'date': "2" + a.get("date")}
                a.update(dd)
            if self.period == 'weekly':
                dt = get_date(a.get("date")).strftime('%W')
                d = {'period': dt}
            elif self.period == 'weekday':
                dt = get_date(a.get("date")).strftime('%A')
                d = {'period': dt}
            elif self.period == 'byhour':
                dt = get_date(a.get("date")).strftime('%H')
                d = {'period': dt}
            elif self.period == 'daily':
                dt = get_date(a.get("date")).strftime('%Y-%m-%d')
                d = {'period': dt}
            elif self.period == 'monthly':
                dt = get_date(a.get("date")).strftime('%Y-%m')
                d = {'period': dt}
            elif self.period == 'complete':
                d = {'period': 'complete'}
            else:
                print("WARNING: Invalid statistical period declaration, please use daily, monthly or complete. Continuing with monthly")
                dt = get_date(a.get("date")).strftime('%Y-%m')
                d = {'period': dt}
            a.update(d)
        return(parent)



    def get_number_of_messages(self, sub_parent):
        self.number_of_messages = len(sub_parent.get('Messages'))
        #print("RESULT: Total number of messages in german and english: "+ str(self.number_of_messages))
        return self.number_of_messages

    def get_no_messages_with_mood(self, sub_parent):
        self.no_messages_with_mood = None
        total = []
        count = 0
        count_pos = 0
        count_neg = 0
        count_neut = 0
        for a in sub_parent.get("Messages"):
            if a.get("moods"):
                count += 1
                if a.get("moods") == "positive":
                    count_pos += 1
                elif a.get("moods") == "negative":
                    count_neg += 1
                elif a.get("moods") == "neutral":
                    count_neut += 1
                # print(a.get("moods"))
        total.append(['total', count])
        total.append(['positive', count_pos])
        total.append(['negative', count_neg])
        total.append(['neutral', count_neut])
        self.no_messages_with_mood = total
        #print("RESULT: Number of Messages with detected mood: " + str(self.no_messages_with_mood))
        return self.no_messages_with_mood

    def get_sentiments_of_text(self, sub_parent):
        self.no_sentiments = None
        total = []
        count = 0
        count_pos = 0
        count_neg = 0
        count_neut = 0
        for a in sub_parent.get("Messages"):
            if a.get("sentiment"):
                count += 1
                if a.get("sentiment") == "positive":
                    count_pos += 1
                elif a.get("sentiment") == "negative":
                    count_neg += 1
                elif a.get("sentiment") == "neutral":
                    count_neut += 1
        total.append(['total', count])
        total.append(['positive', count_pos])
        total.append(['negative', count_neg])
        total.append(['neutral', count_neut])
        self.no_sentiments = total
        #print("RESULT: Sentiments in text: " + str(self.no_sentiments))
        return self.no_sentiments

    def get_activities(self, sub_parent):
        self.activities_by_occurrences = []
        self.type_of_activities = []
        activs_total = []
        activs_only = []
        for a in sub_parent.get("Messages"):
            if a.get("activities"):
                for e in a.get("activities"):
                    activs_total.append(e)
                    activs_only.append(e) if e not in activs_only else activs_only

        self.activities_by_occurrences = Counter(activs_total)
        self.type_of_activities = activs_only
        #print("RESULT: Activities by occurence: ", self.activities_by_occurrences)
        #print("RESULT: Type of activities: ", self.type_of_activities)
        return self.activities_by_occurrences, self.type_of_activities

    def get_no_activities(self, sub_parent):
        self.no_activities = len(self.type_of_activities)
        #print("RESULT: Total number of activities found: ", self.no_activities)
        return self.no_activities

    def get_users(self, sub_parent):
            self.no_users = []
            self.users_with_number_of_messages = []
            users_total = []
            for a in sub_parent.get("Messages"):
                if a.get("username"):
                    users_total.append(a.get("username").lstrip())

            self.users_with_number_of_messages = Counter(users_total)
            self.no_users = len(self.users_with_number_of_messages)
            #print("RESULT: Users users with n.o. messages: ", self.users_with_number_of_messages)
            #print("RESULT: Total number of social media users texting: ", self.no_users)
            #print("RESULT: Top 10 users: ", self.users_with_number_of_messages.most_common(10))
            return self.users_with_number_of_messages, self.no_users, self.users_with_number_of_messages.most_common(10)

    def get_expert_term_usage(self, sub_parent):
        occurrence_expert_term = []
        user_per_expert_term = []
        if self.expert_term:
            for a in sub_parent.get("Messages"):
                if a.get("text"):
                    for e in self.expert_term:
                        if e in a.get("text"):
                            occurrence_expert_term.append(e)
                            user_per_expert_term.append(a.get("username").lstrip() + " " + e)

            self.expert_term_by_user = Counter(user_per_expert_term)
            self.occurrence_expert_term = Counter(occurrence_expert_term)
        print("RESULT: Expert terms mentioned in messages: ",self.occurrence_expert_term)
        print("RESULT: User mentioned expert term how many times: ", self.expert_term_by_user)
        return self.occurrence_expert_term, self.expert_term_by_user

    def get_service_usage(self, sub_parent):
        service_usage = []
        for a in sub_parent.get("Messages"):
            if a.get("service"):
                service_usage.append(a.get("service").lstrip().replace('\n', ''))
        self.service_usage = Counter(service_usage)
        return(self.service_usage)

    def get_length_of_messages(self, sub_parent):
        length_of_messages = []
        for a in sub_parent.get("Messages"):
            if a.get("tokens"):
                message = [a.get("id"), len(a.get("tokens")), len(a.get("clusterable_words"))]
                length_of_messages.append(message)
        self.text_length = length_of_messages
        #print("RESULT: Length of messages with ID: "+ str(self.text_length))
        return self.text_length

    def get_language_of_messages(self, sub_parent):
        language_of_messages = []
        for a in sub_parent.get("Messages"):
            if a.get("language_code"):
                language_of_messages.append(a.get("language_code"))
        self.text_language = Counter(language_of_messages)
        #print("RESULT: Language of messages: "+ str(self.text_language))
        return self.text_language


    def get_topic(self, sub_parent):
        topics_in_messages = []
        for a in sub_parent.get("Messages"):
            if a.get("topic"):
                topics_in_messages.append(a.get("topic"))
        self.topics = Counter(topics_in_messages)
        #print("RESULT: Topics in messages: "+ str(self.topics))
        return self.topics

    def get_phrases(self, sub_parent):
        phrases_in_messages = []
        for a in sub_parent.get("Messages"):
            if a.get("phrases"):
                for p in a.get("phrases"):
                    phrases_in_messages.append(p)
        self.phrases = Counter(phrases_in_messages)
        #print("RESULT: Phrases in messages: "+ str(self.phrases))
        return self.phrases

    def get_pos(self, sub_parent):
        pos_in_messages = []
        for a in sub_parent.get("Messages"):
            if a.get("tagged"):
                for word, speech in a.get('tagged').items():
                    pos_in_messages.append(word+'_'+speech)
                self.pos = Counter(pos_in_messages).most_common(10)
        #print("RESULT: most common words with POS in messages: "+ str(self.pos))
        return self.pos