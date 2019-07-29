import json
import operator
import collections
import matplotlib.pyplot as plt
import numpy as np
import os

with open('output/statistic_per_topic_database_lndmhh_2018.json', 'r', encoding='utf8') as stats_file:
    stats_data = json.load(stats_file)

output_folder = "output"

#####################################
# Plotting for Statistics over topics
#####################################

output_folder_topic = output_folder + '/plots/topic/'

path = os.getcwd()
if not os.path.isdir(output_folder_topic):
    os.makedirs(path + output_folder_topic)

plotable_messages = []
sentiments = []
activities = []
activeUsers = []
topics = []
labels = []

for data in stats_data['Statistics_for_Topic']:
    plotable_messages.append(data['number_of_messages'])
    activities.append(data['number_of_activities'])
    activeUsers.append(data['number_of_active_users'])
    topics.append(data['topic'])
    labels.append(data['topic'])

format = np.arange(len(labels))

# Display number of messages over topics
plt.title('Number of Messages for Topics')
plt.ylabel('Number of Messages')
plt.xlabel('Topic')
plt.bar(topics, plotable_messages)
plt.xticks(format, labels, rotation=90, fontsize=8)
plt.savefig(output_folder_topic + 'number_of_messages_topics.png', bbox_inches='tight')
#plt.show()
plt.close()


# Display number of activities over topics
plt.title('Number of Activities detected in topics')
plt.ylabel('Number of Activities')
plt.xlabel('Topic')
plt.bar(topics, activities)
plt.xticks(format, labels, rotation=90, fontsize=8)
plt.savefig(output_folder_topic + 'number_of_activities_topics.png', bbox_inches='tight')
#plt.show()
plt.close()

# Display number of active users over topics
plt.title('Number of active users detected in topics')
plt.ylabel('Number of active Users')
plt.xlabel('Topic')
plt.bar(topics, activeUsers)
plt.xticks(format, labels, rotation=90, fontsize=8)
plt.savefig(output_folder_topic + 'number_of_active_users_topics.png', bbox_inches='tight')
#plt.show()
plt.close()


# Display expert terms in topic
for topic in stats_data['Statistics_for_Topic']:
    expert_terms = topic['number_of_expert_terms_mentioned']
    sorted_expert_terms = sorted(expert_terms.items(), key=operator.itemgetter(1), reverse=True)
    sorted_dict = collections.OrderedDict(sorted_expert_terms[:50])
    plt.title('Expert terms in topic '+topic['topic'])
    plt.subplots_adjust(bottom=0.2)
    plt.bar(range(len(sorted_dict)), list(sorted_dict.values()), align='center')
    plt.xticks(range(len(sorted_dict)), list(sorted_dict.keys()), rotation=90)
    plt.ylabel('Frequency')
    plt.savefig(output_folder_topic + 'expert_term_{}.png'.format(topic['topic']), bbox_inches='tight')
    #plt.show()
    plt.close()

# Display activities in topic
for topic in stats_data['Statistics_for_Topic']:
    activities = topic['activities_by_occurence']
    sorted_activities = sorted(activities.items(), key=operator.itemgetter(1), reverse=True)
    sorted_dict = collections.OrderedDict(sorted_activities[:20])
    plt.title('Top activities in topic '+topic['topic'])
    plt.subplots_adjust(bottom=0.2)
    plt.bar(range(len(sorted_dict)), list(sorted_dict.values()), align='center')
    plt.xticks(range(len(sorted_dict)), list(sorted_dict.keys()), rotation=90)
    plt.ylabel('Frequency')
    plt.savefig(output_folder_topic + 'activities_bar_{}.png'.format(topic['topic']), bbox_inches='tight')
    # plt.show()
    plt.close()


# Display mood distribution in topic
mood_keys = ['positive', 'negative', 'neutral']
for topic in stats_data['Statistics_for_Topic']:
    moods_total = topic['number_of_messages_with_mood']
    moods = {x: moods_total[x] for x in mood_keys}
    plt.title('Mood distribution in topic '+topic['topic'])
    plt.bar(range(len(moods)), list(moods.values()), align='center')
    plt.xticks(range(len(moods)), list(moods.keys()))
    plt.ylabel('Messages with mood')
    plt.savefig(output_folder_topic + 'moods_bar_{}.png'.format(topic['topic']), bbox_inches='tight')
    # plt.show()
    plt.close()

# Display sentiment distribution in topic
sentiment_keys = ['positive', 'negative', 'neutral']
for topic in stats_data['Statistics_for_Topic']:
    sentiments_total = topic['sentiments_in_messages']
    sentiments = {x: sentiments_total[x] for x in sentiment_keys}
    plt.title('Sentiment distribution in topic '+topic['topic'])
    plt.bar(range(len(sentiments)), list(sentiments.values()), align='center')
    plt.xticks(range(len(sentiments)), list(sentiments.keys()))
    plt.ylabel('Number of messages')
    plt.savefig(output_folder_topic + 'sentiments_bar_{}.png'.format(topic['topic']), bbox_inches='tight')
    # plt.show()
    plt.close()



# Display top nouns, adjectives, verbs and phrases in topic
for topic in stats_data['Statistics_for_Topic']:
    common_pos = topic['most_common_part_of_speech']
    noun_dict = {}
    adj_dict = {}
    verb_dict = {}
    phrase_dict = {}
    for i, pair in enumerate(common_pos):
        if '_NOUN' in pair[0]:
            noun_dict[pair[0].replace('_NOUN', '')] = pair[1]
        elif '_ADJ' in pair[0]:
            adj_dict[pair[0].replace('_ADJ', '')] = pair[1]
        elif '_VERB' in pair[0]:
            verb_dict[pair[0].replace('_VERB', '')] = pair[1]
        elif '_PHRASE' in pair[0]:
            cleaned_phrase = pair[0].replace('%%', ' ')
            cleaned_phrase = cleaned_phrase.replace('_PHRASE', '')
            phrase_dict[cleaned_phrase] = pair[1]

    plt.title('Top nouns in topic '+topic['topic'])
    plt.bar(range(len(noun_dict)), list(noun_dict.values()), align='center')
    plt.xticks(range(len(noun_dict)), list(noun_dict.keys()), rotation=90)
    plt.ylabel('Frequency')
    plt.savefig(output_folder_topic + 'nouns_bar_{}.png'.format(topic['topic']), bbox_inches='tight')
    # plt.show()
    plt.close()

    plt.title('Top adjectives in topic '+topic['topic'])
    plt.bar(range(len(adj_dict)), list(adj_dict.values()), align='center')
    plt.xticks(range(len(adj_dict)), list(adj_dict.keys()))
    plt.ylabel('Frequency')
    plt.savefig(output_folder_topic + 'adjectives_bar_{}.png'.format(topic['topic']), bbox_inches='tight')
    # plt.show()
    plt.close()

    plt.title('Top verbs in topic '+topic['topic'])
    plt.bar(range(len(verb_dict)), list(verb_dict.values()), align='center')
    plt.xticks(range(len(verb_dict)), list(verb_dict.keys()))
    plt.ylabel('Frequency')
    plt.savefig(output_folder_topic + 'verbs_bar_{}.png'.format(topic['topic']), bbox_inches='tight')
    # plt.show()
    plt.close()

    plt.title('Top phrases in topic '+topic['topic'])
    plt.subplots_adjust(bottom=0.5)
    plt.bar(range(len(phrase_dict)), list(phrase_dict.values()), align='center')
    plt.xticks(range(len(phrase_dict)), list(phrase_dict.keys()), rotation=60)
    plt.ylabel('Frequency')
    plt.savefig(output_folder_topic + 'phrases_bar_{}.png'.format(topic['topic']), bbox_inches='tight')
    # plt.show()
    plt.close()

# Display moods over topics
y_positive = []
y_negative = []
y_neutral = []
topic_cnt = 0
labels = []
mood_keys = ['positive', 'negative', 'neutral']
for topic in stats_data['Statistics_for_Topic']:
    moods_total = topic['number_of_messages_with_mood']
    moods = {x: moods_total[x] for x in mood_keys}
    y_positive.append(moods['positive'])
    y_negative.append(moods['negative'])
    y_neutral.append(moods['neutral'])
    labels.append(topic['topic'])
    topic_cnt += 1
ax = plt.subplot()
x_axis = np.arange(topic_cnt)
ax.bar(x_axis-0.2, y_positive, width=0.2, color='g', align='center', label='Positive')
ax.bar(x_axis, y_negative, width=0.2, color='r', align='center', label='Negative')
ax.bar(x_axis+0.2, y_neutral, width=0.2, color='b', align='center', label='Neutral')
ax.autoscale(tight=True)
plt.xticks(x_axis, labels, rotation=90, fontsize=8)
plt.title('Mood distribution over topics')
plt.xlabel('Topic')
plt.ylabel('Messages with mood')
plt.legend()
plt.savefig(output_folder_topic + 'mood_over_topics.png', bbox_inches='tight')
# plt.show()
plt.close()

# Display sentiments over topics
y_positive = []
y_negative = []
y_neutral = []
topic_cnt = 0
labels = []
mood_keys = ['positive', 'negative', 'neutral']
for topic in stats_data['Statistics_for_Topic']:
    moods_total = topic['sentiments_in_messages']
    moods = {x: moods_total[x] for x in mood_keys}
    y_positive.append(moods['positive'])
    y_negative.append(moods['negative'])
    y_neutral.append(moods['neutral'])
    labels.append(topic['topic'])
    topic_cnt += 1
ax = plt.subplot()
x_axis = np.arange(topic_cnt)
ax.bar(x_axis-0.2, y_positive, width=0.2, color='g', align='center', label='Positive')
ax.bar(x_axis, y_negative, width=0.2, color='r', align='center', label='Negative')
ax.bar(x_axis+0.2, y_neutral, width=0.2, color='b', align='center', label='Neutral')
ax.autoscale(tight=True)
plt.xticks(x_axis, labels, rotation=90, fontsize=8)
plt.title('Sentiment distribution over topics')
plt.ylabel('Number of messages')
plt.xlabel('Topic')
plt.legend()
plt.savefig(output_folder_topic + 'sentiment_over_topics.png', bbox_inches='tight')
# plt.show()
plt.close()

# Display top users in topic
for topic in stats_data['Statistics_for_Topic']:
    top_users = topic['top_ten_twitterari']
    sorted_users = sorted(top_users.items(), key=operator.itemgetter(1), reverse=True)
    sorted_dict = collections.OrderedDict(sorted_users)
    plt.title('Top users in topic '+topic['topic'])
    plt.subplots_adjust(bottom=0.4)
    plt.bar(range(len(sorted_dict)), list(sorted_dict.values()), align='center')
    plt.xticks(range(len(sorted_dict)), list(sorted_dict.keys()), rotation=90)
    plt.ylabel('Frequency')
    plt.savefig(output_folder_topic + 'top_users_bar_{}.png'.format(topic['topic']), bbox_inches='tight')
    # plt.show()
    plt.close()

# Display service usage in topic
for topic in stats_data['Statistics_for_Topic']:
    services = topic['service_usage']
    plt.title('Service usage in topic '+topic['topic'])
    plt.subplots_adjust(bottom=0.4)
    plt.bar(range(len(services)), list(services.values()), align='center')
    plt.xticks(range(len(services)), list(services.keys()), rotation=90)
    plt.ylabel('Frequency')
    plt.savefig(output_folder_topic + 'services_bar_{}.png'.format(topic['topic']), bbox_inches='tight')
    # plt.show()
    plt.close()

# Display languages in topic
y_english = []
y_german = []
topic_cnt = 0
labels = []
for topic in stats_data['Statistics_for_Topic']:
    languages = topic['languages_in_topic']
    if 'en' in languages:
        y_english.append(languages['en'])
    else:
        y_english.append(0)
    if 'de' in languages:
        y_german.append(languages['de'])
    else:
        y_german.append(0)
    labels.append(topic['topic'])
    topic_cnt += 1
ax = plt.subplot()
x_axis = np.arange(topic_cnt)
ax.bar(x_axis-0.2, y_english, width=0.2, color='g', align='center', label='English')
ax.bar(x_axis, y_german, width=0.2, color='b', align='center', label='German')
ax.autoscale(tight=True)
plt.xticks(x_axis, labels, rotation=90, fontsize=8)
plt.title('Language distribution over topics')
plt.ylabel('Number of messages')
plt.xlabel('Topic')
plt.legend()
plt.savefig(output_folder_topic + 'language_over_topics.png', bbox_inches='tight')
# plt.show()
plt.close()

# Display average text length and cleaned text length over topics
y_text = []
y_cleaned_text = []
topic_cnt = 0
labels = []
for topic in stats_data['Statistics_for_Topic']:
    text_length = []
    cleaned_text_length = []
    for i, message in enumerate(topic['text_length']):
        length_data = topic['text_length'][i]
        text_length.append(length_data[1])
        cleaned_text_length.append(length_data[2])
    text_length_avg = sum(text_length)/len(text_length)
    cleaned_text_length_avg = sum(cleaned_text_length)/len(cleaned_text_length)
    y_text.append(round(text_length_avg))
    y_cleaned_text.append(round(cleaned_text_length_avg))
    topic_cnt += 1
    labels.append(topic['topic'])

ax = plt.subplot()
x_axis = np.arange(topic_cnt)
ax.bar(x_axis-0.2, y_text, width=0.2, color='r', align='center', label='Original Text')
ax.bar(x_axis, y_cleaned_text, width=0.2, color='g', align='center', label='Cleaned Text')
ax.autoscale(tight=True)
plt.xticks(x_axis, labels, rotation=90, fontsize=8)
plt.title('Average text length over topics')
plt.ylabel('Average Length')
plt.xlabel('Topic')
plt.legend()
plt.savefig(output_folder_topic + 'text_length_over_topics.png', bbox_inches='tight')
# plt.show()
plt.close()


barWidth = 0.25
fig = plt.figure()
ax = fig.add_subplot(111)

y_facebook = []
y_instagram = []
y_fake = []
y_twitter = []

for i in range(topic_cnt):
    y_facebook.append(0)
    y_instagram.append(0)
    y_fake.append(0)
    y_twitter.append(0)

cnt = 0
for topics in stats_data['Statistics_for_Topic']:
    for i in topics['service_usage']:
        if i == 'instagram':
            y_instagram[cnt] = topics['service_usage'][i]
        elif i == 'facebook':
            y_facebook[cnt] = topics['service_usage'][i]
        elif i == 'twitter':
            y_twitter[cnt] = topics['service_usage'][i]
        elif i == 'fake':
            y_fake[cnt] = topics['service_usage'][i]
    cnt += 1

# Set position of bar on X axis
r1 = np.arange(len(y_instagram))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Make the plots
plt.bar(r1, y_facebook, color='r', width=barWidth, edgecolor='white', label='facebook')
plt.bar(r2, y_instagram, color='g', width=barWidth, edgecolor='white', label='instagram')
plt.bar(r3, y_twitter, color='b', width=barWidth, edgecolor='white', label='twitter')
plt.bar(r4, y_fake, color='y', width=barWidth, edgecolor='white', label='fake')

# Add xticks on the middle of the group bars
plt.ylabel('Messages')
plt.xlabel('Topic')
plt.xticks([r + barWidth for r in range(len(y_instagram))], labels, rotation=90, fontsize=8)

# Create legend & Show graphic
plt.legend()
plt.title('Service Usage over topics')
plt.savefig(output_folder_topic + 'service_usage_topics.png', bbox_inches='tight')
#plt.show()
plt.close()