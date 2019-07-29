import json
import operator
import collections
import matplotlib.pyplot as plt
import numpy as np
import os


with open('output/statistics_period_database_lndmhh_2018.json', 'r', encoding='utf8') as stats_period_file:
    stats_period_data = json.load(stats_period_file)

output_folder = "output"

##########################################
# Plotting for Statistics over time period
##########################################

output_folder_period = output_folder + '/plots/period/'
path = os.getcwd()
if not os.path.isdir(output_folder_period):
    os.makedirs(path + output_folder_period)

plotable_messages = []
plotable_dates = []
sentiments = []
activities = []
activeUsers = []
topics = []


for period in stats_period_data['Statistics']:
    plotable_messages.append(period['number_of_messages'])
    plotable_dates.append(period['date'])
    activities.append(period['number_of_activities'])
    activeUsers.append(period['number_of_active_users'])
    topics.append(period['topics'])




basis = stats_period_data['Statistics'][0].get('statistics_period')

#Number of messages
plt.title('Number of Messages over time - on a '+ basis +' basis')
plt.ylabel('Number of Messages')
plt.bar(plotable_dates, plotable_messages)
plt.xticks(rotation=60)
plt.subplots_adjust(bottom=0.2)
plt.savefig(output_folder_period +'number_of_messages_period.png', bbox_inches='tight')
#plt.show()
plt.close()


#Number of activities
plt.title('Number of Activities detected over time - on '+ basis +' basis')
plt.ylabel('Number of Activities')
plt.bar(plotable_dates, activities)
plt.xticks(rotation=60)
plt.subplots_adjust(bottom=0.3)
plt.savefig(output_folder_period +'number_of_activities_period.png', bbox_inches='tight')
#plt.show()
plt.close()

#Number of active users
plt.title('Number of active Users detected over time - on '+ basis +' basis')
plt.ylabel('Number of active Users')
plt.bar(plotable_dates, activeUsers)
plt.xticks(rotation=60)
plt.subplots_adjust(bottom=0.3)
plt.savefig(output_folder_period +'number_of_active_users_period.png', bbox_inches='tight')
#plt.show()
plt.close()


# Display expert terms in period
for period in stats_period_data['Statistics']:
    expert_terms = period['number_of_expert_terms_mentioned']
    sorted_expert_terms = sorted(expert_terms.items(), key=operator.itemgetter(1), reverse=True)
    sorted_dict = collections.OrderedDict(sorted_expert_terms[:50])
    plt.title('Expert terms in period '+period['date'])
    plt.subplots_adjust(bottom=0.2)
    plt.bar(range(len(sorted_dict)), list(sorted_dict.values()), align='center')
    plt.xticks(range(len(sorted_dict)), list(sorted_dict.keys()), rotation=90)
    plt.ylabel('Frequency')
    plt.savefig(output_folder_period + 'expert_term_{}.png'.format(period['date']), bbox_inches='tight')
    #plt.show()
    plt.close()

# Display mood distribution in period
mood_keys = ['positive', 'negative', 'neutral']
for period in stats_period_data['Statistics']:
    moods_total = period['number_of_messages_with_mood']
    moods = {x: moods_total[x] for x in mood_keys}
    plt.title('Mood distribution detected over time - on '+ basis +' basis: '+period['date'])
    plt.bar(range(len(moods)), list(moods.values()), align='center')
    plt.xticks(range(len(moods)), list(moods.keys()), rotation=50)
    plt.subplots_adjust(bottom=0.4)
    plt.ylabel('Messages with mood')
    plt.savefig(output_folder_period +'moods_bar_{}_period.png'.format(period['date']), bbox_inches='tight')
    # plt.show()
    plt.close()

# Display sentiment distribution in period
sentiment_keys = ['positive', 'negative', 'neutral']
for period in stats_period_data['Statistics']:
    sentiments_total = period['sentiments_in_messages']
    sentiments = {x: sentiments_total[x] for x in sentiment_keys}
    plt.title('Sentiment distribution detected over time - on '+ basis +' basis: ' + period['date'])
    plt.bar(range(len(sentiments)), list(sentiments.values()), align='center')
    plt.xticks(range(len(sentiments)), list(sentiments.keys()),  rotation=60)
    plt.ylabel('Number of messages')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(output_folder_period +'sentiments_bar_{}_period.png'.format(period['date']), bbox_inches='tight')
    # plt.show()
    plt.close()

# Display service usage in period of time
for period in stats_period_data['Statistics']:
    services = period['service_usage']
    plt.title('Service usage in over time - on a '+ basis +' basis: ' + period['date'])
    plt.subplots_adjust(bottom=0.4)
    plt.bar(range(len(services)), list(services.values()), align='center')
    plt.xticks(range(len(services)), list(services.keys()), rotation=60)
    plt.ylabel('Frequency')
    plt.savefig(output_folder_period + 'services_bar_{}_period.png'.format(period['date']), bbox_inches='tight')
    # plt.show()
    plt.close()

# Display languages in period of time
y_english = []
y_german = []
period_cnt = 0
labels = []
for period in stats_period_data['Statistics']:
    languages = period['languages_in_period']
    if 'en' in languages:
        y_english.append(languages['en'])
    else:
        y_english.append(0)
    if 'de' in languages:
        y_german.append(languages['de'])
    else:
        y_german.append(0)
    labels.append(period['date'])
    period_cnt += 1
ax = plt.subplot()
x_axis = np.arange(period_cnt)
ax.bar(x_axis-0.2, y_english, width=0.2, color='green', align='center', label='English')
ax.bar(x_axis, y_german, width=0.2, color='blue', align='center', label='German')
ax.autoscale(tight=True)
plt.xticks(x_axis, labels, rotation=60)
plt.subplots_adjust(bottom=0.2)
plt.title('Language distribution over time - on a '+ basis +' basis')
plt.ylabel('Number of messages')
plt.legend()
plt.savefig(output_folder_period + 'language_over_period.png', bbox_inches='tight')
#plt.show()
plt.close()


# Display average text length and cleaned text length over period of time
y_text = []
y_cleaned_text = []
period_cnt = 0
labels = []
for period in stats_period_data['Statistics']:
    text_length = []
    cleaned_text_length = []
    for i, message in enumerate(period['text_length']):
        length_data = period['text_length'][i]
        text_length.append(length_data[1])
        cleaned_text_length.append(length_data[2])
    text_length_avg = sum(text_length)/len(text_length)
    cleaned_text_length_avg = sum(cleaned_text_length)/len(cleaned_text_length)
    y_text.append(round(text_length_avg))
    y_cleaned_text.append(round(cleaned_text_length_avg))
    period_cnt += 1
    labels.append(period['date'])

ax = plt.subplot()
x_axis = np.arange(period_cnt)
ax.bar(x_axis-0.2, y_text, width=0.2, color='r', align='center', label='Original Text')
ax.bar(x_axis, y_cleaned_text, width=0.2, color='g', align='center', label='Cleaned Text')
ax.autoscale(tight=True)
plt.xticks(x_axis, labels, rotation=60)
plt.subplots_adjust(bottom=0.2)
plt.title('Average text length over time - on '+ basis +' basis')
plt.ylabel('Average Length')
plt.legend()
plt.savefig(output_folder_period +'text_length_period.png', bbox_inches='tight')
# plt.show()
plt.close()

# Display moods over time period
y_positive = []
y_negative = []
y_neutral = []
period_cnt = 0
labels = []
mood_keys = ['positive', 'negative', 'neutral']
for period in stats_period_data['Statistics']:
    moods_total = period['number_of_messages_with_mood']
    moods = {x: moods_total[x] for x in mood_keys}
    y_positive.append(moods['positive'])
    y_negative.append(moods['negative'])
    y_neutral.append(moods['neutral'])
    labels.append(period['date'])
    period_cnt += 1
ax = plt.subplot()
x_axis = np.arange(period_cnt)
ax.bar(x_axis-0.2, y_positive, width=0.2, color='g', align='center', label='Positive')
ax.bar(x_axis, y_negative, width=0.2, color='r', align='center', label='Negative')
ax.bar(x_axis+0.2, y_neutral, width=0.2, color='b', align='center', label='Neutral')
ax.autoscale(tight=True)
plt.xticks(x_axis, labels, rotation=60)
plt.subplots_adjust(bottom=0.3)
plt.title('Mood distribution over time - on a '+ basis +' basis')
plt.ylabel('Messages with mood')
plt.legend()
plt.savefig(output_folder_period +'mood_over_period.png', bbox_inches='tight')
#plt.show()
plt.close()

# Display sentiments over period of time
y_positive = []
y_negative = []
y_neutral = []
period_cnt = 0
labels = []
mood_keys = ['positive', 'negative', 'neutral']
for period in stats_period_data['Statistics']:
    moods_total = period['sentiments_in_messages']
    moods = {x: moods_total[x] for x in mood_keys}
    y_positive.append(moods['positive'])
    y_negative.append(moods['negative'])
    y_neutral.append(moods['neutral'])
    labels.append(period['date'])
    period_cnt += 1
ax = plt.subplot()
x_axis = np.arange(period_cnt)
ax.bar(x_axis-0.2, y_positive, width=0.2, color='g', align='center', label='Positive')
ax.bar(x_axis, y_negative, width=0.2, color='r', align='center', label='Negative')
ax.bar(x_axis+0.2, y_neutral, width=0.2, color='b', align='center', label='Neutral')
ax.autoscale(tight=True)
plt.xticks(x_axis, labels, rotation=60)
plt.subplots_adjust(bottom=0.3)
plt.title('Sentiment distribution over time - on '+ basis +' basis')
plt.ylabel('Number of messages')
plt.legend()
plt.savefig(output_folder_period +'sentiment_over_period.png', bbox_inches='tight')
#plt.show()
plt.close()

# Display top users in period
for period in stats_period_data['Statistics']:
    top_users = period['top_ten_twitterari']
    sorted_users = sorted(top_users.items(), key=operator.itemgetter(1), reverse=True)
    sorted_dict = collections.OrderedDict(sorted_users)
    plt.title('Top users in period '+ period['date']+ ' - on '+ basis +' basis')
    plt.subplots_adjust(bottom=0.4)
    plt.bar(range(len(sorted_dict)), list(sorted_dict.values()), align='center')
    plt.xticks(range(len(sorted_dict)), list(sorted_dict.keys()), rotation=60)
    plt.ylabel('Frequency')
    plt.savefig(output_folder_period +'top_users_bar_{}.png'.format(period['date']), bbox_inches='tight')
    #plt.show()
    plt.close()

# Display top nouns, adjectives, verbs and phrases in period
for period in stats_period_data['Statistics']:
    common_pos = period['most common_part_of_speech']
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

    plt.title('Top nouns in period '+ period['date'])
    plt.bar(range(len(noun_dict)), list(noun_dict.values()), align='center')
    plt.xticks(range(len(noun_dict)), list(noun_dict.keys()), rotation=60)
    plt.ylabel('Frequency')
    plt.savefig(output_folder_period +'nouns_bar_{}_period.png'.format(period['date']), bbox_inches='tight')
    #plt.show()
    plt.close()

    plt.title('Top adjectives in period '+ period['date'])
    plt.bar(range(len(adj_dict)), list(adj_dict.values()), align='center')
    plt.xticks(range(len(adj_dict)), list(adj_dict.keys()), rotation=60)
    plt.ylabel('Frequency')
    plt.savefig(output_folder_period +'adjectives_bar_{}_period.png'.format(period['date']), bbox_inches='tight')
    #plt.show()
    plt.close()

    plt.title('Top verbs in period '+ period['date'])
    plt.bar(range(len(verb_dict)), list(verb_dict.values()), align='center')
    plt.xticks(range(len(verb_dict)), list(verb_dict.keys()), rotation=60)
    plt.ylabel('Frequency')
    plt.savefig(output_folder_period +'verbs_bar_{}_period.png'.format(period['date']), bbox_inches='tight')
    #plt.show()
    plt.close()

    plt.title('Top phrases in period '+ period['date'])
    plt.subplots_adjust(bottom=0.5)
    plt.bar(range(len(phrase_dict)), list(phrase_dict.values()), align='center')
    plt.xticks(range(len(phrase_dict)), list(phrase_dict.keys()), rotation=60)
    plt.ylabel('Frequency')
    plt.savefig(output_folder_period +'phrases_bar_{}_period.png'.format(period['date']), bbox_inches='tight')
    #plt.show()
    plt.close()


# Display top verbs or activities in period
for period in stats_period_data['Statistics']:
    activities = period['activities_by_occurence']
    sorted_activities = sorted(activities.items(), key=operator.itemgetter(1), reverse=True)
    sorted_dict = collections.OrderedDict(sorted_activities[:20])
    plt.title('Top activities in period '+period['date'])
    plt.subplots_adjust(bottom=0.3)
    plt.bar(range(len(sorted_dict)), list(sorted_dict.values()), align='center')
    plt.xticks(range(len(sorted_dict)), list(sorted_dict.keys()), rotation=90)
    plt.ylabel('Frequency')
    plt.savefig(output_folder_period +'activities_bar_{}_period.png'.format(period['date']), bbox_inches='tight')
    #plt.show()
    plt.close()



# Display service usage in period
barWidth = 0.25
fig = plt.figure()
ax = fig.add_subplot(111)

y_facebook = []
y_instagram = []
y_fake = []
y_twitter = []

for i in range(period_cnt):
    y_facebook.append(0)
    y_instagram.append(0)
    y_fake.append(0)
    y_twitter.append(0)

cnt = 0
for period in stats_period_data['Statistics']:
    for i in period['service_usage']:
        if i == 'instagram':
            y_instagram[cnt] = period['service_usage'][i]
        elif i == 'facebook':
            y_facebook[cnt] = period['service_usage'][i]
        elif i == 'twitter':
            y_twitter[cnt] = period['service_usage'][i]
        elif i == 'fake':
            y_fake[cnt] = period['service_usage'][i]
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
plt.xticks([r + barWidth for r in range(len(y_instagram))], labels,  rotation=60)
plt.subplots_adjust(bottom=0.2)

# Create legend & Show graphic
plt.legend()
plt.title('Service Usage over time - on '+ basis +' basis')
plt.savefig(output_folder_period +'service_usage_period.png', bbox_inches='tight')
#plt.show()
plt.close()