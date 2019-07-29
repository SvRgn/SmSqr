from gensim.models.phrases import Phrases, Phraser

# Use this file with sentences within to train the gensim phraser
file = open('data/example_sentences.txt', 'r', encoding='utf8')
data = file.readlines()

#prepare data for phraser
sentence_stream = [line.split(" ") for line in data]

bigram = Phrases(sentence_stream, min_count=3, threshold=5, delimiter=b'%%')
trigram = Phrases(bigram[sentence_stream],  min_count=3, threshold=5, delimiter=b'%%')
bigram_phraser = Phraser(bigram)
trigram_phraser = Phraser(trigram)

# want to see the phrases use:
#print(trigram_phraser.phrasegrams.items())

bigram.save("phrase_bigram.model")
trigram.save("phrase_trigram.model")
bigram_phraser.save("phraser_bigram.model")
trigram_phraser.save("phraser_trigram.model")



#Testing Sentences...
sent1 = [u'der', u'Dialog', u'im', u'Dunkeln', u'in', u'Hamburg']
sent2 = [u'Samstag', u'ist', u'die', u'Lange', u'Nacht', u'der', u'Museen']
sent3 = [u'Sonderaustellung', u'Archäologische', u'Museum', u'Hamburg', u'lustig']
sent4 = [u'Early', u'Bird', u'Ticket', u'Hamburg']
sent5 = [u'Hafen', u'City', u'Hamburg']
sent6 = [u'FC', u'St', u'Pauli', u'spielt']
sent7 = [u'das', u'Tor', u'zur', u'Welt']
sent8 = [u'Museum', u'für', u'Hamburger', u'Geschichte']
sent9 = [u'Besuch', u'im', u'Alten', u'Elbtunnel']

#print(bigram_phraser[sent2])
print(trigram_phraser[bigram_phraser[sent2]])

#print(bigram_phraser[sent3])
print(trigram_phraser[bigram_phraser[sent3]])

#print(bigram_phraser[sent1])
print(trigram_phraser[bigram_phraser[sent1]])

#print(bigram_phraser[sent4])
print(trigram_phraser[bigram_phraser[sent4]])

print(trigram_phraser[bigram_phraser[sent5]])

print(trigram_phraser[bigram_phraser[sent6]])

print(trigram_phraser[bigram_phraser[sent7]])

print(trigram_phraser[bigram_phraser[sent8]])

print(trigram_phraser[bigram_phraser[sent9]])