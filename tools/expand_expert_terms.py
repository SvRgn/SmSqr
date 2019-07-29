# Script queries a german model trained on german wikipedia articles
# to get most similar words to a term
#
# input_file: line break seperated single words
# output_file: line break seperated words with resulting similar words
#
# Download the german model before running the code.
# Source of german.model: https://github.com/devmount/GermanWordEmbeddings
# wget.download('http://cloud.devmount.de/d2bc5672c523b086/german.model')

import gensim
import os
import wget

number_of_similar_words = 10

def remove_umlauts(string):
   if not "#" in string:
      if "ü" in string:
         string = string.replace("ü", "ue")
      elif "ö" in string:
         string = string.replace("ö", "oe")
      elif "ä" in string:
         string = string.replace("ä", "ae")
      elif "ß" in string:
         string = string.replace("ß", "ss")
   return string

def integrate_umlauts(string):
   if "ue" in string:
      string = string.replace("ue", "ü")
   elif "oe" in string:
      string = string.replace("oe", "ö")
   elif "ae" in string:
      string = string.replace("ae", "ä")
   return string

def load_expert_terms(filename):
   f = open(filename, 'r+', encoding='utf8')
   data = f.readlines()
   expert_terms = []
   for line in data:
      expert_terms.append(line.rstrip('\n'))
   f.close()
   return expert_terms


def prepare_output_file(filename):
   if os.path.isfile(filename):
      os.remove(filename)
   output_file = open(filename, 'w+', encoding='utf8')
   return output_file


print("Info: Loading expert terms")
terms = load_expert_terms("expert_terms_hammaburg_book.txt")
output_file = prepare_output_file("expanded_expert_terms_hammaburg_book.txt")


## Find most similar Words from german model
print("Info: Getting the ten most similar terms for expert terms")
#Source of german.model: https://github.com/devmount/GermanWordEmbeddings
#wget.download('http://cloud.devmount.de/d2bc5672c523b086/german.model')

model = gensim.models.KeyedVectors.load_word2vec_format("german.model", binary=True)
dublicate_terms = []
for term in terms:
   output_file.write(term + '\n')
   dublicate_terms.append(term)

   if "ä" in term or "ü" in term or "ö" in term or "ß" in term:
      term = remove_umlauts(term)

   # excluding hashtags
   if not "#" in term:
      print("\n---------------------------------")
      print("Info: Trying to find ", number_of_similar_words, " terms most similar to ", term, "...")
      try:
         for res in model.most_similar(term, topn=number_of_similar_words):
            t = res[0]
            if "_" in t: t = t.split("_")[-1]
            if "›" not in t and not t.isdigit():
               t = integrate_umlauts(t)
               if not t in dublicate_terms:
                  output_file.write(t + '\n')
                  dublicate_terms.append(t)
                  print(t)
      except:
         print("Error: Could not find most similar terms for " + term + "!")
      print("---------------------------------")


output_file.close()