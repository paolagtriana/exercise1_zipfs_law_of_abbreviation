######## LOADING THE DATA ##########
# Importing packages
import spacy
import en_core_web_sm
import es_core_news_sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter 

# File path of the datasets
dataset_english = "./dataset_english.txt"
dataset_spanish = "./dataset_spanish.txt"

# Opening the TXT files
with open(dataset_english, 'r') as file:
  data_en = file.read().replace('\n', ' ') # the .replace() method is used to replace the line break symbol (\n) with spaces

with open(dataset_spanish, 'r') as file:
  data_es = file.read().replace('\n', ' ')
  
# Defining functions that will be used later
def tokenize(doc): # function that tokenizes, removes punctuation from the list, and lowercases tokens
  words = [token.text.lower()
         for token in doc
         if not token.is_punct]
  return words

def table(words): # function that creates a dataframe with words and their length sorted by frequency
  df = pd.DataFrame.from_records(list(dict(Counter(words)).items()), columns=['word','frequency'])
  
  df = df.sort_values(by=['frequency'], ascending=False)
  df['rank'] = list(range(1, len(df) + 1))
  df['length'] = [len(token)
              for token in df['word']]
  
  return df

def plot(df, color_plot): # function to visualize the dataframe in a graph, arguments are the data analyzed and the desired color of the graph
  sns.set_theme(style="whitegrid") # setting the style of visualization
  sns.relplot(x="length", y="frequency", data=df, color=color_plot); # only length and frequency of words are included in the graph
  plt.show()
  plt.close()
  
  
############ ENGLISH DATASET ################
# Pre-processing the data
en_nlp = en_core_web_sm.load() # we download the English small model of spaCy as larger models are not needed for this analysis

en_doc = en_nlp(data_en) # we apply the model to the English dataset

en_words = tokenize(en_doc) # tokenizing words from the English dataset

# Zipf's Law Of Abbreviation
en_df = table(en_words) # creation of a dataframe with the length and the rank of words by frequency
print(en_df)

en_plot = plot(en_df, "blue") # visualization of the rank by frequency and length of words


############ SPANISH DATASET #############
# Pre-processing the data
es_nlp = es_core_news_sm.load() #  we download the available Spanish model of spaCy in the small version

es_doc = es_nlp(data_es)

es_words = tokenize(es_doc)

# Zipf's Law Of Abbreviation
es_df = table(es_words)
print(es_df)

es_plot = plot(es_df, "pink")
