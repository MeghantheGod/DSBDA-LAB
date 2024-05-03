import pandas as pd
df = pd.read_csv("/home/meghan/Downloads/AlexaDataset.csv")
df.sample(5)
df['verified_reviews'][3]
df['verified_reviews'] = df['verified_reviews'].str.lower()
df['verified_reviews']
feedback = df['feedback'].value_counts()
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.pie(feedback, labels = ['Negative Feedback', 'Positive Feedback'],
        autopct = '%1.1f%%', colors = ['Red' , 'Blue'])
plt.title('Feedback of Customers')
plt.show()
import nltk
import re
def remove_html(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'',text)
df['verified_reviews'] = df['verified_reviews'].apply(remove_html)
df



df['verified_reviews'][2]
def remove_url(text):
    pattern = re.compile(r'https?://(?:[-\w.] | (?:%[\da-fA-F]{2}))+')
    return pattern.sub(r'', text)
df['verified_reviews'] = df['verified_reviews'][3].apply(remove_url)
import string
exclude = string.punctuation
exclude
def remove_punc(text):
    for char in exclude:
        text = text.replace(char,'')
        return text
df['verified_reviews'] = df['verified_reviews'][3].apply(remove_punc)
from textblob import TextBlob
def spell_correction(text):
    text1 = TextBlob(text)
    correct = text1.correct()
    return str(correct)
spell_correction(df['verified_reviews'][3])
df
text2 = 'heye girlish, wanna getk a drinks befored went going'
spell_correction(text2)
from nltk.corpus import stopwords
def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words ]
    filtered_text = ''.join(filtered_words)
    return filtered_text
df['verified_reviews'] = df['verified_reviews'].apply(rem_stopwords)
import emoji
def rem_emoji(text):
    return emoji.demojize(text)
df['verified_reviews'] =  df['verified_reviews'].apply(rem_emoji)
from nltk.tokenize import word_tokenize
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens
df['verified_reviews'] = df['verified_reviews'].apply(tokenize)
from nltk.stem import PorterStemmer
def stem(text):
    stemmer = PorterStemmer()
    stem_words = [stemmer.stem(word) for word in text]
    stem_text = ''.join(stem_words)
    return stem_text
df['verified_reviews'] = df['verified_reviews'].apply(stem)

from nltk.stem import WordNetLemmatizer
word_net = WordNetLemmatizer()
def lem(text):
    for word in text:
        print(word, word_net.Lemmatize(word, pos = 'v'))
 from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf.fit_transform(data_frame['text']).toarray()

data_frame = pd.DataFrame({'text':['people watch netflix','netflix watch␣
↪netflix','people write comment','netflix write comment','meghan watch␣
↪netflix','meghan write comment']})

print(bow[0].toarray())
print(bow[23].toarray())

bow=cv.fit_transform(df['verified_reviews'])