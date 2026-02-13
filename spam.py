import pandas as pd
import numpy as np
import nltk
import spacy

from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

df = pd.read_csv('spam_data.csv', encoding='latin-1')

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
test_message = "I need you to help on this project."

test_message_tokenized = tokenizer.tokenize(test_message)
test_message_lowercased = [word.lower() for word in test_message_tokenized]

nlp = spacy.load('en_core_web_sm')

text = " ".join(test_message_lowercased)
doc = nlp(text)

lemmatized_tokens = [token.lemma_ for token in doc]
print("Lemmatized Words:", lemmatized_tokens)

stopwords_set = set(stopwords.words('english'))
html_tokens = [
    "font", "td", "br", "size", "b", "tr", "p", "face", "color", "width", "align",
    "center", "height", "table", "border", "href", "u", "html", "style", "div",
    "bgcolor", "src", "img", "value", "option", "text", "type", "span", "content",
    "body", "cellspacing", "li", "cellpadding", "blockquote", "input", "valign",
    "address", "strong", "left", "margin", "head", "order", "colspan", "title",
    "form", "tbody", "class", "alt", "meta", "link", "background", "id", "title",
    "name", "method", "action", "lang", "hidden", "submit", "padding", "document",
    "search", "server", "target", "hr", "ul", "em", "select", "small", "red",
    "plain", "option", "bordercolor", "charset", "equiv", "decoration", "label",
    "nbsp", "ptsize", "3d", "html", "-"
]
stopwords_set.update(html_tokens)

test_message_useful_tokens = [word for word in lemmatized_tokens if word not in stopwords_set]

def message_to_token_list(s):
    token = tokenizer.tokenize(s)
    lowercased_tokens = [word.lower() for word in token]
    text = " ".join(lowercased_tokens)
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    useful_tokens = [word for word in lemmatized_tokens if word not in stopwords_set]
    return useful_tokens
message_to_token_list(test_message)

token_counter = {}
for index, row in df.iterrows():
    if row['CATEGORY'] == 0:
        
        message_as_token_list = message_to_token_list(row['MESSAGE'])
        for token in message_as_token_list:
            if token in token_counter:
                token_counter[token] += 1
            else:
                token_counter[token] = 1

sorted_tokens = dict(sorted(token_counter.items(), key=lambda item: item[1], reverse=True))
sorted_tokens

keys = list(sorted_tokens.keys())[:50] 
values = list(sorted_tokens.values())[:50]  
plt.figure(figsize=(12, 6))

plt.bar(keys, values, color='blue')

plt.xlabel("Tokens")  
plt.ylabel("Frequency") 
plt.xticks(rotation=55, ha="right", fontsize=10) 
plt.title("Histogram of Token Frequencies for Non-Spam Emails")  

plt.show()

spam_tokens = ['helvetica', 'congratulations', 'list', 'free', 'click', 'image', 'get', 'please', 'send', 'gif','claim', 'ticket']

if not spam_tokens:
    raise ValueError("Error: The spam_tokens array is empty. Please go back to the previous code block and read the instructions carefully. \
                      You need to select words and type them into the spam_tokens array for this code to work.\
                      For example, you could type spam_tokens = ['win', 'prize'] and so on. ")

token_to_index_mapping = {t:i for t, i in zip(spam_tokens, range(len(spam_tokens)))}
token_to_index_mapping

def message_to_count_vector(message):
    count_vector = np.zeros(len(spam_tokens)) 

    processed_list_of_tokens = message_to_token_list(message)  

    for token in processed_list_of_tokens:  
        if token not in spam_tokens:  
            continue
        index = token_to_index_mapping[token]  
        count_vector[index] += 1  
    return count_vector 
message_to_count_vector('helvetica hi click on this free gif')

df = df.sample(frac=1, random_state=1).reset_index(drop=True)
split_index = int(len(df) * 0.8)
train_df, test_df = df[:split_index], df[split_index:]
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
print("train_df shape:", train_df.shape)
print("test_df shape:", test_df.shape)

index = 13
message_to_count_vector(train_df['MESSAGE'].iloc[index])
train_df.iloc[index]
print(train_df.iloc[index]['MESSAGE'])

def df_to_X_y(dff):
    y = dff['CATEGORY'].to_numpy().astype(int)

    message_col = dff['MESSAGE']

    count_vectors = []


    for message in message_col:

        count_vector = message_to_count_vector(message)

        # Append the vector to the list
        count_vectors.append(count_vector)

    X = np.array(count_vectors).astype(int)

    return X, y


X_train, y_train = df_to_X_y(train_df)


X_test, y_test = df_to_X_y(test_df)


X_train.shape, y_train.shape, X_test.shape, y_test.shape

scaler = MinMaxScaler().fit(X_train)

X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

X_train

lr = LogisticRegression().fit(X_train, y_train)

print(classification_report(y_test, lr.predict(X_test)))


rf = RandomForestClassifier().fit(X_train, y_train)

print(classification_report(y_test, rf.predict(X_test)))

def predict_message(message, model):
    vec = message_to_count_vector(message)
    vec = scaler.transform([vec]) 
    pred = model.predict(vec)[0]
    return "SPAM" if pred == 1 else "NOT SPAM"


user_message = "You just won a free ticket to the concert click here to claim your prize."
print("Message:", user_message)
print("Prediction (Logistic Regression):", predict_message(user_message, lr))
print("Prediction (Random Forest):", predict_message(user_message, rf))
