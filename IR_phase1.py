# -*- coding: utf-8 -*-
import arabic_reshaper
from bidi.algorithm import get_display
import hazm
from hazm import *
import parsivar
from string import punctuation
import json
import re
import numpy as np
import collections
from collections import Counter
import concurrent.futures
from functools import partial



def convert(text):
    reshaped_text = arabic_reshaper.reshape(text)
    converted = get_display(reshaped_text)
    return converted

def convert_list(token_list):
    converted = []
    for idx, word in enumerate(token_list):
       
        converted.append(convert(word))
    return converted


file_path = '/Users/sara/Desktop/amirkabir/fall02-03/Information Retrieval/project/IR_data_news_12k.json'
def Load_Docs(): 
    try:
        f = open(file_path, 'r', encoding='utf-8')
        data_raw = json.load(f)
        print("File opened successfully!")
        f.close()
    except IOError:
        print("Error opening file.")

    data = {}
    for docID, body in data_raw.items():
        data[docID] = {}
        data[docID]['title'] = body['title']
        data[docID]['content'] = body['content']
        data[docID]['url'] = body['url']

    return data    


def tokenize(my_str):
    #print('Tokenizing...')
    result = re.split(' |\t|\n', my_str)
    result = list(filter(None, result))
    return result

def custom_normalize(tokens):
    i = 0
    normalized_tokens = []

    while i < len(tokens):
        current_token = tokens[i]
        previous_token = normalized_tokens[-1] if normalized_tokens else None

        # Unicode replacement
        current_token = current_token.replace('ي', 'ی').replace('ك', 'ک').replace('آ', 'ا').replace('﷽', 'بسم الله الرحمن الرحیم')
        # Additional unicode replacements
        current_token = current_token.replace("﷼", "ریال").replace("(ﷰ|ﷹ)", "صلی").replace("ﷲ", "الله").replace("ﷳ", "اکبر").replace("ﷴ", "محمد")
        current_token = current_token.replace("ﷵ", "صلعم").replace("ﷶ", "رسول").replace("ﷷ", "علیه").replace("ﷸ", "وسلم")
        current_token = re.sub(r'ﻵ|ﻶ|ﻷ|ﻸ|ﻹ|ﻺ|ﻻ|ﻼ', 'لا', current_token)

        # Convert English numbers to Persian numbers
        for j in range(10):
            current_token = current_token.replace(str(j), str(j).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")))


        # Remove specified characters
        current_token = re.sub(r"[\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652]", "", current_token)


        if current_token in {"ی", "ای", "ها", "های", "هایی", "تر", "تری", "ترین", "گر", "گری", "ام", "ات", "اش"}:
            if previous_token:
                # Correct spacing by merging with the previous token
                merged_token = previous_token + '\u200c' + current_token
                normalized_tokens[-1] = merged_token
                i += 1
            else:
               # There is no previous token (at the beginning), just append the current one
                normalized_tokens.append(current_token)
                i += 1
        elif current_token in {"می", "نمی"}:
            if i < len(tokens) - 1:
                # Combine with the next token if "می" or "نمی" is a single token
                next_token = tokens[i + 1]
                combined_token = current_token + '\u200c'+ next_token
                normalized_tokens.append(combined_token)
                i += 2 
            else:
                # The last token is "می" or "نمی", just append it
                normalized_tokens.append(current_token)
                i += 1
        else:
            # Check if the current token starts with "می" or "نمی" and fix spacing
            for prefix in {"می", "نمی"}:
                if current_token.startswith(prefix):
                    # Separate the prefix with the rest using half space
                    normalized_tokens.append(prefix + '\u200c' + current_token[len(prefix):])
                    break
            else:
                normalized_tokens.append(current_token)

            # Check if the current token ends with specific suffixes and fix spacing
            for suffix in {"ی", "ای", "ها", "های", "هایی", "تر", "تری", "ترین", "گر", "گری", "ام", "ات", "اش"}:
                if current_token.endswith(suffix):
                    # Separate the suffix with the previous part using half space
                    normalized_tokens[-1] = normalized_tokens[-1][:-1 * len(suffix)] + '\u200c' + suffix    

            i += 1

    return normalized_tokens



def delete_frequent_words(tokenized_docs):
    # Tokenize and normalize all documents
    # tokenized_docs = {doc_id: process_document(doc) for doc_id, doc in documents.items()}

    # Flatten the list of tokens from all documents
    all_tokens = [token for tokens in tokenized_docs.values() for token in tokens]

    # Calculate word frequencies
    word_frequencies = collections.Counter(all_tokens)

    # Find the top 50 most frequent words
    top_frequent_words = [word for word, _ in word_frequencies.most_common(50)]
    top_frequent_words_with_frq = [(word, word_frequencies[word]) for word, _ in word_frequencies.most_common(50)]

    # Deleted words info
    deleted_words_info = {
        'deleted_words': [],
        'deleted_words_frequencies': {}
    }

    # Remove top 50 frequent words from each document
    for doc_id, tokens in tokenized_docs.items():
        updated_tokens = [token for token in tokens if token not in top_frequent_words]
        deleted_words_info['deleted_words'].extend(list(set(tokens) - set(updated_tokens)))
        deleted_words_info['deleted_words_frequencies'][doc_id] = {word: word_frequencies[word] for word in set(tokens) - set(updated_tokens)}
        tokenized_docs[doc_id] = updated_tokens

    return tokenized_docs, deleted_words_info, top_frequent_words, top_frequent_words_with_frq


def stemming(updated_documents):
    stemmer = Stemmer()
    stemmed_words = [stemmer.stem(word) for word in updated_documents]
    return stemmed_words

def process_document(doc_id, doc):
    #print('processing doc', doc)
    return custom_normalize(tokenize(doc['content']))

documents = Load_Docs()
#print('document length', len(documents.items()))
tokenized_docs = {doc_id: process_document(doc_id, doc) for doc_id, doc in documents.items()}


# Example usage
updated_documents, deleted_words_info, overall_top_frequent_words, top_frequent_words_with_frq = delete_frequent_words(tokenized_docs)


# Print the overall top 50 frequent words
print("Overall Top 50 Frequent Words:")
print(convert_list(overall_top_frequent_words))

# Print the list of deleted words and their frequencies
print("Deleted words:")
for word, frq in top_frequent_words_with_frq:
    # print('info', info)
    print(f"{convert(word)}: {frq}")


print("Stemmed Words:" ,stemming(convert_list(overall_top_frequent_words)))


















""" Example usage
input_string = "امید چقد تر خوشگله می روم مطالعه میکنم نمیآیم میخواهم"
tokens = tokenize(input_string)
normalized_tokens = convert_list(custom_normalize(tokens))
#result = ' '.join(normalized_tokens)
print(normalized_tokens)"""



#print(convert_list(normalized_tokens))
#print(normalize(tokenize('salam  va omid  tgr \n efv5b 67')))
#original_tokens = ["یکی", "از", "مهم ترین", "اعداد", "6", "ترین", "گری", "ام", "می", "کند."]
#normalized_tokens = normalize(original_tokens)
#print(convert_list(normalized_tokens))
#print(convert_list(tokenize('امید‌بیب چقد خوشگله')))
#li = convert_list(tokenize('امید چقد تر خوشگله'))

