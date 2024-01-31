import json
import math
from string import punctuation
from hazm import *
import re
import math
import numpy as np
from IR_phase1 import preprocess_query , convert , convert_list
# from unidecode import unidecode

data = {}
positional_index_dic = {}
postings_list = {}
data_preprocessed = {}
champion_list = {}
docs = {}
N = 0



def tf_idf(nt, ftd):
    global N
    tf = 1 + math.log(ftd) if ftd > 0 else 0
    idf = math.log(N / nt)
    return tf * idf


def create_champion_list():
    global positional_index_dic, champion_list
    for term, postings in positional_index_dic.items():
        postings_list = []
        df = postings['total']['count']

        for docID, d in postings.items():
            if docID != 'total':
                tf = d['count']
                postings_list.append({'docID': docID, 'tf': tf})
        # Sort the postings list by TF in descending order
        postings_list.sort(key=lambda x: x['tf'], reverse=True)

        # Select the top N documents as champions
        N = min(10, len(postings_list))  # Example: Select top 10 documents
        champion_docs = [postings['docID'] for postings in postings_list[:N]]

        champion_list[term] = champion_docs

    return champion_list


def openFiles():
    global data, positional_index_dic, N, postings_list, data_preprocessed, docs
    # Opening positional index file
    file_path = '/Users/sara/Desktop/amirkabir/fall02-03/Information Retrieval/project/IR_data_news_12k_positional_index_dic.json'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            positional_index_dic = json.load(f)
            print("Positional Index File opened successfully!")
    except IOError:
        print("Error opening file.")

    # Opening origin file
    file_path = '/Users/sara/Desktop/amirkabir/fall02-03/Information Retrieval/project/IR_data_news_12k.json'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_raw = json.load(f)
            print("Origin File opened successfully!")
    except IOError:
        print("Error opening file.")

    # Opening preprocessed file
    file_path = '/Users/sara/Desktop/amirkabir/fall02-03/Information Retrieval/project/IR_data_news_12k_preprocessed.json'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_preprocessed = json.load(f)
            print("Preprocessed File opened successfully!")
    except IOError:
        print("Error opening file.")

    # print('Creating data...')
    for docID, body in data_raw.items():
        N = N + 1
        data[docID] = {}
        data[docID]['title'] = body['title']
        data[docID]['content'] = body['content']
        data[docID]['url'] = body['url']

    # print('Creating docs..')
    for docID, d in data_preprocessed.items():
        docs[docID] = d['content']
    max_tfidf = 0
    for term, postings in positional_index_dic.items():
        if term not in postings_list:
            postings_list[term] = []
        nt = postings['total']['count']
        for docID, da in postings.items():
            if docID != 'total':
                ftd = da['count']
                tfidf = tf_idf(nt, ftd)
                if max_tfidf > (tfidf):
                    print('max_tfidf:' , max_tfidf)
                postings_list[term].append({'docID': docID, 'tfidf': tfidf})

   

    # Creating champion list from positional index
    create_champion_list()


def normalize_vector(input):
    vector = []
    for term in input.items():
        vector.append(term[1])
    norm = math.sqrt(sum(x ** 2 for x in vector))
    # Check if norm is zero before dividing
    if norm == 0:
        # Handle the zero norm case (e.g., return a special value or raise an exception)
        return vector
    normalized_vector = [x / norm for x in vector]
    # normalized_vector = [x for x in vector]
    return normalized_vector


def calculate_query_vector(query):
    global positional_index_dic, N
    # Calculate term frequency (tf) in the query
    tf_query = {}
    for term in query:
        if term not in tf_query:
            tf_query[term] = 0
        tf_query[term] += 1

    # Calculate inverse document frequency (idf) for each query term
    
    idf_query = {}

    for term in tf_query:
        if term in positional_index_dic:
            nt = positional_index_dic[term]['total']['count']
            idf_query[term] = math.log(N / nt)
        else:
            idf_query[term] = 0  # Term not found in the positional index, assign idf as 0
    
   

    # Calculate query term weight (tf-idf)
    query_vector = {}
    for term in tf_query:
        tf = 1 + math.log(tf_query[term])
        tfidf = tf * idf_query[term]
        query_vector[term] = tfidf

    return query_vector


def calc_vectors(query):
    global postings_list, docs
    doc_vectors = {}
    for term in query.items():
        list = postings_list[term[0]]
        for l in range(len(list)):
            docID = list[l]['docID']
            if docID not in doc_vectors:
                doc_vectors[docID] = {}
            for t in docs.get(docID):
                docs_of_term = postings_list[t]
                for d in range(len(docs_of_term)):
                    if docs_of_term[d]['docID'] == docID:
                        doc_vectors[docID][t] = docs_of_term[d]['tfidf']
    return doc_vectors


def calc_vectors_cosine_by_champion(query):
    global postings_list, champion_list
    doc_vectors = {}
    for term in query.items():
        term_key = term[0]
        if term_key in postings_list:
            tf_idf_list = postings_list[term_key]
            docs_list = champion_list[term_key]
            for docID in docs_list:
                if docID not in doc_vectors:
                    doc_vectors[docID] = {}
                for i in range(len(tf_idf_list)):
                    if tf_idf_list[i]['docID'] == docID:
                        doc_vectors[docID][term_key] = tf_idf_list[i]['tfidf']
        
    for term, l in query.items():
        for docID, list in doc_vectors.items():
            if term not in list:
                doc_vectors[docID][term] = 0

    return doc_vectors


def calc_vectors_cosine(query):
    global postings_list
    doc_vectors = {}
    for term in query.items():
        list = postings_list[term[0]]
        for l in range(len(list)):
            docID = list[l]['docID']
            if docID not in doc_vectors:
                doc_vectors[docID] = {}
            doc_vectors[docID][term[0]] = list[l]['tfidf']

    for term, l in query.items():
        for docID, list in doc_vectors.items():
            if term not in list:
                doc_vectors[docID][term] = 0
    return doc_vectors




def cosine_similarity(query_vector, doc_vectors):
    similarities = {}
    for docID, doc_vector in doc_vectors.items():
        dot_product = 0
        a2 = 0
        b2 = 0
        # Calculate dot product and a2 b2
        for i in range(len(query_vector)):
            dot_product += query_vector[i] * doc_vector[i]
            a2 += query_vector[i] * query_vector[i]
            b2 += doc_vector[i] * doc_vector[i]

        cosine = dot_product / (math.sqrt(a2) + math.sqrt(b2))
        similarities[docID] = cosine

    # Sort the similarities in descending order
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    return sorted_similarities


def toPrint(sorted_docs):
    global data
    i = 0
    j = 0

    for doc in sorted_docs:
        if j < 10:
            docID = doc[0]
            score = doc[1]
            raw = data.get(docID)
            title = data[docID]['title']
            url = data[docID]['url']
            print(f'{i + 1}. DocID = {docID}  \n   Title = {convert(title)} \n   URL   = {convert(url)}')
            i += 1
            j = j + 1
            


def queryProcessor(query, mode ):
    global positional_index_dic, postings_list
    # preprocess query
    print(query)
    preprocessed_query = preprocess_query(query)
    print(preprocess_query)
    # calculating tf-idf
    query_tfidf = calculate_query_vector(preprocessed_query)
    # normalize query
    normalized_query_vector = {}
    normalized_query_vector = normalize_vector(query_tfidf)
    
    if mode == 0:  # without champion list
        print('ALL DATA')
        # calculating available docs tf-idf
        docs_vectors = []
        docs_vectors_eliminated = []
        docs_vectors = calc_vectors(query_tfidf)
        docs_vectors_eliminated = calc_vectors_cosine(query_tfidf)
        # normalize doc vectors
        normalized_docs_vector = {}
       
        for docID, vector in docs_vectors_eliminated.items():
            normalized_docs_vector[docID] = normalize_vector(vector)
        c_sim = cosine_similarity(normalized_query_vector, normalized_docs_vector)
        J = 10
        print('j: ', J)
        print('RESULTS BY COSINE SIMILARITY:')
        if len(c_sim) > 0:
            toPrint(c_sim)
            
        else:
            print('داده ای یافت نشد')
        print('************************************************************************************')
        
    else:
        # CHAMPION LIST
        print('CHAMPION LIST')
        docs_vectors_eliminatied_by_champion = []
        docs_vector_by_champion = []
        docs_vectors_eliminatied_by_champion = calc_vectors_cosine_by_champion(query_tfidf)
        normalized_query_vector = normalize_vector(query_tfidf)
        # normalize doc vectors
        normalized_docs_vector_by_champion = {}
        for docID, vector in docs_vectors_eliminatied_by_champion.items():
            normalized_docs_vector_by_champion[docID] = normalize_vector(vector)
        c_sim_by_champion = cosine_similarity(normalized_query_vector, normalized_docs_vector_by_champion)
        #docs_vector_by_champion = calc_vectors_by_champion(query_tfidf)
        
        print('RESULTS BY COSINE SIMILARITY:')
        if len(c_sim_by_champion) > 0:
            toPrint(c_sim_by_champion)
        else:
            print(convert('داده ای یافت نشد'))
        

    
def main():
    inputQ = input('Enter Query: \n')
    print(f'Input: {inputQ}')
    openFiles()
    queryProcessor(inputQ, 1)


if __name__ == "__main__":
    main()