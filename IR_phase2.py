import json
import pickle
from IR_phase1 import convert
#opening preprocessed file
file_path = '/Users/sara/Desktop/amirkabir/fall02-03/Information Retrieval/project/IR_data_news_12k_preprocessed.json'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        preprocessed_data = json.load(f)
        print("File opened successfully!")
except IOError:
    print("Error opening file.")

#creating positional index dic
positional_index_dic = {}
min_total = 3000
i = 0
# Iterate over each document in the preprocessed data
for docID, doc in preprocessed_data.items():
    # Iterate over each term and its position in the document
    for position, term in enumerate(doc['content']):
        # Check if the term is not in the positional index dictionary
        if term not in positional_index_dic:
            positional_index_dic[term] = {}
        # Check if the document ID is not in the positional index for the term
        if docID not in positional_index_dic[term]:
            positional_index_dic[term][docID] = {'count': 0, 'positions': []}

        positional_index_dic[term][docID]['count'] += 1
        # Add the position to the list of positions for the term in the document
        positional_index_dic[term][docID]['positions'].append(position)

        # Check if 'total' is not in the positional index for the term
        # Initialize a sub-dictionary for the total count
        if 'total' not in positional_index_dic[term]:
            positional_index_dic[term]['total'] = {'count': 0}
        positional_index_dic[term]['total']['count'] += 1
  
      
# print(positional_index_dic['روس'])
# print(positional_index_dic['مهر']['1053'])
# print(positional_index_dic['مهر']['total'])


output_file_path = '/Users/sara/Desktop/amirkabir/fall02-03/Information Retrieval/project/IR_data_news_12k_positional_index_dic.json'
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(positional_index_dic, f, ensure_ascii=False, indent=4)