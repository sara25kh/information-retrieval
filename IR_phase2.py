import json
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
for docID, doc in preprocessed_data.items():
    for position, term in enumerate(doc['content']):
        if term not in positional_index_dic:
            positional_index_dic[term] = {}
        if docID not in positional_index_dic[term]:
            positional_index_dic[term][docID] = {'count': 0, 'positions': []}
        positional_index_dic[term][docID]['count'] += 1
        positional_index_dic[term][docID]['positions'].append(position)

        if 'total' not in positional_index_dic[term]:
            positional_index_dic[term]['total'] = {'count': 0}
        positional_index_dic[term]['total']['count'] += 1


#print(positional_index_dic['مهر'])
print(positional_index_dic['فوتبال']['1053'])
print(positional_index_dic['مهر']['total'])

# save positional index dic as a JSON file
output_file_path = '/Users/sara/Desktop/amirkabir/fall02-03/Information Retrieval/project/IR_data_news_12k_positional_index_dic.json'
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(positional_index_dic, f, ensure_ascii=False, indent=4)