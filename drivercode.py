import json
import re
import os.path
import glob
import pandas as pd
import numpy as np
import dataconfig as cfg


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas.io.json import json_normalize
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from nltk.util import ngrams
from nltk.corpus import stopwords
pd.set_option('max_columns', 22)



'''Please specify the values for variables filepath, output_file and
ontology_file according to the data used'''



in_filepath = cfg.in_filepath
out_filepath = cfg.out_filepath
filenamepattern = cfg.filenamepattern
ontology_file = cfg.ontology_file
inputfiles = in_filepath+"\*"


num_files = len([f for f in os.listdir(in_filepath) if os.path.isfile(os.path.join(in_filepath, f))])
print(num_files)


threshold_value = 60.00

def enhance_index(df):
    a=[]
    for i in df.index:
        if(i.split('_')[-1].isdigit()):
            a.append(i.split('_')[-2])
        else:
            a.append(i.split('_')[-1])
    return a


df = pd.DataFrame()
dict_list = pd.DataFrame()
if ontology_file.endswith('.json'):
    with open(ontology_file, encoding="utf8") as f:
        target_data = json.load(f)
    def flatten_json(nested_json):
        out = {}
        def flatten(x, name=''):
            if type(x) is dict:
                for a in x:
                    flatten(x[a], name + a + '_')
            elif type(x) is list:
                i = 0
                for a in x:
                    flatten(a, name + str(i) + '_')
                    i += 1
            else:
                out[name[:-1]] = x

        flatten(nested_json)
        return out
    d = flatten_json(target_data)
    df = pd.Series(d).to_frame()
    df['index'] = enhance_index(df)
    df.set_index('index', inplace=True)
    df['index'] = enhance_index(df)
    df.set_index('index', inplace=True)
    dict_list = (df.values.tolist())

elif ontology_file.endswith('.csv'):
    df = pd.read_csv(ontology_file, low_memory=False)
    dict_list = (df[['Preferred Label', 'Synonyms']].values.tolist())

print(df)

#dict_list = (df[['Preferred Label', 'Synonyms']].values.tolist())

target_list = [item for sublist in dict_list for item in sublist]
target_list = list(set(target_list))
target_list = [str(i) for i in target_list]
target_list = [x for x in target_list if not x.isdigit()]

print(target_list)

count=1
for name in glob.glob(inputfiles):
    print(name)
    with open(name, encoding="utf8") as file:
        input_data = json.load(file)
    dfinput = pd.DataFrame.from_dict(json_normalize(input_data), orient='columns')

    #print(dfinput.T)
    dfinput = dfinput.astype(str)
    dfinput['target_coli'] = dfinput.apply(lambda x: ' '.join(x), axis = 1)
    dfinput['target_coli'] = dfinput['target_coli'].str.replace('[^\w\s]','')
    dfinput['target_coli'] = dfinput['target_coli'].str.lower()

    #print(dfinput)
    ls = dfinput['target_coli'].tolist()
    res = ("".join(map(str, ls)))
    tokens = word_tokenize(res)
    text = [tokens for tokens in tokens if str(tokens) != 'nan']
    # print('\n')
    input_str = (' '.join(map(str, text)))
    # print(input_str)
    #ngrams
    def generate_ngrams(input_str, n):
        s = input_str.lower()
        s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
        tokens = [token for token in s.split(" ") if token != ""]
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return [" ".join(ngram).lower() for ngram in ngrams]


    onegram = generate_ngrams(input_str, 1)
    twogram = generate_ngrams(input_str, 2)
    threegram = generate_ngrams(input_str, 3)
    allgrams = onegram + twogram + threegram


    def cos_similarity(ngram):

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix_matched = tfidf_vectorizer.fit_transform(ngram)
        similarity_matrix = cosine_similarity(tfidf_matrix_matched[0:1], tfidf_matrix_matched)
        score_list = [item for sublist in similarity_matrix.tolist() for item in sublist]
        # print(score_list)
        return score_list


    #similarity score
    similarity_list = []
    a  = 0
    for i in target_list:
        allgrams.insert(0,i)
        similarity_score = cos_similarity(allgrams)
        similarity_score = list(filter((0.0).__ne__, similarity_score))
        similarity_score = sorted(similarity_score, reverse= True)
        similarity_list.append(similarity_score[1:2])
        allgrams.pop(0)
        a += 1

    similarity_df = pd.DataFrame({'Target': target_list, 'Similarity_Score': similarity_list })
    similarity_df = (similarity_df[similarity_df['Similarity_Score'].str.len() != 0])
    similarity_df.loc[:, 'Similarity_Score'] = similarity_df.Similarity_Score.map(lambda x: x[0])
    print(similarity_df)

    if not similarity_df.empty:
        similarity_df['Similarity_Score'] = pd.Series([float("{0:.2f}".format(val * 100)) for val in similarity_df['Similarity_Score']], index = similarity_df.index)
        similarity_df.loc[similarity_df.Similarity_Score >= float(threshold_value), 'Score_Threshold'] = 'Accepted'
        similarity_df = similarity_df.sort_values(['Similarity_Score'], ascending = False)
        print(similarity_df)
        threshold_df= similarity_df.dropna()
        threshold_df = threshold_df.sort_values(['Similarity_Score'], ascending = False)
        threshold_df = threshold_df.round(0)
        print(threshold_df)
        threshold = (threshold_df['Similarity_Score'].max() // 10) * 10
        print(threshold)
        if(threshold == 100):
            threshold = 90
        print(threshold)
        threshold_df = threshold_df.loc[threshold_df['Similarity_Score'] >= threshold]
        print(threshold_df)
        count_values = len(threshold_df)
        if not threshold_df.empty:

            tag_list = list(set(threshold_df["Target"]))

            new_dict = {}

            with open(name, encoding="utf8") as file:
                data = json.load(file)
            for i in data:
                new_dict.update(i)

            for i in new_dict["tags"]:
                tag_list.append(i)
            new_dict["tags"] = tag_list

            annotated_json = json.dumps(new_dict, indent=4)
            '''specify the file name to be generated'''
            output_file = str(count) + filenamepattern + ".json"
            outfile = os.path.join(out_filepath, output_file)
            with open(outfile, 'w') as file:

                json.dump(new_dict, file, indent=4)

            with open(outfile) as f:
                outdata = json.load(f)
            dftarget = pd.DataFrame.from_dict(json_normalize(outdata), orient='columns')


            count += 1
        else:

            # tag_list = list(set(threshold_df["Target"]))
            tag_list = []
            new_dict = {}
            with open(name, encoding="utf8") as file:
                data = json.load(file)
            for i in data:
                new_dict.update(i)

            for i in new_dict["tags"]:
                tag_list.append(i)
            new_dict["tags"] = tag_list

            annotated_json = json.dumps(new_dict, indent=4)
            '''specify the file name to be generated'''
            output_file = str(count) + filenamepattern + ".json"
            outfile = os.path.join(out_filepath, output_file)
            with open(outfile, 'w') as file:

                json.dump(new_dict, file, indent=4)

            with open(outfile) as f:
                outdata = json.load(f)
            dftarget = pd.DataFrame.from_dict(json_normalize(outdata), orient='columns')
            count += 1



    else:

        tag_list = []
        new_dict = {}
        with open(name, encoding="utf8") as file:
            data = json.load(file)
        for i in data:
            new_dict.update(i)

        for i in new_dict["tags"]:
            tag_list.append(i)
        new_dict["tags"] = tag_list

        annotated_json = json.dumps(new_dict, indent=4)
        '''specify the file name to be generated'''
        output_file = str(count) + filenamepattern + ".json"
        outfile = os.path.join(out_filepath, output_file)
        with open(outfile, 'w') as file:

            json.dump(new_dict, file, indent=4)

        with open(outfile) as f:
            outdata = json.load(f)
        dftarget = pd.DataFrame.from_dict(json_normalize(outdata), orient='columns')
        count += 1




