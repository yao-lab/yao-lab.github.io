import argparse
import re
import sys
import pandas as pd
import numpy as np
import gensim 
from tqdm import tqdm
import spacy 
from nltk.corpus import stopwords
import time

parser = argparse.ArgumentParser()
parser.add_argument('--emb_size', type=int, default=50,
              help="embedding size")
parser.add_argument('--emb_type', type=str, default="word",
                    choices=["word", "phrase"],
              help="embedding size")
args = parser.parse_args()

emb_size = args.emb_size
emb_type = args.emb_type

nlp = spacy.load('en_core_web_sm')

def clean_paper_text(text):
    
    cleaned_text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)
    cleaned_text = re.sub(r"[~:]"," ",cleaned_text)    
    parsed_text = cleaned_text.split("\n")
    cleaned_text = [sen.strip() for sen in parsed_text if len(sen.split()) > 3]
    return "\n".join(cleaned_text)

papers = pd.read_csv("nips_paper/papers.csv")
paper_authors = pd.read_csv("paper_authors.csv")

# Paper id Reindexing
paper_id_reindexing = {}
i = 0
for id in papers["id"]:
    assert id not in paper_id_reindexing
    paper_id_reindexing[id] = i
    i += 1
# trn/dev/tst split
trn_idx, val_idx, tst_idx = np.load("split.npy", allow_pickle=True)
# author id Reindexing
authod_id_reindexing = {}
i = 0
for id in paper_authors["author_id"]:
    if id in authod_id_reindexing :
        continue
    else:
        authod_id_reindexing[id] = i
        i += 1

# Prepare paper -> author mapping (all with reindexed ids)
paper_id_author_dict = {}

for paper_id, author_id in zip(paper_authors["paper_id"], paper_authors["author_id"]):
    if not paper_id in paper_id_reindexing:
        continue
    if paper_id_reindexing[paper_id] in paper_id_author_dict:
        paper_id_author_dict[paper_id_reindexing[paper_id]].append(authod_id_reindexing[author_id])
    else:
        paper_id_author_dict[paper_id_reindexing[paper_id]] = [authod_id_reindexing[author_id]]
print("total number of unique authors", len(set(paper_authors["author_id"])), max([max(a_list) for p_id, a_list in paper_id_author_dict.items()]))
print("total number of unique papers", len(set(papers["id"])), max(paper_id_author_dict.keys()))
print("average number of authors per paper", np.mean([len(a_list) for p_id, a_list in paper_id_author_dict.items()]))

papers['cleaned_paper_text'] = papers ['paper_text'].apply(lambda x:clean_paper_text(x))

papers_phrase = pd.read_csv("nips_paper/papers_phrased.csv")

from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

def reformat_phrase(text):
    if type(text) != str:
        text = str(text)
    text = text.replace("<phrase>", "<phrase> ").replace("</phrase>", " </phrase> ")
    
    reformatted_text = []
    for sent in text.split("\n"):
        #print(sent)
        stack = []
        reformatted_sent = []
        for c in sent.split()[:200]:
            if "<phrase>" in c:
                stack.append(c)
            elif len(stack) >0:
                stack.append(c)
                if stack[-1] == "</phrase>":
                    phrase = []
                    while stack:
                        phrase.append(stack.pop().lower())
                    reformatted_sent.append("_".join((phrase)[-2:0:-1]))
                else:
                    continue
            else:
                reformatted_sent.append(c.lower())
        reformatted_text.append(" ".join(reformatted_sent))
    
    ## original sentences are seperated. 
    ## Re-tokenize the text into sentences and split the words with puncts.
    tokenize_sents = [" ".join(s.split()) for s in sent_tokenize(" ".join(reformatted_text)) if len(s.split()) >2 ]
    
    return tokenize_sents



if emb_type == "word":
    all_sentences = []
    for id, text in zip(papers['id'], 
                               papers['cleaned_paper_text']):
        if paper_id_reindexing[id] in trn_idx:
            all_sentences += text.lower().split("\n")
elif emb_type == "phrase":
    papers_phrase['phrased_paper_text'] = papers_phrase['cleaned_paper_text'].apply(lambda x:reformat_phrase(x))

    def merge_sent(x):
        return "\n".join(x)

    ## merge the sentence list into one text and output one csv file
    papers_phrase['saved_phrase_paper_text'] = papers_phrase['phrased_paper_text'].apply(lambda x:merge_sent(x))
    headers = ["id", "saved_phrase_paper_text"]
    papers_phrase.columns = ['id', 'cleaned_paper_text', 'phrased_paper_text', 'saved_phrase_paper_text']

    print(len(papers_phrase['id']))
    print(len(papers_phrase['saved_phrase_paper_text']))
    saved_phrased = papers_phrase[headers].copy()

    # saved_phrased.to_csv('nips_paper/papers_cleaned_phrased.csv', index=False)
    all_sentences = []
    for id, phrase_text in zip(papers_phrase['id'], 
                               papers_phrase['phrased_paper_text']):
        if paper_id_reindexing[id] in trn_idx:
            all_sentences += phrase_text

               

class MySentences(object):
    def __init__(self, sentences):
         self.sents = sentences
    def __iter__(self):
        for line in self.sents:
            yield line.split()

st = time.time()
sentences = MySentences(all_sentences)
model = gensim.models.Word2Vec(sentences, size=emb_size, min_count=2, workers=20, iter=5)
model.save("data/nips_{}.trn.model.{}".format(emb_type, emb_size))
print('Finished in {:.2f}'.format(time.time()-st))
# trained_model = gensim.models.Word2Vec.load("nips_paper/nips_phrase.model")

X = np.zeros((len(papers_phrase), emb_size))

for id, text in zip(papers['id'], 
                           papers['cleaned_paper_text']):
    vectors = [model.wv[w] for w in " ".join(text.lower().split("\n")).split() if w in model.wv]
    if len(vectors) > 0:
        X[paper_id_reindexing[id], :] = np.mean(vectors, axis=0)
            
num_author = len(set(paper_authors["author_id"]))
Y = np.zeros((len(papers_phrase), num_author))
for p_id, a_list in paper_id_author_dict.items():
    for a_id in a_list:
        Y[p_id][a_id] = 1
        
X_train, Y_train = X[trn_idx], Y[trn_idx]
X_val, Y_val = X[val_idx], Y[val_idx]
X_tst, Y_tst = X[tst_idx], Y[tst_idx]
np.save("data/{}_{}d".format(emb_type, emb_size), 
        {"trn":[X_train, Y_train], "val":[X_val, Y_val], 
         "tst":[X_tst, Y_tst]})