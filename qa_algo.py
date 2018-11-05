import os
import re
import numpy as np
import spacy
import pandas as pd
from copy import deepcopy
#from qa_io import TextBlob
from qa_io import nlp
import en_core_web_lg

nlp = en_core_web_lg.load()

#qa.story_data, qa.question_and_ans_data, qa.story_ids

def extract_answer(story_data, question_and_ans_data, story_ids):
    """ (1) get answer, then modify self.question_and_ans_data by add the answer to it. 
        (2) for each story id, extract its question, then look up in story_data, find the best sentence"""

    for story_id in story_ids:
        story = story_data.loc[lambda df: df.story_id == story_id, 'story'].values[0]
        question_ids = question_and_ans_data.loc[lambda df: df.story_id == story_id, 'question_id']

        for question_id in question_ids:
            # get the question and answer
            question = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'question'].values[0]
            if 'answer' in question_and_ans_data:
                answer = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer'].values[0]

            ans = []
            for sent in story.sentences:
                sim = sent.similarity(question)
                ans.append({'question_id': question_id, 'answer_pred': sent, 'similarity': sim})

            ans = pd.DataFrame(ans).reindex(['question_id', 'answer_pred', 'similarity'], axis = 1)
            ans.sort_values(by = ['similarity'], ascending = False, inplace = True)

            question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer_pred'] = ans.iloc[0]['answer_pred'].text

    question_and_ans_data['answer_pred'] = question_and_ans_data['answer_pred'].apply(nlp)

    return question_and_ans_data


def extract_answer_JACCARD(story_data, question_and_ans_data, story_ids):
    """ (1) get answer, then modify self.question_and_ans_data by add the answer to it. 
        (2) for each story id, extract its question, then look up in story_data, find the best sentence"""

    for story_id in story_ids:
        story = story_data.loc[lambda df: df.story_id == story_id, 'story'].values[0]
        question_ids = question_and_ans_data.loc[lambda df: df.story_id == story_id, 'question_id']

        for question_id in question_ids:
            # get the question and answer
            question = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'question'].values[0]
            if 'answer' in question_and_ans_data:
                answer = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer'].values[0]

            ans = []
            for sent in story.sents:
                #sim = sent.similarity(question)
                sim=sum([min(question.vector[i], sent.vector[i]) for i in range(len(sent.vector))])/\
                sum([max(question.vector[i], sent.vector[i]) for i in range(len(sent.vector))])
                ans.append({'question_id': question_id, 'answer_pred': sent, 'similarity': sim})

            ans = pd.DataFrame(ans).reindex(['question_id', 'answer_pred', 'similarity'], axis = 1)
            ans.sort_values(by = ['similarity'], ascending = False, inplace = True)

            question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer_pred'] = ans.iloc[0]['answer_pred'].text

    question_and_ans_data['answer_pred'] = question_and_ans_data['answer_pred'].apply(nlp)

    return question_and_ans_data

def extract_answer_MANHATTAN(story_data, question_and_ans_data, story_ids):
    """ (1) get answer, then modify self.question_and_ans_data by add the answer to it. 
        (2) for each story id, extract its question, then look up in story_data, find the best sentence"""

    for story_id in story_ids:
        story = story_data.loc[lambda df: df.story_id == story_id, 'story'].values[0]
        question_ids = question_and_ans_data.loc[lambda df: df.story_id == story_id, 'question_id']

        for question_id in question_ids:
            # get the question and answer
            question = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'question'].values[0]
            if 'answer' in question_and_ans_data:
                answer = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer'].values[0]

            ans = []
            for sent in story.sents:
                #sim = sent.similarity(question)
                sim=sum([abs(question.vector[i]-sent.vector[i]) for i in range(len(sent.vector))])
                ans.append({'question_id': question_id, 'answer_pred': sent, 'similarity': sim})

            ans = pd.DataFrame(ans).reindex(['question_id', 'answer_pred', 'similarity'], axis = 1)
            ans.sort_values(by = ['similarity'], ascending = False, inplace = True)

            question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer_pred'] = ans.iloc[0]['answer_pred'].text

    question_and_ans_data['answer_pred'] = question_and_ans_data['answer_pred'].apply(nlp)

    return question_and_ans_data




def build_vocabulary(story_data):
    sentences=[]
    for i in range(story_data.shape[0]):
        for sent in story_data.loc[i,'story'].sentences:
            sentences.append(sent)
    return sentences

def cosine(u,v):
    return np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))

def extract_answer_IFST(story_data, question_and_ans_data, story_ids, model_version, Vocab_Size):
    """ (1) get answer, then modify self.question_and_ans_data by add the answer to it. 
        (2) for each story id, extract its question, then look up in story_data, find the best sentence"""
    import re
    import pandas as pd
    
    import torch
    import numpy as np
    from models import InferSent
    
    #sentence_list=build_vocabulary(story_data)
    W2V_PATH='dataset/GloVe/glove.840B.300d.txt' if model_version==1 else 'dataset/fastText/crawl-300d-2M.vec'
    MODEL_PATH='encoder/infersent%s.pkl' %model_version
    params_model={'bsize':64, 'word_emb_dim':300, 'enc_lstm_dim':2048,'pool_type':'max','dpout_model':0.0,'version':model_version}
    model=InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.set_w2v_path(W2V_PATH)
    if model_version==3:
        sentence_list=build_vocabulary(story_data)
        model.build_vocab(sentence_list)
    else:
        model.build_vocab_k_words(K=Vocab_Size)
    
    
    for story_id in story_ids:
        story = story_data.loc[lambda df: df.story_id == story_id, 'story'].values[0]
        question_ids = question_and_ans_data.loc[lambda df: df.story_id == story_id, 'question_id']

        for question_id in question_ids:
            # get the question and answer
            question = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'question'].values[0]
            if 'answer' in question_and_ans_data:
                answer = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer'].values[0]
            
            question_encoded=model.encode(str(question_and_ans_data.loc[question_and_ans_data.index[question_and_ans_data['question_id']==question_id][0],'question']))[0]

            ans = []
            for sent in story.sents:
                #sim = sent.similarity(question)
                sim=cosine(question_encoded, model.encode(str(sent))[0])
                
                ans.append({'question_id': question_id, 'answer_pred': sent, 'similarity': sim})

            ans = pd.DataFrame(ans).reindex(['question_id', 'answer_pred', 'similarity'], axis = 1)
            ans.sort_values(by = ['similarity'], ascending = False, inplace = True)

            question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer_pred'] = str(ans.iloc[0]['answer_pred']).replace('\n',' ')#.text

    #question_and_ans_data['answer_pred'] = question_and_ans_data['answer_pred'].apply(TextBlob)

    return question_and_ans_data