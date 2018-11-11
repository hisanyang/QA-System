import os
import re
import numpy as np
import spacy
import pandas as pd
from copy import deepcopy
#from qa_io import TextBlob
import qa_io
from qa_io import nlp
from sklearn import linear_model
clf = linear_model.LinearRegression()

nlp = spacy.load('en_core_web_lg')


# Classic Spacy
def extract_answer(story_data, question_and_ans_data, story_ids, only_np = False):
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
                sim=sent.similarity(question)

                if only_np == True:
                    sent = nlp(' '.join([n.text for n in sent.noun_chunks]))

                ans.append({'question_id': question_id, 'answer_pred': sent, 'similarity': sim})

            ans = pd.DataFrame(ans).reindex(['question_id', 'answer_pred', 'similarity'], axis = 1)
            ans.sort_values(by = ['similarity'], ascending = False, inplace = True)

            question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer_pred'] = ans.iloc[0]['answer_pred'].text

    question_and_ans_data['answer_pred'] = question_and_ans_data['answer_pred'].apply(nlp)

    return question_and_ans_data


# Classic Spacy + Regression
def extract_answer_Regression(story_data, question_and_ans_data, story_ids, only_np = False):
    """ (1) get answer, then modify self.question_and_ans_data by add the answer to it. 
        (2) for each story id, extract its question, then look up in story_data, find the best sentence"""
    Question_Answer_Vectors=pd.DataFrame(columns=['q','a'])
    count=0
    for story_id in story_ids:
        story = story_data.loc[lambda df: df.story_id == story_id, 'story'].values[0]
        question_ids = question_and_ans_data.loc[lambda df: df.story_id == story_id, 'question_id']
        

        for question_id in question_ids:
            # get the question and answer
            question = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'question'].values[0]
            if 'answer' in question_and_ans_data:
                answer = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer'].values[0]

            #Regress            
            # I just considered that | can exist only once, but found out that actually there can be many... So it needs to be fixed
            if '|' in str(answer):
                ans_1=nlp(str(answer)[:str(answer).index('|')].strip())
                ans_2=nlp(str(answer)[str(answer).index('|')+1:].strip())
                #Question_Answer_Vectors.loc[count]=(question.vector.mean(),ans_1.vector.mean()) 
                #Question_Answer_Vectors.loc[count]=(question.vector.mean(),ans_2.vector.mean())
                Question_Answer_Vectors.loc[count]=(question.vector,ans_1.vector) 
                Question_Answer_Vectors.loc[count]=(question.vector,ans_2.vector)
            else:
                #Question_Answer_Vectors.loc[count]=(question.vector.mean(),answer.vector.mean())
                Question_Answer_Vectors.loc[count]=(question.vector,answer.vector)
            count+=1
    # Get the value for each dimension of word vectors
    A=[]
    Q=[]
    for i in range(Question_Answer_Vectors.shape[0]):
        temp_A=[]
        temp_Q=[]
        for j in range(300):
            temp_A.append(Question_Answer_Vectors.loc[i]['a'][j])
            temp_Q.append(Question_Answer_Vectors.loc[i]['q'][j])
        A.append(temp_A)
        Q.append(temp_Q)
    
    # Linear regression performed
    clf.fit(Q, A)
    
    #clf.fit(np.array(Question_Answer_Vectors['q']).reshape(-1,1),np.array(Question_Answer_Vectors['a']))
    
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
                sim=cosine(sent.vector,clf.predict(question.vector)[0]) #.reshape(-1,1)))
                
                #sent.similarity(clf.predict(question.vector))
                #sim=sent.similarity(question)
                

                #if only_np == True:
                #    sent = nlp(' '.join([n.text for n in sent.noun_chunks]))

                ans.append({'question_id': question_id, 'answer_pred': sent, 'similarity': sim})

            ans = pd.DataFrame(ans, columns=['question_id', 'answer_pred', 'similarity'])
            ans.sort_values(by = ['similarity'], ascending = False, inplace = True)
            
            question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer_pred'] = ans.iloc[0]['answer_pred'].text

    question_and_ans_data['answer_pred'] = question_and_ans_data['answer_pred'].apply(nlp)

    return question_and_ans_data

# Spacy + question type
def extract_answer_qtype(story_data, question_and_ans_data, story_ids, only_np = False):
    """ (1) get answer, then modify self.question_and_ans_data by add the answer to it. 
        (2) for each story id, extract its question, then look up in story_data, find the best sentence"""

    for story_id in story_ids:
        story = story_data.loc[lambda df: df.story_id == story_id, 'story'].values[0]
        question_ids = question_and_ans_data.loc[lambda df: df.story_id == story_id, 'question_id']
        

        for question_id in question_ids:
            # get the question
            question = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'question'].values[0]
            questiontxt = [t.text.lower() for t in question]
            # get answer
            if 'answer' in question_and_ans_data:
                answer = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer'].values[0]

            # classify questions
            question_type = ''
            if 'when' in questiontxt[0:1]:
                question_type = 'datetime'
            if 'who' in questiontxt[0:1]:
                question_type = 'person'
            if 'where' in questiontxt[0:1]:
                question_type = 'loc'


            ans = []
            
            for sent in story.sents:         
                sim=sent.similarity(question)

                if only_np == True:
                    sent = nlp(' '.join([n.text for n in sent.noun_chunks]))
                if question_type == 'datetime':
                    sent = nlp(' '.join([ent.text for ent in sent.ents if ent.label_ in ['DATE', 'TIME']]))
                elif question_type == 'person':
                    sent = nlp(' '.join([ent.text for ent in sent.ents if ent.label_ in ['PERSON', 'NORP', 'ORG']]))
                elif question_type == 'loc':
                    sent = nlp(' '.join([ent.text for ent in sent.ents if ent.label_ in ['LOC', 'GPE', 'FAC']]))


                ans.append({'question_id': question_id, 'answer_pred': sent, 'similarity': sim})

            ans = pd.DataFrame(ans).reindex(['question_id', 'answer_pred', 'similarity'], axis = 1)
            ans.sort_values(by = ['similarity'], ascending = False, inplace = True)

            question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer_pred'] = ans.iloc[0]['answer_pred'].text

    question_and_ans_data['answer_pred'] = question_and_ans_data['answer_pred'].apply(nlp)

    return question_and_ans_data


# JACCARD
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

# Lemmatized and stopwords are ignored
def extract_answer_lemmatize_stopwords(story_data, question_and_ans_data, story_ids):
    """ (1) get answer, then modify self.question_and_ans_data by add the answer to it. 
        (2) for each story id, extract its question, then look up in story_data, find the best sentence"""

    for story_id in story_ids:
        story = story_data.loc[lambda df: df.story_id == story_id, 'story'].values[0]
        question_ids = question_and_ans_data.loc[lambda df: df.story_id == story_id, 'question_id']

        for question_id in question_ids:
            # get the question and answer
            question = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'question'].values[0]
            # I lemmatized the words in questions and considered only those words that are not stopwords
            question_lemmatized= nlp(' '.join([token.lemma_.lower() for token in question if token.lemma_.lower() not in nlp.Defaults.stop_words]))
            if 'answer' in question_and_ans_data:
                answer = question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer'].values[0]

            ans = []
            
            for sent in story.sents:
                # I lemmatized the words in sentences and considered only those words that are not stopwords
                sent_lemmatized=nlp(' '.join([token.lemma_.lower() for token in sent if token.lemma_.lower() not in nlp.Defaults.stop_words]))
                # Lemmatized sentence and question are compared against each other
                sim = sent_lemmatized.similarity(question_lemmatized) 
                #sim=sent.similarity(question)
                ans.append({'question_id': question_id, 'answer_pred': sent, 'similarity': sim})

            ans = pd.DataFrame(ans).reindex(['question_id', 'answer_pred', 'similarity'], axis = 1)
            ans.sort_values(by = ['similarity'], ascending = False, inplace = True)

            question_and_ans_data.loc[lambda df: df.question_id == question_id, 'answer_pred'] = ans.iloc[0]['answer_pred'].text

    question_and_ans_data['answer_pred'] = question_and_ans_data['answer_pred'].apply(nlp)

    return question_and_ans_data


"""
You may ignore the code below this line, as code below is used for InferSent model and GloVe/fastText dataset
"""
def build_vocabulary(story_data):
    T=[]
    for i in range(story_data.shape[0]):
        for sent in story_data.loc[i,'story'].sentences:
            T.append(sent)
    return T

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
    