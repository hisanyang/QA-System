import os
import re
import spacy
import pandas as pd
import qa_io
import qa_algo
from copy import deepcopy
import en_core_web_lg

nlp = en_core_web_lg.load()

##set working directory
##please change dir_path to where your solution located."""
#dir_path = 'C:/Users/rossz/OneDrive/Academy/the U/Assignment/AssignmentSln/NLP-04-QA/QA-System/QA-System/'
#os.chdir(dir_path)

#create_input()

class QA:
    def __init__(self, Sim_Method, model_version, Vocab_Size):
        input_fpath ='developset/input.txt'
        self.input_dir, self.story_ids = qa_io.get_story_id_from_input(input_fpath)

        # create question and story dataset. Read from disk if they exist else create from scratch.
        self.story_data = (qa_io.get_story_data(self.story_ids, self.input_dir) if not os.path.exists('story_data.pkl') else pd.read_pickle('story_data.pkl'))
        self.question_and_ans_data = (qa_io.get_question_and_ans_data(self.story_ids, self.input_dir) if not os.path.exists('question_and_ans_data.pkl') else pd.read_pickle('question_and_ans_data.pkl'))
        self.similarity_method=Sim_Method
        self.model_version=model_version
        self.Vocab_Size=Vocab_Size
            
    # produce answer
    def _extract_answer(self):
        if self.similarity_method=='Spacy':
            self.question_and_ans_data = qa_algo.extract_answer(self.story_data, self.question_and_ans_data, self.story_ids)
        elif self.similarity_method=='Jaccard':
            self.question_and_ans_data = qa_algo.extract_answer_JACCARD(self.story_data, self.question_and_ans_data, self.story_ids)
        elif self.similarity_method=='Manhattan':
            self.question_and_ans_data= qa_algo.extract_answer_MANHATTAN(self.story_data, self.question_and_ans_data, self.story_ids)
        elif self.similarity_method=='IFST':
            self.question_and_ans_data=qa_algo.extract_answer_IFST(self.story_data, self.question_and_ans_data, self.story_ids, self.model_version, self.Vocab_Size)

    # score output
    def _score(self):
        self.question_and_ans_data = qa_io.score(self.question_and_ans_data)

model_version= 1# Database 1:GloVe 2:fastText 3:Use the training set
Sim_Method='IFST'
Vocab_Size=100000
qa = QA(Sim_Method, model_version, Vocab_Size)
qa._extract_answer()
qa._score()
ans = qa.question_and_ans_data

# For scoring program
qa_algo.formatting(ans)
qa_algo.overall_formatting(ans)
qa_algo.grab_answers()

ans.to_csv('testans_'+Sim_Method+'.csv', index = False)