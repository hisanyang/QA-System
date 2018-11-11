import os
import sys
import re
import spacy
import pandas as pd
import qa_algo
import qa_io
from copy import deepcopy



##set working directory
##please change dir_path to where your solution located."""
#dir_path = 'C:/Users/rossz/OneDrive/Academy/the
#U/Assignment/AssignmentSln/NLP-04-QA/QA-System/QA-System/'
#os.chdir(dir_path)

class QA:
    def __init__(self, similarity_method, model_version, Vocab_Size, input_fpath):
        #input_fpath = 'developset/input.txt'
        #qa_io.create_input()
        self.input_dir, self.story_ids = qa_io.get_story_id_from_input(input_fpath)

        # create question and story dataset.  Read from disk if they exist else
        # create from scratch.
        self.story_data = qa_io.get_story_data(self.story_ids, self.input_dir)
        self.question_and_ans_data = qa_io.get_question_and_ans_data(self.story_ids, self.input_dir)
        self.similarity_method = similarity_method
        self.model_version = model_version
        self.Vocab_Size = Vocab_Size
            
    # produce answer
    def _extract_answer(self):
        if self.similarity_method == 'spacy':
            self.question_and_ans_data = qa_algo.extract_answer(self.story_data, self.question_and_ans_data, self.story_ids, only_np = False)
        elif self.similarity_method == 'spacy_qtype':
            self.question_and_ans_data = qa_algo.extract_answer_qtype(self.story_data, self.question_and_ans_data, self.story_ids)
        elif self.similarity_method == 'Jaccard':
            self.question_and_ans_data = qa_algo.extract_answer_JACCARD(self.story_data, self.question_and_ans_data, self.story_ids)
        elif self.similarity_method == 'Manhattan':
            self.question_and_ans_data = qa_algo.extract_answer_MANHATTAN(self.story_data, self.question_and_ans_data, self.story_ids)
        elif self.similarity_method == 'IFST': # You may ignore this: just used for other dataset and model.
            self.question_and_ans_data = qa_algo.extract_answer_IFST(self.story_data, self.question_and_ans_data, self.story_ids, self.model_version, self.Vocab_Size)
        elif self.similarity_method == 'Lemma':
            self.question_and_ans_data = qa_algo.extract_answer_lemmatize_stopwords(self.story_data, self.question_and_ans_data, self.story_ids, self.model_version, self.Vocab_Size)
        elif self.similarity_method == 'Regr':
            self.question_and_ans_data = qa_algo.extract_answer_Regression(self.story_data, self.question_and_ans_data, self.story_ids)

    # score output
    def _score(self):
        self.question_and_ans_data = qa_io.score(self.question_and_ans_data)
        

model_version = 3 # You may ignore this: just used for other dataset and model.  Database
                  # 1:GloVe 2:fastText 3:Use the training set
Vocab_Size = 100000 # You may ignore this: just used for other dataset and model.
similarity_method = 'Regr' #'spacy_qtype' # Put 'Lemma' if you want to run a lemmatized version with no stopwords

if __name__ == '__main__':
    args = sys.argv[1:]
    input_fpath = args[0]
    
    qa = QA(similarity_method, model_version, Vocab_Size, input_fpath)

    # Remove Stopwords
    #qa._process_StopWords()
    # Remove Wh- words + How
    qa._extract_answer()
    #qa._score() # As I rely on scoring program, do not need this function
    ans = qa.question_and_ans_data

    # For scoring program
    #qa_io.formatting(ans) # Question-id.response
    #qa_io.overall_formatting(ans)  # all responses for questions in one file
    #qa_io.overall_formatting_Output(ans)  # all responses for questions in one file
    #qa_io.grab_answers() # grab all answers from the *.answers file and put it in a single file

