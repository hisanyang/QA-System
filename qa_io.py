import os
import re
import pandas as pd
from copy import deepcopy
import en_core_web_lg

nlp = en_core_web_lg.load()#spacy.load('en_core_web_lg')



def get_story_id_from_input(input_fpath):
    """function to get story_id from "input.txt"
    """
    with open(input_fpath) as f:
        lines = f.readlines()
    input_dir = lines[0].strip()
    story_id = [l.strip() for l in lines[1:]]
    return (input_dir, story_id)

def get_story_data(story_ids, input_dir):
    """function to read ".answers" and ".questions" data
    """
    stories = []
    for s in story_ids:
        with open('%s%s.story' % (input_dir[1:], s)) as f:
        #with open('developset/1999-W02-5.story') as f:
            story = f.read()
            m = re.search(r'HEADLINE:(.+)\n', story)
            if m: 
                headline = m.group(1).strip()
            else: 
                print('NO HEADLINE!')
            m = re.search(r'DATE:(.+)\n', story)

            if m: 
                date = m.group(1).strip()
            else:
                print('NO DATE!')
            m = re.search(r'TEXT:([\s\S]+)', story)

            if m:
                storytxt = nlp(m.group(1).strip())
            else:
                print('NO STORY CONTENT!')
            stories.append({'story_id': s, 'headline': headline, 'date': date, 'story': storytxt})                

    df = pd.DataFrame(stories).reindex(['story_id', 'headline', 'date', 'story'], axis = 1)
    # write to disk

    #df.to_pickle('story_data.pkl')
    return df

def get_question_and_ans_data(story_ids, input_dir, has_ans = True):
    """functions to read ".answers" and ".questions" data
        if has_ans = True, read ".answers" files; intead, read ".questions" files
    """
    questions = []
    for s in story_ids:
        if has_ans == False:
            p = '%s%s.questions' % (input_dir[1:], s)
        elif has_ans == True:
            p = '%s%s.answers' % (input_dir[1:], s)
        with open(p) as f:
            lines = f.read()
        for q in lines.split('\n\n'):
            if q.strip() != '':
                m = re.search(r'QuestionID: *(\S+)\n?', q)
                if m:
                    question_id = m.group(1).strip()
                else:
                    print('NO QUESTION ID!')
                m = re.search(r'Question:(.+)\n?', q)
                if m:
                    question = nlp(m.group(1).strip())
                else:
                    print('NO QUESTION!')
                m = re.search(r'Difficulty:(.+)\n?', q)
                if m:
                    difficulty = m.group(1).strip()
                else:
                    print('NO DIFFICULTY!')
                m = re.search(r'Answer:(.+)\n?', q)
                if m:
                    answer = nlp(m.group(1).strip())
                else:
                    print('NO ANSWER!')

                questions.append({'story_id': s, 'question_id': question_id, 'question': question, 'difficulty': difficulty, 'answer': answer})

    df = pd.DataFrame(questions).reindex(['story_id', 'question_id', 'question', 'difficulty', 'answer'], axis = 1)

    # write to disk
    #df.to_pickle('question_and_ans_data.pkl')
    return df

def score(question_and_ans_data):
    """functions to produce precision, recall and f_score based on queston_and_ans_data 
    """
    def _make_precision(row):
        precision = len(_overlap_tokens(row['answer'], row['answer_pred'])) / len(row['answer_pred'])
        return precision

    def _make_recall(row):
        recall = len(_overlap_tokens(row['answer'], row['answer_pred'])) / len(row['answer'])
        return recall
    def _make_overlap(row):
        overlap = _overlap_tokens(row['answer'], row['answer_pred'])
        return overlap
    def _overlap_tokens(doc, other_doc):
        """Get the tokens from the original Doc that are also in the comparison Doc.
        """
        overlap = []
        if type(other_doc)==float:
            return overlap
        other_tokens = [token.text for token in nlp(other_doc.strip())]
        for token in doc:
            if token.text in other_tokens:
                overlap.append(token)
        return overlap
        
    question_and_ans_data['precision'] = question_and_ans_data.apply(_make_precision, axis = 1)
    question_and_ans_data['recall'] = question_and_ans_data.apply(_make_recall, axis = 1)
    question_and_ans_data['overlap'] = question_and_ans_data.apply(_make_overlap, axis = 1)
    question_and_ans_data = question_and_ans_data.assign(f_score = lambda x: (2 * x['precision'] * x['recall']) / (x['precision'] + x['recall']))
    return question_and_ans_data



def create_input():
    """ create an input file (list file names of stories)

        required by pp. 3 of the instruction"""

    fpath = os.listdir(dir_path + 'developset')


    # list the file path of all answers and questions

    story_fpath = [f for f in fpath if os.path.splitext(f)[1] == '.story']

    story_fpath.sort()

    answer_fpath = [f for f in fpath if os.path.splitext(f)[1] == '.answers']

    answer_fpath.sort()

    question_fpath = [f for f in fpath if os.path.splitext(f)[1] == '.questions']

    question_fpath.sort()
    with open('developset/input.txt', 'w') as f:

        f.writelines('/developset/\n')

        for s in story_fpath:

            f.writelines(os.path.splitext(s)[0])

            f.writelines('\n')

def formatting(df):
    for story_id in set(df['story_id']):
        with open(os.getcwd()+'\\developset\\'+str(story_id)+'.response','w') as f:
            for num in range(df[df['story_id']==q].shape[0]):
                f.write('QuestionID: '+df[df['story_id']==q].loc[num,'question_id'])
                f.write('\n')
                f.write('Answer: '+df[df['story_id']==q].loc[num,'answer_pred'])
                f.write('\n')
                f.write('\n')
        f.close()

def overall_formatting(df):
    with open(os.getcwd()+'\\developset\\'+'All.response','w') as f:
        for i in range(df.shape[0]):
            f.write('QuestionID: '+df.loc[i,'question_id'].strip().replace('\r','').replace('\n',' '))
            f.write('\n')
            f.write('Answer: '+df.loc[i,'answer_pred'].strip().replace('\r','').replace('\n',' '))
            f.write('\n\n')
    f.close()

def grab_answers():
    with open(os.getcwd()+'\\developset\\'+'input.txt','r') as input:
        q=input.readlines()
    input.close()
    Total_list=[]
    for id in q[1:]:        
        with open(os.getcwd()+'\\developset\\'+id.replace('\n','')+'.answers','r') as f:
            Total_list.extend(f.readlines())
    f.close()
    with open(os.getcwd()+'\\developset\\'+'All.answers','w') as writer:
        for data in Total_list:
            writer.write(data)
    writer.close()
    