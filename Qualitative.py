import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize
import django

# global vars
group_kcs = ['group_1','group_2','group_3','group_4',
             'group_5','group_6','group_7']

# load lda model
lda = joblib.load('lda.pkl')

# clean docs (lowercase, remove punctuation)
def clean_doc(s):
    
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    return s

# clean and make docs
def make_doc(answer):
    
    text = []
    for ans in answer:
        text.append(ans['text'])
    clean = clean_doc(" ".join(text))
    words = word_tokenize(clean)
    
    # reduce instances of "teach-" to "teach"
    t = ['teaching','teacher','teachers']
    cleaner = [i if i not in t else "teach" for i in words]
    doc = " ".join(cleaner)
        
    return doc

# make tfidf matrix (sparse word frequency matrix)
def make_tfidf_matrix(doc):
    
    tfidf = TfidfVectorizer(max_features = 1000,stop_words = 'english')
    stops = list(tfidf.get_stop_words()) +['education','students','students','school','learning',
                                           'learn','experience','teach','working']
    tfidf.set_params(stop_words=stops)
    tfidf_matrix = tfidf.fit_transform(doc)
    return tfidf_matrix

# update or create group weights
def group_weights(user):
    
    answers = QualitativeAnswer.objects.filter(learner=user).values('text')
    doc = make_doc(answers)
    tfidf_matrix = make_tfidf_matrix(doc) 
    weights = lda.transform(tfidf_matrix)

    for ind, weight in enumerate(weights):
        knowledge_component = KnowledgeComponent.objects.get(kc_id=group_kcs[ind])
        Score.objects.update_or_create(knowledge_component = knowledge_component,
        learner = user, defaults = {'value': weight})

