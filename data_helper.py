# -*- coding:utf-8 -*-
import pandas as pd
from konlpy.tag import Kkma
from sklearn.feature_extraction.text import TfidfVectorizer


kkma = Kkma()


def load_data(path='./data/combine_cancer_counseling.xlsx'):
    frame = pd.read_excel(path)
    columns = frame.columns[1:3]
    frame = frame[columns]
    frame.columns = ['title', 'content']
    return frame


def filter_pos_with_kkma(sentence):
    pos_list = [
        'NNG',      # 일반 명사
        'NNP',      # 고유 명사
        'NNB',      # 의존 명사
        'VV',       # 명사
        'SL'        # 외국어

        # 'NP',     # 대명사
        # 'VA',     # 형용사
        # 'XR',     # 어근
    ]

    # load dat
    words = []
    for word, pos in kkma.pos(sentence):
        if pos in pos_list:
            words.append(word)

    sentence = ' '.join(words)
    return sentence


def pos_tagging(path='./data/combine_cancer_counseling.xlsx'):
    # combine_cancer_counseling.xlxs  =>  combine_cancer_counseling_base.txt
    #                                 =>  combine_cancer_counseling_pos.txt
    frame = load_data(path)

    base_fid = open('./data/combine_cancer_counseling_base.txt', 'w', encoding='utf-8')
    pos_fid = open('./data/combine_cancer_counseling_pos.txt', 'w', encoding='utf-8')

    for title, content in frame.values:
        try:
            base_line = title.strip() + ' <SPLIT> ' + ' '.join(content.split('\n')) + '\n'  # <SPLIT> is split word
            pos_line = filter_pos_with_kkma(title) + ' ' + filter_pos_with_kkma(content) + '\n'

            base_fid.write(base_line)
            pos_fid.write(pos_line)
        except TypeError:
            pass
        except AttributeError:
            pass


def tf_idf(path='./data/combine_cancer_counseling_pos.txt'):
    # combine_cancer_counseling_pos.txt => combine_cancer_counseling_tfidf.txt
    corpus = open(path, encoding='utf-8').readlines()
    vectorizer = TfidfVectorizer(
        min_df=5,
        max_df=0.95,
        max_features=8000,
    )
    x_data = vectorizer.fit_transform(corpus)
    return x_data


if __name__ == '__main__':
    pos_tagging()
    # tf_idf()
