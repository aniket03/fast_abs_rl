import os
import json
import pickle

import numpy as np
import pandas as pd

from matplotlib import  pyplot as plt


def plot_article_len_vs_diff_score(article_len_list, diff_scores):

    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    plt.scatter(article_len_list, diff_scores, marker='.')

    plt.title('Variation in difficulty scores computed using cross review with '
              'proportion of 5k most frequent words in article')
    plt.ylabel('Difficulty score for the article')
    plt.xlabel('Proportion of 5k most frequent words')

    plt.legend(loc='best')
    plt.savefig('./plots/scatter_plt_diff_vs_5k_mf.png')
    plt.close()


if __name__ == '__main__':

    # Set up target-bucket-id, vocab-file and rouge-scores-df
    target_bucket_id = 2
    vocab_file = 'word_cnt.pkl'
    rouge_scores_df = pd.read_csv('./rouge_score_files/'
                                  'difficulty_scores_for_buc_{}.csv'.format(target_bucket_id))

    # Get the article filenames and their difficulty scores
    filenames = list(rouge_scores_df['filename'])[:1200]
    diff_scores = list(rouge_scores_df['diff_score'])[:1200]

    # Get the vocab counts dictionary and obtain most common 5k words
    with open('vocab_cnt.pkl', 'rb') as f:
        wc = pickle.load(f)
    word_list = []
    for i, (w, _) in enumerate(wc.most_common(5000), 4):
        word_list.append(w)
    words_set_5k = set(word_list)

    # Set up the train_data_dir variable
    DATA_DIR = os.environ['DATA']
    train_data_dir = os.path.join(DATA_DIR, 'train')

    # Extract all article paragraphs and get proportion of words in them which belong to most frequent 5k words
    prop_mf_5k_list = []
    for ind, filename in enumerate(filenames):
        with open(os.path.join(train_data_dir, filename)) as f:
            js = json.loads(f.read())
        art_sents = js['article']
        article_text = '\n'.join(art_sents)
        cnt_mf_5k = words_set_5k.intersection(article_text.split())
        prop_mf_5k = len(cnt_mf_5k) / len(set(article_text.split()))
        prop_mf_5k_list.append(prop_mf_5k)


    # article_len_q25, article_len_q75 = np.percentile(article_len_list, [25, 75])
    # iqr = article_len_q75 - article_len_q25
    # lower_bound_len = article_len_q25 - 1.5 * iqr
    # upper_bound_len = article_len_q75 + 1.5 * iqr
    #
    # final_article_len_list, final_diff_scores_list = [], []
    # for article_len, diff_score in zip(article_len_list, diff_scores):
    #     if article_len < lower_bound_len or article_len > upper_bound_len:
    #         continue
    #     else:
    #         final_article_len_list.append(article_len)
    #         final_diff_scores_list.append(diff_score)

    corr_coef = np.corrcoef(prop_mf_5k_list, diff_scores)
    plot_article_len_vs_diff_score(prop_mf_5k_list, diff_scores)
    print (corr_coef)


