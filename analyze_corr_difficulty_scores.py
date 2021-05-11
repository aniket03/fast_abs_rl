import os
import json
import numpy as np
import pandas as pd

from matplotlib import  pyplot as plt


def plot_article_len_vs_diff_score(article_len_list, diff_scores):

    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    plt.scatter(article_len_list, diff_scores, marker='.')

    plt.title('Variation in difficulty scores computed using cross review with length of article')
    plt.ylabel('Difficulty score for the article')
    plt.xlabel('Count of words in the article')

    plt.legend(loc='best')
    plt.savefig('./plots/scatter_plt_diff_vs_len.png')
    plt.close()


if __name__ == '__main__':

    target_bucket_id = 2
    rouge_scores_df = pd.read_csv('./rouge_score_files/'
                                  'difficulty_scores_for_buc_{}.csv'.format(target_bucket_id))

    filenames = list(rouge_scores_df['filename'])[:1200]
    diff_scores = list(rouge_scores_df['diff_score'])[:1200]

    DATA_DIR = os.environ['DATA']
    train_data_dir = os.path.join(DATA_DIR, 'train')

    article_len_list = []
    for ind, filename in enumerate(filenames):
        with open(os.path.join(train_data_dir, filename)) as f:
            js = json.loads(f.read())
        art_sents = js['article']
        article_text = '\n'.join(art_sents)

        article_len_list.append(len(article_text.split()))

    corr_coef = np.corrcoef(article_len_list, diff_scores)
    plot_article_len_vs_diff_score(article_len_list, diff_scores)
    print (corr_coef)


