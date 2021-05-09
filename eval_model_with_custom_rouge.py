""" Evaluate the baselines ont ROUGE/METEOR"""
import argparse
import json
import os
from os.path import join, exists
import pandas as pd

from metric import compute_rouge_n

try:
    _DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


def main(args):

    dec_dir = args.decoded_data_dir
    act_dir = args.actual_data_dir

    decoded_files = os.listdir(dec_dir)
    ref_files = os.listdir(act_dir)
    file_ids = [dec_file.split('.')[0] for dec_file in decoded_files]

    rouge_scores_list = []
    for ind, file_id in enumerate(file_ids):
        act_file_name = '.'.join([file_id, 'json'])
        with open(join(act_dir, act_file_name)) as f:
            js = json.loads(f.read())
            abstract = '\n'.join(js['abstract'])

        dec_file_name = decoded_files[ind]
        with open(join(dec_dir, dec_file_name)) as f:
            generation = f.read()

        rouge_score = compute_rouge_n(generation.split(' '), abstract.split(' '))
        rouge_scores_list.append(rouge_score)

    df = pd.DataFrame()
    df['file_id'] = file_ids
    df['rouge_score'] = rouge_scores_list
    df.to_csv(join('./rouge_score_files', args.rouge_scores_file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the output files for the RL full models')

    # choose metric to evaluate
    parser.add_argument('--actual-data-dir', default=None, required=True,
                        help='directory of actual article and summary pairs')
    parser.add_argument('--decoded-data-dir', default=None, required=True,
                        help='directory of decoded summaries')
    parser.add_argument('--rouge-scores-file-name', default=None, required=True,
                        help='File where rouge scores will be saved')

    args = parser.parse_args()
    main(args)
