import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Difficulty evaluation script arguments')
    parser.add_argument('--cross-rev-bucket', default=None, type=int,
                        help='cross review bucket id if using agent to get difficulty scores for summarization')
    args = parser.parse_args()

    target_bucket = args.cross_rev_bucket
    rouge_scores_df = pd.DataFrame()
    no_models_evaluated = 0
    rouge_score_cols = []
    for buc_id in range(5):
        if buc_id == target_bucket:
            continue

        rouge_scores_file_path = './rouge_score_files/rl_buc_dir_{}_evals_buc_{}.csv'.format(buc_id, target_bucket)
        raw_df = pd.read_csv(rouge_scores_file_path)

        if no_models_evaluated == 0:
            rouge_scores_df['filename'] = raw_df['filename']

        rl_buc_rouge_score_head = 'rouge_score_rl_buc_{}'.format(buc_id)
        rouge_scores_df[rl_buc_rouge_score_head] = raw_df['rouge_score']
        rouge_score_cols.append(rl_buc_rouge_score_head)

        no_models_evaluated += 1

    rouge_scores_df['average_rouge'] = rouge_scores_df[rouge_score_cols].mean(axis=1)
    rouge_scores_df['diff_score'] = 1 - rouge_scores_df['average_rouge']

    rouge_scores_df.to_csv('./rouge_score_files/difficulty_scores_for_buc_{}.csv'.format(target_bucket))



