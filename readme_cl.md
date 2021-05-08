# Readme for CL experiments

Procedure for training a summarization agent on a particular cross-review bucket elements.

1. pip install requirements. `python version used 3.6.13` Note pyrouge in requirements is not required right now.
2. Download the dataset from google drive link [here](https://drive.google.com/drive/folders/1E1AF1xGxEAAKd2F-1sUPWwH9OYkHFQGL?usp=sharing).
Once the dataset is downloaded, it will need to be decompressed.
3. Set the os environment variable `DATA` to the path of the directory, where the final decompressed folders for `train`, `val` and `test` are maintained.
4. Also files `cross_rev_bucket_*.csv` will need to be downloaded in the same `DATA` directory. Each `cross_rev_bucket_<i>.csv` file indicates to the list of files that should be referred by the model during training of the summarization agent on ith bucket data.
5. Create a directory `models` in the current project directory and download and unzip the folder `word2vec_dir.zip` in the same (This is also present in the google drive link).
6. Run the script `tr_all_three.sh`, it will first train the extractor, then the abstractor and finally fine-tune the extractor pipeline using RL. The current shell script will do this for bucket index 3, but we will need to change this to `<i>` in order to train the model on ith bucket article summary pairs.