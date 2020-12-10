import datasets
import time

from metric import compute_bertscore


# Load BERTscore metric
t1 = time.time()
compute_bertscore.metric = datasets.load_metric('bertscore')
dt = time.time() - t1
print ('Time taken to load bertscore metric', dt)

# Experiment with the loaded metric
refs = ['I am an intelligent boy', 'This person is kind', 'That movie was great watch.']
gens = ['I am a smart person', 'This woman is nice to others', 'what is it now that i told you']

for ref, gen in zip(refs, gens):
    t1 = time.time()
    final_score = compute_bertscore(gen, ref)
    print (final_score)
    dt = time.time() - t1
    print ('Time to compute bertscore', dt)
