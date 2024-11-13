import json

DATA_ROOT = "data/arc-prize-2024/"

train_input_path = f'{DATA_ROOT}/arc-agi_training_challenges.json'
train_output_path = f'{DATA_ROOT}/arc-agi_training_solutions.json'

eval_input_path = f'{DATA_ROOT}/arc-agi_evaluation_challenges.json'
eval_output_path = f'{DATA_ROOT}/arc-agi_evaluation_solutions.json'

test_path = f'{DATA_ROOT}/arc-agi_test_challenges.json'
sample_path = f'{DATA_ROOT}/sample_submission.json'


with open("outputs/solutions_val.json", "r") as f:
    results = json.load(f)
    
with open(eval_output_path, "r") as f:
    targets = json.load(f)
    
c = n = 0
for k in targets:
    res = results[k]
    gt = targets[k]

    if json.dumps(gt[0]) == res:
        c += 1
        print(k)
    n += 1
    
print(c, n)
