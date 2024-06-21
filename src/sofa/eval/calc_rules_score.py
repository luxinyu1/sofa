import pandas as pd
from argparse import ArgumentParser
import glob
import os

parser = ArgumentParser()
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--final-score-file", type=str, default="final_score.csv")
args = parser.parse_args()

p_manual = 0
p_systematic = 0

manual_all = 0
systematic_all = 0

manual_matching_files = glob.glob(os.path.join(args.output_dir, 'score_manual*.csv'))
systematic_matching_files = glob.glob(os.path.join(args.output_dir, 'score_systematic*.csv'))

for file in manual_matching_files:
    res = pd.read_csv(file)
    passed, all_ins = res.iloc[-2:, 0].tolist()
    p_manual += int(passed)
    manual_all += int(all_ins)

for file in systematic_matching_files:
    res = pd.read_csv(file)
    passed, all_ins = res.iloc[-2:, 0].tolist()
    p_systematic += int(passed)
    systematic_all += int(all_ins)

assert manual_all == 870
assert systematic_all == 862

p_manual = round(p_manual / manual_all, 3)
p_systematic = round(p_systematic / systematic_all, 3)

print(f"P_manual:{p_manual}", f"P_systematic:{p_systematic}")