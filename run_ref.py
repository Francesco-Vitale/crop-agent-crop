import json, warnings; warnings.filterwarnings("ignore")
from cropagent.reference import best_plan; from cropagent.env import *
out = {n: best_plan(s) for n, s in (("train", TRAIN), ("shift_eval", SHIFT_EVAL))}
json.dump(out, open("results/reference.json","w")); print(out)
