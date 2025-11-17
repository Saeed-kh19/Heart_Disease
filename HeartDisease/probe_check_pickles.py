import pickle, glob
paths = glob.glob("data/processed/**/*.pkl", recursive=True) + glob.glob("reports/results/*.pkl")
for p in paths:
    try:
        with open(p,"rb") as f:
            obj = pickle.load(f)
        print(p, "->", type(obj))
    except Exception as e:
        print(p, "LOAD-ERROR:", type(e).__name__, str(e))
