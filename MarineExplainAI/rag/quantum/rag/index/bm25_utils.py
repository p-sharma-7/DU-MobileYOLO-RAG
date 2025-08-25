import pickle

def load_bm25_index(index_path):
    with open(index_path, "rb") as f:
        bm25 = pickle.load(f)
    return bm25
