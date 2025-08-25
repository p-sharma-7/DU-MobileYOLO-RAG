import pickle

def load_bm25_index(index_path):
    with open(index_path, "rb") as f:
        bm25 = pickle.load(f)
    return bm25

def load_id_map(id_map_path):
    import json
    with open(id_map_path, "r") as f:
        id_map = json.load(f)
    return id_map
