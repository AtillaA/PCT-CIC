import h5py
import os
import os.path as osp
import gzip
import torch
import json
import torch.nn as nn
import numpy as np


wordmap_dir = 'gmap_floor1_mpp_word_0.05_channel_last_with_bounds'
embedding_dir = 'data/annt/embeddings.json.gz'

obj2wordidx = {
    "wall": 2392,
    "floor": 883,
    "chair": 424,
    "door": 701,
    "table": 2159,
    "picture": 1634,
    "cabinet": 375,
    "cushion": 606,
    "window": 2449,
    "sofa": 2020,
    "bed": 243,
    "curtain": 598,
    "chest_of_drawers": 728,
    "plant": 1667,
    "sink": 1972,
    "stairs": 2058,
    "ceiling": 414,
    "toilet": 2248,
    "stool": 2101,
    "towel": 2261,
    "mirror": 1390,
    "tv_monitor": 2306,
    "shower": 1951,
    "column": 501,
    "bathtub": 224,
    "counter": 553,
    "fireplace": 867,
    "lighting": 1270,
    "beam": 231,
    "railing": 1766,
    "shelving": 1941,
    "blinds": 290,
    "gym_equipment": 803,
    "seating": 1895,
    "board_panel": 1561,
    "furniture": 945,
    "appliances": 119,
    "clothes": 486,
    "objects": 1469,
    "misc": 1517,
}
room2wordidx = {'meetingroom/conferenceroom': 1370, 'office': 1474, 'hallway': 1036, 'kitchen': 1205, 'other room': 1517, 'classroom': 469, 'lounge': 1311, 'library': 1265, 'dining booth': 309, 'rec/game': 953, 'spa/sauna': 2029, 'stairs': 2058, 'bathroom': 222, 'dining room': 660, 'familyroom/lounge': 1311, 'living room': 1291, 'entryway/foyer/lobby': 1294, 'bedroom': 246, 'laundryroom/mudroom': 1235, 'closet': 482, 'toilet': 2248, 'porch/terrace/deck': 1706, 'balcony': 183, 'utilityroom/toolroom': 2344, 'junk': 956, 'workout/gym/exercise': 1027, 'tv': 2306, 'garage': 955, 'bar': 197, 'outdoor': 1524}
nav2wordidx = {
    "path": 1594,
    "current": 597, 
    "full": 942,
    "empty": 778
}

def get_maps(gmap_path):
    print(gmap_path)
    with h5py.File(gmap_path, "r") as f:
        nav_map  = f['nav_map'][()].astype(float)
        room_map = f['room_map'][()].astype(float)
        obj_maps = f['obj_maps'][()].astype(float)

        bounds = f['bounds'][()]
    print(np.amax(room_map), np.amin(room_map))
    print(np.amax(obj_maps), np.amin(obj_maps))
    return room_map, obj_maps

def load_embeddings():
    with gzip.open(embedding_dir, "rt") as f:
        embeddings = torch.tensor(json.load(f))

    embedding_layer = nn.Embedding.from_pretrained(
        embeddings=embeddings,
        freeze=True,
    )
    return embedding_layer

if __name__ == "__main__":
    embedding_layer =load_embeddings()
    file_list = os.listdir(f'data/maps/{wordmap_dir}')
    for file in file_list:
        scene_id = file.split('_')[0]
        room_map, obj_maps = get_maps(f'data/maps/{wordmap_dir}/{file}')
        room_glove_wordmap = embedding_layer(torch.from_numpy(room_map).long())
        print(room_glove_wordmap.shape)
        obj_glove_wordmap = embedding_layer(torch.from_numpy(obj_maps).long())
        print(obj_glove_wordmap.shape)
        print(f"Finish scene {scene_id}")