from typing import Union, List, Tuple
from mini_object import Point
import pickle
import os

def save_points(points: List['Point'], path: str, file_name: str) -> None:
    save_path = os.path.join(path, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(points, f)

def load_points(file_path: str) -> List['Point']:
    with open(file_path, 'rb') as f:
        points = pickle.load(f)
    return points

def get_jump_index(index_list: List[int]) -> Tuple[List[int], int]:
    """ Get jump index list for non-continuous file index list.
    """
    M = max(index_list)
    jump_id = [i for i in range(M) if i not in index_list]
    return jump_id, M

def next_file_index(jump_id: List[int], M: int) -> int:
    """ Get next file index for continuous or non-continuous file index list.

    Input:
        jump_id: jump_id which come from "get_jump_index".
        M: M which come from "get_jump_index".

    """
    if len(jump_id) == 0:
        return M + 1
    else:
        return jump_id.pop()
    
