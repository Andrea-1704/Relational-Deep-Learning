#main:
def flip_rel(rel_name: str) -> str:
    """
    Converts the name of the relation to its reversed version.
    """
    return rel_name[4:] if rel_name.startswith("rev_") else f"rev_{rel_name}"

def to_canonical(mp_outward):
    """
    This function takes as input the metapath found by the model and converts
    it to a metapath with reversed paths, meaning that the source of each 
    path becomes destination, the destination the source and the name of the 
    relation gets converted (flip_rel).
    
    It should be verified that mp[-1][2] == target node. 
    """
    mp = [(dst, flip_rel(rel), src) for (src, rel, dst) in mp_outward[::-1]]
    # assert mp[-1][2] == "drivers"
    print(f"changed metapath order from {mp_outward} to {mp}")
    return tuple(mp)
if __name__ == '__main__':
    mp_outward = [("drivers", "drives", "cars"), ("cars", "drives", "passengers")]
    print(to_canonical(mp_outward))
