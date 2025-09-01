# utility functions:
def flip_rel(rel_name: str) -> str:
    return rel_name[4:] if rel_name.startswith("rev_") else f"rev_{rel_name}"

def to_canonical(mp_outward):
    # mp_outward: [(src, rel, dst), ...] dalla costruzione RL (parte da 'drivers')
    mp = [(dst, flip_rel(rel), src) for (src, rel, dst) in mp_outward[::-1]]
    assert mp[-1][2] == "drivers"
    return tuple(mp)

metapaths = [[('drivers', 'rev_f2p_driverId', 'standings'), ('standings', 'f2p_raceId', 'races')]]


canonical = []
for mp in metapaths:
    #change to canonical:
    mp = mp.copy()
    mp_key   = to_canonical(mp)         
    canonical.append(mp_key)
print(f"I metapath canonici sono: {canonical}")
