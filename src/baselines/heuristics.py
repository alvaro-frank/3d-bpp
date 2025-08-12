from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class HBox:
    w: int; h: int; d: int; id: int
    rot: int = 0
    pos: Optional[Tuple[int,int,int]] = None

    def rotated(self, rot: int) -> Tuple[int,int,int]:
        w,h,d = self.w, self.h, self.d
        if rot == 0: return (w,h,d)
        if rot == 1: return (w,d,h)
        if rot == 2: return (h,w,d)
        if rot == 3: return (h,d,w)
        if rot == 4: return (d,w,h)
        if rot == 5: return (d,h,w)
        raise ValueError("rot in [0..5]")

class Grid3D:
    def __init__(self, W:int,H:int,D:int):
        self.W,self.H,self.D=W,H,D
        self.placed: List[HBox] = []

    def fits(self, dims: Tuple[int,int,int], pos: Tuple[int,int,int]) -> bool:
        x,y,z = pos; w,h,d = dims
        return x>=0 and y>=0 and z>=0 and (x+w)<=self.W and (y+h)<=self.H and (z+d)<=self.D

    def overlaps(self, dims: Tuple[int,int,int], pos: Tuple[int,int,int]) -> bool:
        x,y,z = pos; w,h,d = dims
        for b in self.placed:
            bx,by,bz = b.pos
            bw,bh,bd = b.rotated(b.rot)
            sep = (x+w<=bx) or (bx+bw<=x) or (y+h<=by) or (by+bh<=y) or (z+d<=bz) or (bz+bd<=z)
            if not sep:
                return True
        return False

    def top_z_at(self, dims_xy: Tuple[int,int], pos_xy: Tuple[int,int]) -> int:
        x,y = pos_xy; w,h = dims_xy
        max_top = 0
        for b in self.placed:
            bx,by,bz = b.pos
            bw,bh,bd = b.rotated(b.rot)
            overlap_x = not (x+w<=bx or bx+bw<=x)
            overlap_y = not (y+h<=by or by+bh<=y)
            if overlap_x and overlap_y:
                max_top = max(max_top, bz+bd)
        return max_top

    def support_ratio(self, dims: Tuple[int,int,int], pos: Tuple[int,int,int]) -> float:
        x,y,z = pos; w,h,d = dims
        if z==0:
            return 1.0
        supported = 0
        sample = 0
        for xi in range(x, x+w):
            for yi in range(y, y+h):
                sample += 1
                top = 0
                for b in self.placed:
                    bx,by,bz = b.pos
                    bw,bh,bd = b.rotated(b.rot)
                    if (bx <= xi < bx+bw) and (by <= yi < by+bh):
                        top = max(top, bz+bd)
                if top == z:
                    supported += 1
        if sample == 0: return 0.0
        return supported / sample

    def place(self, box: HBox, rot: int, pos: Tuple[int,int,int]):
        nb = HBox(box.w, box.h, box.d, id=box.id, rot=rot, pos=pos)
        self.placed.append(nb)
        return nb

def lbf_blb(W:int,H:int,D:int, boxes: List[Tuple[int,int,int,int]], support_thresh: float = 0.7) -> Dict[str,Any]:
    """Largest-Box-First + Bottom-Left-Back with stability check."""
    grid = Grid3D(W,H,D)
    order = sorted(boxes, key=lambda b: (-(b[0]*b[1]*b[2]), -max(b[0],b[1],b[2]), -(b[0]*b[1])))
    placements = []
    for (w,h,d,i) in order:
        cand: Optional[Tuple[float,int,Tuple[int,int,int]]] = None
        for rot in range(6):
            rw,rh,rd = HBox(w,h,d,i).rotated(rot)
            for x in range(0, W - rw + 1):
                for y in range(0, H - rh + 1):
                    z = grid.top_z_at((rw,rh), (x,y))
                    pos = (x,y,z)
                    if not grid.fits((rw,rh,rd), pos): 
                        continue
                    if grid.overlaps((rw,rh,rd), pos):
                        continue
                    sup = grid.support_ratio((rw,rh,rd), pos)
                    if sup < support_thresh: 
                        continue
                    score = (-z) + 0.5*sup + -0.01*(x+y)
                    if (cand is None) or (score > cand[0]):
                        cand = (score, rot, pos)
        if cand is not None:
            _, rot, pos = cand
            nb = grid.place(HBox(w,h,d,i), rot, pos)
            placements.append(nb)

    used = sum(p.rotated(p.rot)[0]*p.rotated(p.rot)[1]*p.rotated(p.rot)[2] for p in grid.placed)
    util = used / (W*H*D) if (W*H*D)>0 else 0.0
    return {
        "utilization": util,
        "placed": len(grid.placed),
        "attempted": len(boxes),
        "placements": [{"id": p.id, "rot": p.rot, "pos": p.pos, "dims": p.rotated(p.rot)} for p in grid.placed],
    }
