import pygame
import numpy as np
import math
import sys
import time
import random
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from collections import deque
from enum import Enum, auto

W, H            = 1200, 780
PLAN_W, PLAN_H  = 860, 780
PANEL_X         = PLAN_W
PANEL_W         = W - PLAN_W
FPS             = 60

STEP_SIZE       = 24       
MAX_NODES       = 3500   
GOAL_BIAS       = 0.12   
GOAL_RADIUS     = 22   
ROBOT_RADIUS    = 10    
REWIRE_RADIUS   = 60   
STEPS_PER_FRAME = 8     
STEPS_FAST      = 120

BG          = (10, 11, 16)
GRID_COL    = (20, 22, 30)
GRID_ACC    = (28, 32, 44)

TREE_BASE   = (30, 55, 95)
TREE_NEW    = (60, 120, 200)
TREE_REWIRE = (120, 80, 200)

PATH_COL    = (255, 210, 50)
PATH_SMOOTH = (50, 220, 150)
PATH_GLOW   = (255, 230, 100)

START_COL   = (50, 200, 120)
GOAL_COL    = (230, 70, 70)
OBS_COL     = (180, 50, 40)
OBS_EDGE    = (230, 80, 60)
OBS_VEL     = (255, 140, 60)

PANEL_BG    = (14, 16, 22)
PANEL_BORD  = (35, 40, 55)
TEXT        = (170, 178, 200)
TEXT_DIM    = (70, 78, 100)
TEXT_HI     = (240, 245, 255)
TEXT_GOOD   = (60, 210, 130)
TEXT_WARN   = (230, 160, 50)
TEXT_BAD    = (220, 70, 70)
WHITE       = (240, 243, 250)
ACCENT      = (80, 140, 255)


class KDNode:
    __slots__ = ("idx", "point", "left", "right")
    def __init__(self, idx, point, left=None, right=None):
        self.idx, self.point, self.left, self.right = idx, point, left, right


class KDTree:

    def __init__(self):
        self.root   = None
        self._pts   = [] 

    def build(self, points: List[Tuple[float, float]]):
        self._pts = points
        indices   = list(range(len(points)))
        self.root = self._build(indices, depth=0)

    def _build(self, indices, depth):
        if not indices:
            return None
        axis = depth % 2
        indices.sort(key=lambda i: self._pts[i][axis])
        mid  = len(indices) // 2
        idx  = indices[mid]
        return KDNode(
            idx, self._pts[idx],
            self._build(indices[:mid],  depth+1),
            self._build(indices[mid+1:], depth+1),
        )

    def nearest(self, query) -> int:
        best = [None, float("inf")]
        self._search(self.root, query, 0, best)
        return best[0]

    def _search(self, node, query, depth, best):
        if node is None:
            return
        axis = depth % 2
        d    = (node.point[0]-query[0])**2 + (node.point[1]-query[1])**2
        if d < best[1]:
            best[0], best[1] = node.idx, d
        diff = query[axis] - node.point[axis]
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
        self._search(near, query, depth+1, best)
        if diff**2 < best[1]:
            self._search(far, query, depth+1, best)

    def within_radius(self, query, radius) -> List[int]:
        results = []
        self._within(self.root, query, radius**2, 0, results)
        return results

    def _within(self, node, query, r2, depth, results):
        if node is None:
            return
        d = (node.point[0]-query[0])**2 + (node.point[1]-query[1])**2
        if d <= r2:
            results.append(node.idx)
        axis = depth % 2
        diff = query[axis] - node.point[axis]
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
        self._within(near, query, r2, depth+1, results)
        if diff**2 <= r2:
            self._within(far, query, r2, depth+1, results)


@dataclass
class Obstacle:
    x:  float
    y:  float
    r:  float
    vx: float
    vy: float
    hue: float = 0.0  

    def update(self, dt: float, speed_scale: float = 1.0):
        self.x += self.vx * dt * speed_scale
        self.y += self.vy * dt * speed_scale
        if self.x - self.r < 0 or self.x + self.r > PLAN_W:
            self.vx = -self.vx
            self.x  = max(self.r, min(PLAN_W - self.r, self.x))
        if self.y - self.r < 0 or self.y + self.r > PLAN_H:
            self.vy = -self.vy
            self.y  = max(self.r, min(PLAN_H - self.r, self.y))

    def collides_point(self, px, py, margin=ROBOT_RADIUS):
        return (px-self.x)**2 + (py-self.y)**2 < (self.r + margin)**2

    def collides_segment(self, ax, ay, bx, by, margin=ROBOT_RADIUS):
        dx, dy = bx-ax, by-ay
        L2 = dx*dx + dy*dy
        if L2 == 0:
            return self.collides_point(ax, ay, margin)
        t  = max(0, min(1, ((self.x-ax)*dx + (self.y-ay)*dy) / L2))
        cx, cy = ax + t*dx, ay + t*dy
        return (cx-self.x)**2 + (cy-self.y)**2 < (self.r + margin)**2


class RRTTree:
    def __init__(self, start):
        self.nodes   = [np.array(start, dtype=float)] 
        self.parent  = [-1]                          
        self.cost    = [0.0]      
        self.kdtree  = KDTree()
        self.dirty   = True 

    def add(self, pos, parent_idx, cost):
        idx = len(self.nodes)
        self.nodes.append(np.array(pos, dtype=float))
        self.parent.append(parent_idx)
        self.cost.append(cost)
        self.dirty = True
        return idx

    def rebuild_kd(self):
        pts = [(n[0], n[1]) for n in self.nodes]
        self.kdtree.build(pts)
        self.dirty = False

    def nearest(self, pos):
        if self.dirty:
            self.rebuild_kd()
        return self.kdtree.nearest((pos[0], pos[1]))

    def near(self, pos, radius):
        if self.dirty:
            self.rebuild_kd()
        return self.kdtree.within_radius((pos[0], pos[1]), radius)

    def path_to(self, idx):
        path = []
        while idx != -1:
            path.append(self.nodes[idx].copy())
            idx = self.parent[idx]
        return path[::-1]


def catmull_rom(points, n_samples=8):
    if len(points) < 4:
        return points
    smooth = []
    pts    = [points[0]] + points + [points[-1]]
    for i in range(1, len(pts)-2):
        p0, p1, p2, p3 = pts[i-1], pts[i], pts[i+1], pts[i+2]
        for j in range(n_samples):
            t  = j / n_samples
            t2 = t*t; t3 = t2*t
            val = 0.5 * (
                2*p1
                + (-p0 + p2)*t
                + (2*p0 - 5*p1 + 4*p2 - p3)*t2
                + (-p0 + 3*p1 - 3*p2 + p3)*t3
            )
            smooth.append(val)
    smooth.append(points[-1])
    return smooth


class RRTSim:

    class State(Enum):
        PLANNING  = auto()
        FOUND     = auto()
        REPLAN    = auto()
        MAX_NODES = auto()

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("RRT* Path Planner  ·  Moving Obstacles")
        self.clock  = pygame.time.Clock()

        self.fnt_ttl = pygame.font.SysFont("Courier New", 17, bold=True)
        self.fnt_lg  = pygame.font.SysFont("Courier New", 14, bold=True)
        self.fnt_md  = pygame.font.SysFont("Courier New", 13)
        self.fnt_sm  = pygame.font.SysFont("Courier New", 11)

        self.plan_surf = pygame.Surface((PLAN_W, PLAN_H))

        self.use_rrt_star   = True
        self.use_goal_bias  = True
        self.use_smoothing  = True
        self.fast_mode      = False
        self.paused         = False
        self.obs_speed      = 1.0

        self._init_scenario()

    def _init_scenario(self):
        self.start = np.array([80.0,  390.0])
        self.goal  = np.array([780.0, 390.0])
        self.tree  = RRTTree(self.start)
        self.state = self.State.PLANNING
        self.path  : List[np.ndarray] = []
        self.smooth: List[np.ndarray] = []
        self.goal_node = -1
        self.stats = {
            "nodes": 0, "iterations": 0,
            "plan_time": 0.0, "replans": 0,
            "path_len": 0.0,
        }
        self.plan_start_time = time.perf_counter()
        self.last_replan     = 0.0
        self.edge_flash      = []  

        rng = random.Random(42)
        self.obstacles: List[Obstacle] = []
        configs = [
            (220, 200, 38, 70,  55),
            (450, 120, 30, -60, 80),
            (600, 550, 42, 50, -70),
            (300, 550, 28, 80, -50),
            (700, 200, 35, -55, 65),
            (150, 600, 25, 65,  45),
            (520, 390, 32, -75, 30),
            (380, 300, 22, 60, -80),
        ]
        for x, y, r, vx, vy in configs:
            self.obstacles.append(Obstacle(x, y, r, vx, vy, rng.random()))

        self.edge_buf: List[Tuple] = [] 

    def collides_point(self, pos):
        for o in self.obstacles:
            if o.collides_point(pos[0], pos[1]):
                return True
        if pos[0] < ROBOT_RADIUS or pos[0] > PLAN_W-ROBOT_RADIUS:
            return True
        if pos[1] < ROBOT_RADIUS or pos[1] > PLAN_H-ROBOT_RADIUS:
            return True
        return False

    def collides_segment(self, a, b):
        for o in self.obstacles:
            if o.collides_segment(a[0], a[1], b[0], b[1]):
                return True
        return False

    def path_clear(self, path):
        for i in range(len(path)-1):
            if self.collides_segment(path[i], path[i+1]):
                return False
        return True

    def _sample(self):
        if self.use_goal_bias and random.random() < GOAL_BIAS:
            return self.goal.copy()
        return np.array([random.uniform(ROBOT_RADIUS, PLAN_W-ROBOT_RADIUS),
                         random.uniform(ROBOT_RADIUS, PLAN_H-ROBOT_RADIUS)])

    def _steer(self, from_pos, to_pos):
        d = to_pos - from_pos
        n = np.linalg.norm(d)
        if n < 1e-6:
            return from_pos.copy()
        return from_pos + (d / n) * min(n, STEP_SIZE)

    def _expand(self):
        if len(self.tree.nodes) >= MAX_NODES:
            self.state = self.State.MAX_NODES
            return

        q_rand  = self._sample()
        near_i  = self.tree.nearest(q_rand)
        near_pt = self.tree.nodes[near_i]
        q_new   = self._steer(near_pt, q_rand)

        if self.collides_point(q_new):
            return
        if self.collides_segment(near_pt, q_new):
            return

        if self.use_rrt_star:
            neighbours = self.tree.near(q_new, REWIRE_RADIUS)
            best_parent = near_i
            best_cost   = self.tree.cost[near_i] + np.linalg.norm(q_new - near_pt)

            for ni in neighbours:
                nc   = self.tree.cost[ni] + np.linalg.norm(q_new - self.tree.nodes[ni])
                if nc < best_cost and not self.collides_segment(self.tree.nodes[ni], q_new):
                    best_parent, best_cost = ni, nc

            new_i = self.tree.add(q_new, best_parent, best_cost)
            self.stats["nodes"] += 1

            for ni in neighbours:
                if ni == best_parent:
                    continue
                nc = best_cost + np.linalg.norm(self.tree.nodes[ni] - q_new)
                if nc < self.tree.cost[ni] and \
                   not self.collides_segment(q_new, self.tree.nodes[ni]):
                    self.tree.parent[ni] = new_i
                    self.tree.cost[ni]   = nc
                    self.edge_buf.append(
                        (q_new[0], q_new[1],
                         self.tree.nodes[ni][0], self.tree.nodes[ni][1], 0, True))
        else:
            cost  = self.tree.cost[near_i] + np.linalg.norm(q_new - near_pt)
            new_i = self.tree.add(q_new, near_i, cost)
            self.stats["nodes"] += 1

        parent_pt = self.tree.nodes[self.tree.parent[new_i]]
        self.edge_buf.append((parent_pt[0], parent_pt[1],
                               q_new[0], q_new[1], 0, False))
        if len(self.edge_buf) > 8000:
            self.edge_buf = self.edge_buf[-6000:]

        self.stats["iterations"] += 1

        if np.linalg.norm(q_new - self.goal) < GOAL_RADIUS and \
           not self.collides_segment(q_new, self.goal):
            if self.state != self.State.FOUND:
                self.stats["plan_time"] = time.perf_counter() - self.plan_start_time
                self.state = self.State.FOUND
                self.goal_node = new_i
                self._extract_path()

    def _extract_path(self):
        self.path = self.tree.path_to(self.goal_node)
        if self.path:
            self.path.append(self.goal.copy())
        if self.use_smoothing and len(self.path) >= 4:
            self.smooth = catmull_rom(self.path)
        else:
            self.smooth = self.path.copy()
        total = 0.0
        for i in range(len(self.path)-1):
            total += np.linalg.norm(self.path[i+1] - self.path[i])
        self.stats["path_len"] = total

    def _replan(self):
        self.tree  = RRTTree(self.start)
        self.state = self.State.PLANNING
        self.path  = []
        self.smooth = []
        self.goal_node = -1
        self.stats["nodes"]      = 0
        self.stats["iterations"] = 0
        self.stats["replans"]   += 1
        self.plan_start_time     = time.perf_counter()
        self.edge_buf            = []

    def step(self, dt):
        if self.paused:
            return

        for o in self.obstacles:
            o.update(dt, self.obs_speed)

        if self.state == self.State.FOUND and self.path:
            if not self.path_clear(self.path):
                self.state = self.State.REPLAN

        if self.state == self.State.REPLAN:
            self._replan()

        if self.state == self.State.PLANNING:
            n = STEPS_FAST if self.fast_mode else STEPS_PER_FRAME
            for _ in range(n):
                self._expand()
                if self.state == self.State.FOUND:
                    break

        self.edge_buf = [(a,b,c,d,age+1,rw) for a,b,c,d,age,rw in self.edge_buf]

    def _lerp_col(self, c1, c2, t):
        t = max(0.0, min(1.0, t))
        return tuple(int(c1[i]+(c2[i]-c1[i])*t) for i in range(3))

    def draw_grid(self, surf):
        for x in range(0, PLAN_W, 40):
            pygame.draw.line(surf, GRID_COL, (x, 0), (x, PLAN_H))
        for y in range(0, PLAN_H, 40):
            pygame.draw.line(surf, GRID_COL, (0, y), (PLAN_W, y))
        for x in range(0, PLAN_W, 200):
            pygame.draw.line(surf, GRID_ACC, (x, 0), (x, PLAN_H))
        for y in range(0, PLAN_H, 200):
            pygame.draw.line(surf, GRID_ACC, (0, y), (PLAN_W, y))

    def draw_obstacles(self, surf):
        for o in self.obstacles:
            glow_surf = pygame.Surface((int(o.r*2+20)*2, int(o.r*2+20)*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*OBS_COL, 25),
                               (int(o.r+20), int(o.r+20)), int(o.r+16))
            surf.blit(glow_surf, (int(o.x-o.r-20), int(o.y-o.r-20)))
            pygame.draw.circle(surf, OBS_COL, (int(o.x), int(o.y)), int(o.r))
            pygame.draw.circle(surf, OBS_EDGE, (int(o.x), int(o.y)), int(o.r), 2)
            spd = math.hypot(o.vx, o.vy)
            if spd > 1:
                nx, ny = o.vx/spd, o.vy/spd
                ax1, ay1 = int(o.x + nx*o.r*0.5), int(o.y + ny*o.r*0.5)
                ax2, ay2 = int(o.x + nx*(o.r+14)), int(o.y + ny*(o.r+14))
                pygame.draw.line(surf, OBS_VEL, (ax1,ay1), (ax2,ay2), 2)

    def draw_tree(self, surf):
        MAX_AGE = 60
        for ax, ay, bx, by, age, rewire in self.edge_buf:
            if age > MAX_AGE and not rewire:
                col = TREE_BASE
            elif rewire:
                t   = min(age / 20, 1.0)
                col = self._lerp_col(TREE_REWIRE, TREE_BASE, t)
            else:
                t   = min(age / MAX_AGE, 1.0)
                col = self._lerp_col(TREE_NEW, TREE_BASE, t)
            pygame.draw.line(surf, col, (int(ax),int(ay)), (int(bx),int(by)), 1)

    def draw_path(self, surf):
        if not self.path:
            return
        draw_pts = self.smooth if self.use_smoothing and self.smooth else self.path

        if len(draw_pts) > 1:
            pts = [(int(p[0]), int(p[1])) for p in draw_pts]
            pygame.draw.lines(surf, (*PATH_GLOW, 40), False, pts, 9)
            col = PATH_SMOOTH if self.use_smoothing else PATH_COL
            pygame.draw.lines(surf, col, False, pts, 3)

        step = max(1, len(draw_pts)//8)
        for i in range(step, len(draw_pts)-1, step):
            p1, p2 = draw_pts[i], draw_pts[i+1]
            dx, dy = p2[0]-p1[0], p2[1]-p1[1]
            n = math.hypot(dx, dy)
            if n < 1e-4: continue
            dx, dy = dx/n, dy/n
            mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
            size   = 6
            tip    = (int(mx+dx*size), int(my+dy*size))
            left   = (int(mx-dy*size*0.5-dx*size*0.5), int(my+dx*size*0.5-dy*size*0.5))
            right  = (int(mx+dy*size*0.5-dx*size*0.5), int(my-dx*size*0.5-dy*size*0.5))
            pygame.draw.polygon(surf, PATH_COL, [tip, left, right])

    def draw_start_goal(self, surf):
        sx, sy = int(self.start[0]), int(self.start[1])
        pygame.draw.circle(surf, START_COL, (sx, sy), 14)
        pygame.draw.circle(surf, WHITE, (sx, sy), 14, 2)
        lbl = self.fnt_sm.render("START", True, WHITE)
        surf.blit(lbl, (sx - lbl.get_width()//2, sy - 26))

        gx, gy = int(self.goal[0]), int(self.goal[1])
        glow = pygame.Surface((GOAL_RADIUS*2+4, GOAL_RADIUS*2+4), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*GOAL_COL, 50),
                           (GOAL_RADIUS+2, GOAL_RADIUS+2), GOAL_RADIUS)
        surf.blit(glow, (gx-GOAL_RADIUS-2, gy-GOAL_RADIUS-2))
        pygame.draw.circle(surf, GOAL_COL, (gx, gy), 14)
        pygame.draw.circle(surf, WHITE, (gx, gy), 14, 2)
        pygame.draw.circle(surf, GOAL_COL, (gx, gy), GOAL_RADIUS, 1)
        lbl = self.fnt_sm.render("GOAL", True, WHITE)
        surf.blit(lbl, (gx - lbl.get_width()//2, gy - 26))

    def draw_panel(self):
        px, py = PANEL_X, 0
        pygame.draw.rect(self.screen, PANEL_BG, (px, py, PANEL_W, H))
        pygame.draw.line(self.screen, PANEL_BORD, (px, 0), (px, H), 1)

        y = 18
        x = px + 14

        def lbl(txt, col=TEXT, font=None, indent=0):
            nonlocal y
            f = font or self.fnt_md
            s = f.render(txt, True, col)
            self.screen.blit(s, (x+indent, y))
            y += s.get_height() + 3

        def divider():
            nonlocal y
            y += 4
            pygame.draw.line(self.screen, PANEL_BORD, (x-4, y), (x+PANEL_W-18, y))
            y += 8

        def badge(txt, col):
            s    = self.fnt_sm.render(f" {txt} ", True, BG)
            w, h = s.get_size()
            bg   = pygame.Surface((w+2, h+2)); bg.fill(col)
            self.screen.blit(bg, (x, y))
            self.screen.blit(s, (x+1, y+1))
            return w+4

        lbl("RRT* PLANNER", TEXT_HI, self.fnt_ttl)
        lbl("Moving Obstacles  ·  Live Replan", TEXT_DIM, self.fnt_sm)
        y += 4; divider()

        alg = "RRT*  (optimal)" if self.use_rrt_star else "RRT   (basic)"
        lbl("ALGORITHM", TEXT_DIM, self.fnt_sm)
        lbl(alg, ACCENT, self.fnt_lg)
        y += 2; divider()

        lbl("STATISTICS", TEXT_DIM, self.fnt_sm)
        y += 2

        nodes = len(self.tree.nodes)
        lbl(f"Nodes grown    {nodes:>5}", TEXT)
        lbl(f"Iterations     {self.stats['iterations']:>5}", TEXT)

        pt = self.stats["plan_time"]
        tcol = TEXT_GOOD if pt < 1.0 else TEXT_WARN if pt < 3.0 else TEXT_BAD
        lbl(f"Plan time      {pt:>5.2f}s", tcol)
        lbl(f"Replans        {self.stats['replans']:>5}", TEXT_WARN if self.stats['replans'] else TEXT)

        if self.stats["path_len"] > 0:
            lbl(f"Path length  {self.stats['path_len']:>7.1f}px", TEXT_GOOD)
        else:
            lbl(f"Path length       —", TEXT_DIM)
        y += 4; divider()

        lbl("STATE", TEXT_DIM, self.fnt_sm)
        y += 4
        state_info = {
            self.State.PLANNING:  ("GROWING TREE", (50, 140, 230)),
            self.State.FOUND:     ("PATH FOUND",   (50, 200, 120)),
            self.State.REPLAN:    ("REPLANNING",   (220, 140, 50)),
            self.State.MAX_NODES: ("MAX NODES",    (200, 60, 60)),
        }
        slbl, scol = state_info.get(self.state, ("UNKNOWN", TEXT_DIM))
        badge(slbl, scol)
        y += 22; divider()

        def tog(txt, val):
            col = TEXT_GOOD if val else TEXT_DIM
            lbl(f"  {'●' if val else '○'}  {txt}", col)

        lbl("SETTINGS", TEXT_DIM, self.fnt_sm)
        y += 2
        tog(f"RRT*  rewiring  [S]", self.use_rrt_star)
        tog(f"Goal bias {GOAL_BIAS:.0%}  [G]", self.use_goal_bias)
        tog(f"Path smoothing  [P]", self.use_smoothing)
        tog(f"Fast mode       [F]", self.fast_mode)
        lbl(f"  Obs speed  {self.obs_speed:.1f}x  [+/-]", TEXT)
        y += 2; divider()

        lbl("LEGEND", TEXT_DIM, self.fnt_sm)
        y += 4
        items = [
            (TREE_BASE,   "Tree edges"),
            (TREE_NEW,    "New edges (flash)"),
            (TREE_REWIRE, "Rewired edges (RRT*)"),
            (PATH_COL,    "Raw path"),
            (PATH_SMOOTH, "Smoothed path"),
            (START_COL,   "Start node"),
            (GOAL_COL,    "Goal node"),
            (OBS_COL,     "Moving obstacle"),
            (OBS_VEL,     "Velocity arrow"),
        ]
        for col, desc in items:
            pygame.draw.rect(self.screen, col, (x, y+3, 14, 8), border_radius=2)
            s = self.fnt_sm.render(f"  {desc}", True, TEXT_DIM)
            self.screen.blit(s, (x+14, y))
            y += s.get_height() + 3
        divider()

        lbl("CONTROLS", TEXT_DIM, self.fnt_sm)
        y += 2
        ctrl = [
            ("LClick",  "move start"),
            ("RClick",  "move goal"),
            ("SPACE",   "pause/resume"),
            ("R",       "full reset"),
            ("S",       "RRT / RRT*"),
            ("G",       "goal bias"),
            ("P",       "smoothing"),
            ("F",       "fast mode"),
            ("+/-",     "obstacle speed"),
            ("Q/ESC",   "quit"),
        ]
        for key, desc in ctrl:
            ks = self.fnt_sm.render(f"[{key}]", True, ACCENT)
            ds = self.fnt_sm.render(f" {desc}", True, TEXT_DIM)
            self.screen.blit(ks, (x, y))
            self.screen.blit(ds, (x + ks.get_width(), y))
            y += ks.get_height() + 3

        fps  = self.clock.get_fps()
        fcol = TEXT_GOOD if fps > 50 else TEXT_WARN if fps > 30 else TEXT_BAD
        fs   = self.fnt_sm.render(f"FPS {fps:.0f}", True, fcol)
        self.screen.blit(fs, (px + PANEL_W - fs.get_width() - 8, H - 18))

        if self.paused:
            ps = self.fnt_ttl.render("  PAUSED  ", True, BG)
            pw2 = ps.get_width() + 16
            prect = pygame.Surface((pw2, 30), pygame.SRCALPHA)
            prect.fill((220, 170, 40, 220))
            self.screen.blit(prect, (px + (PANEL_W-pw2)//2, H//2 - 14))
            self.screen.blit(ps, (px + (PANEL_W-ps.get_width())//2, H//2 - 10))

    def draw_frame(self):
        self.plan_surf.fill(BG)
        self.draw_grid(self.plan_surf)
        self.draw_tree(self.plan_surf)
        self.draw_obstacles(self.plan_surf)
        self.draw_path(self.plan_surf)
        self.draw_start_goal(self.plan_surf)

        ratio = min(len(self.tree.nodes) / MAX_NODES, 1.0)
        bar_w = int(PLAN_W * ratio)
        col   = self._lerp_col(TREE_NEW, (200,60,60), ratio)
        pygame.draw.rect(self.plan_surf, col, (0, PLAN_H-3, bar_w, 3))

        self.screen.blit(self.plan_surf, (0, 0))
        self.draw_panel()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    return False
                elif k == pygame.K_SPACE:
                    self.paused = not self.paused
                elif k == pygame.K_r:
                    self._init_scenario()
                elif k == pygame.K_s:
                    self.use_rrt_star = not self.use_rrt_star
                    self._replan()
                elif k == pygame.K_g:
                    self.use_goal_bias = not self.use_goal_bias
                elif k == pygame.K_p:
                    self.use_smoothing = not self.use_smoothing
                    if self.path:
                        self._extract_path()
                elif k == pygame.K_f:
                    self.fast_mode = not self.fast_mode
                elif k == pygame.K_PLUS or k == pygame.K_EQUALS:
                    self.obs_speed = min(5.0, self.obs_speed + 0.25)
                elif k == pygame.K_MINUS:
                    self.obs_speed = max(0.0, self.obs_speed - 0.25)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if mx < PLAN_W: 
                    if event.button == 1:
                        self.start = np.array([float(mx), float(my)])
                        self._replan()
                    elif event.button == 3:
                        self.goal  = np.array([float(mx), float(my)])
                        self._replan()

        return True

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            running = self.handle_events()
            self.step(dt)
            self.draw_frame()
            pygame.display.flip()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    sim = RRTSim()
    sim.run()
