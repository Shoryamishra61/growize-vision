"""Curved-centreline metrology for elongated produce.

Implements the skeleton-graph method from the engineering blueprint
(sections 10.6 and 18): rasterise the calibrated contour, thin it to a
medial-axis skeleton, build an 8-connected pixel graph, prune short spurs
by *physical* length, then sum the endpoint-to-endpoint path in millimetres.
Terminal radii from the distance transform compensate the tip shortening of
the medial axis.  Ambiguous skeletons (branches, rings, fragments) are
refused; a straight-axis proxy is never substituted silently.

Everything here uses only numpy and OpenCV so the core package gains no
new dependency (no scikit-image, no networkx).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import cv2
import numpy as np

_NEIGHBOURS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)

# Zhang-Suen neighbour order P2..P9 as (dy, dx), clockwise from north.
_ZS_ORDER = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))

MAX_THINNING_ITERATIONS = 400
MAX_SKELETON_NODES = 50_000


@dataclass
class CenterlineResult:
    """Outcome of a skeleton-graph centreline measurement.

    ``length_mm`` is ``None`` unless ``status == "OK"``; callers must turn
    every non-OK status into REVIEW, never into a fallback measurement.
    """

    status: str
    length_mm: float | None = None
    endpoint_count: int = 0
    pruned_branch_count: int = 0
    terminal_radius_start_mm: float | None = None
    terminal_radius_end_mm: float | None = None
    skeleton_nodes: int = 0
    volume_proxy_cm3: float | None = None
    reasons: list[str] = field(default_factory=list)


def _transform(homography: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite transformed coordinates")
    return out.astype(np.float64)


def _local_mm_per_px(homography: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return np.array([])
    # Stack as (N, 3, 2) triples [p, p+ex, p+ey] so grouping survives the
    # round-trip through the flat transform.
    triples = np.concatenate(
        [
            points[:, None, :],
            (points + np.array([1.0, 0.0]))[:, None, :],
            (points + np.array([0.0, 1.0]))[:, None, :],
        ],
        axis=1,
    )
    mm = _transform(homography, triples.reshape(-1, 2)).reshape(-1, 3, 2)
    scales = 0.5 * (
        np.linalg.norm(mm[:, 1] - mm[:, 0], axis=1)
        + np.linalg.norm(mm[:, 2] - mm[:, 0], axis=1)
    )
    return scales


def _rasterize(contour_px: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    contour = np.asarray(contour_px, np.float32).reshape(-1, 1, 2)
    x, y, w, h = cv2.boundingRect(contour)
    canvas = np.zeros((h + 2, w + 2), dtype=np.uint8)
    shifted = contour - np.array([[[x - 1.0, y - 1.0]]], dtype=np.float32)
    cv2.fillPoly(canvas, [np.round(shifted).astype(np.int32)], 1)
    return canvas, (x - 1, y - 1)


def _zhang_suen(mask: np.ndarray) -> tuple[np.ndarray, bool]:
    """Thin to a medial axis; the flag reports full convergence."""
    img = (mask > 0).astype(np.uint8)
    for _ in range(MAX_THINNING_ITERATIONS):
        if not _zhang_suen_pass(img):
            return img, True
    return img, False


def _zhang_suen_pass(img: np.ndarray) -> bool:
    changed = False
    for step in (0, 1):
        padded = np.pad(img, 1)
        views = [
            padded[1 + dy : 1 + dy + img.shape[0], 1 + dx : 1 + dx + img.shape[1]]
            for dy, dx in _ZS_ORDER
        ]
        neighbour_count = np.zeros(img.shape, dtype=np.int16)
        transitions = np.zeros(img.shape, dtype=np.int16)
        for i in range(8):
            neighbour_count += views[i]
            # A(P1): number of 0 -> 1 transitions in the ordered cyclic ring.
            transitions += ((views[i] == 0) & (views[(i + 1) % 8] == 1)).astype(np.int16)
        p2, _, p4, _, p6, _, p8, _ = views
        if step == 0:
            edge = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
        else:
            edge = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
        delete = (
            (img == 1)
            & (neighbour_count >= 2)
            & (neighbour_count <= 6)
            & (transitions == 1)
            & edge
        )
        if delete.any():
            img[delete] = 0
            changed = True
    return changed


def _drop_small_components(skeleton: np.ndarray, min_nodes: int = 3) -> np.ndarray:
    labels = cv2.connectedComponents(skeleton, connectivity=8)[1]
    counts = np.bincount(labels.ravel())
    keep = counts >= min_nodes
    keep[0] = False  # background
    return keep[labels].astype(np.uint8)


def _adjacency(skeleton: np.ndarray) -> dict[tuple[int, int], set[tuple[int, int]]]:
    graph: dict[tuple[int, int], set[tuple[int, int]]] = {}
    ys, xs = np.nonzero(skeleton)
    for y, x in zip(ys.tolist(), xs.tolist()):
        graph[(y, x)] = set()
    for (y, x) in graph:
        for dy, dx in _NEIGHBOURS:
            if (y + dy, x + dx) in graph:
                graph[(y, x)].add((y + dy, x + dx))
    return graph


def _edge_px(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _largest_component(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
) -> tuple[dict[tuple[int, int], set[tuple[int, int]]], int]:
    seen: set[tuple[int, int]] = set()
    best: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for start in graph:
        if start in seen:
            continue
        component = {start}
        queue = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            for other in graph[node]:
                if other not in component:
                    component.add(other)
                    seen.add(other)
                    queue.append(other)
        if len(component) > len(best):
            best = component
    sub = {node: {n for n in graph[node] if n in best} for node in best}
    return sub, len(graph) - len(best)


def _prune_spurs(
    graph: dict[tuple[int, int], set[tuple[int, int]]], prune_px: float
) -> int:
    """Remove leaf branches shorter than ``prune_px``; returns count pruned."""
    pruned = 0
    while True:
        leaves = [node for node, links in graph.items() if len(links) == 1]
        if len(leaves) <= 2:
            break
        removable: list[list[tuple[int, int]]] = []
        for leaf in leaves:
            branch = [leaf]
            length = 0.0
            node, previous = leaf, None
            while True:
                onward = [n for n in graph[node] if n != previous]
                if len(onward) != 1:
                    break  # reached a junction or a degenerate end
                following = onward[0]
                length += _edge_px(node, following)
                branch.append(following)
                previous, node = node, following
                if len(graph[node]) == 1:
                    break  # reached the opposite leaf
            if length < prune_px:
                removable.append(branch)
        if not removable:
            break
        for branch in removable:
            for node in branch[:-1]:  # keep the anchored junction/leaf
                graph.pop(node, None)
            pruned += 1
        for node in [n for n, links in graph.items() if not links]:
            graph.pop(node, None)
        for node in graph:
            graph[node] = {n for n in graph[node] if n in graph}
    return pruned


def _is_cut_vertex(
    graph: dict[tuple[int, int], set[tuple[int, int]]], node: tuple[int, int]
) -> bool:
    """True when removing ``node`` disconnects the remaining graph."""
    neighbours = graph[node]
    if len(neighbours) <= 1:
        return False
    rest = {n for n in graph if n != node}
    start = next(iter(rest))
    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for other in graph[current]:
            if other != node and other not in seen:
                seen.add(other)
                queue.append(other)
    return len(seen) != len(rest)


def _break_cycles(
    graph: dict[tuple[int, int], set[tuple[int, int]]]
) -> int:
    """Delete non-cut vertices until the component is a tree.

    Zhang-Suen thinning leaves small loops where a thick band curves, so the
    medial axis of a bent pod is often a lasso rather than a path.  Removing
    a non-cut vertex only happens inside a loop (every interior vertex of a
    simple path is a cut vertex), and shifts the path by at most one pixel.
    """
    removed = 0
    while True:
        edge_count = sum(len(links) for links in graph.values()) // 2
        if edge_count <= len(graph) - 1:
            break
        candidates = sorted(
            node
            for node, links in graph.items()
            if len(links) >= 2 and not _is_cut_vertex(graph, node)
        )
        if not candidates:
            break
        victim = candidates[0]
        for other in graph[victim]:
            graph[other].discard(victim)
        del graph[victim]
        removed += 1
    return removed


def _path_nodes(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    start: tuple[int, int],
) -> Iterator[tuple[int, int]]:
    """Yield the leaf-to-leaf node sequence starting at ``start``.

    Callers must have verified the graph is a simple path (a tree with
    exactly two leaves); the step cap is a defensive backstop so a cyclic
    skeleton can never hang the pipeline.
    """
    node = start
    previous: tuple[int, int] | None = None
    for _ in range(len(graph) + 1):
        yield node
        onward = [n for n in graph[node] if n != previous]
        if not onward:
            return
        previous, node = node, onward[0]
    raise ValueError("skeleton path did not terminate: cyclic medial axis")


def centerline_analysis(
    contour_px: np.ndarray,
    homography: np.ndarray,
    *,
    prune_mm: float = 3.0,
    mm_per_px_approx: float | None = None,
) -> CenterlineResult:
    """Measure the curved centreline of a calibrated silhouette contour."""
    homography = np.asarray(homography, dtype=np.float64)
    mask, origin = _rasterize(contour_px)
    if int(mask.sum()) < 12:
        return CenterlineResult(status="AMBIGUOUS_SKELETON_TOO_SMALL")

    skeleton, converged = _zhang_suen(mask)
    skeleton = _drop_small_components(skeleton)
    node_count = int(skeleton.sum())
    if node_count == 0:
        return CenterlineResult(status="AMBIGUOUS_SKELETON_TOO_SMALL")
    if node_count > MAX_SKELETON_NODES or not converged:
        return CenterlineResult(
            status="AMBIGUOUS_SKELETON_TOO_COMPLEX",
            skeleton_nodes=node_count,
            reasons=["Thinning did not converge or the skeleton is too large."],
        )

    if mm_per_px_approx is None or mm_per_px_approx <= 0:
        ys, xs = np.nonzero(mask)
        probe = np.column_stack([xs, ys]).astype(np.float32)
        mm_per_px_approx = float(np.median(_local_mm_per_px(homography, probe)))
    if not math.isfinite(mm_per_px_approx) or mm_per_px_approx <= 0:
        return CenterlineResult(status="AMBIGUOUS_SKELETON_TOO_SMALL")
    prune_px = max(prune_mm / mm_per_px_approx, 1.0)

    graph, _dropped = _largest_component(_adjacency(skeleton))
    broken = _break_cycles(graph)
    pruned = _prune_spurs(graph, prune_px)
    endpoints = [node for node, links in graph.items() if len(links) == 1]
    # A connected graph is a simple path iff it has exactly two leaves and
    # edges == nodes - 1.  Thick curved silhouettes can leave small loops in
    # the medial axis; those are ambiguous anatomy, not measurable pods.
    edge_count = sum(len(links) for links in graph.values()) // 2
    if len(graph) > 0 and edge_count != len(graph) - 1:
        return CenterlineResult(
            status="AMBIGUOUS_SKELETON_RING",
            endpoint_count=len(endpoints),
            pruned_branch_count=pruned,
            skeleton_nodes=len(graph),
            reasons=[
                "Skeleton contains a loop; the silhouette may be ring-shaped, "
                "folded, or touching itself. No centreline is measured."
            ],
        )
    if len(endpoints) != 2:
        if not endpoints:
            return CenterlineResult(
                status="AMBIGUOUS_SKELETON_RING",
                pruned_branch_count=pruned,
                skeleton_nodes=len(graph),
                reasons=["Skeleton has no free endpoints (ring or junction mesh)."],
            )
        return CenterlineResult(
            status="AMBIGUOUS_SKELETON_BRANCHED",
            endpoint_count=len(endpoints),
            pruned_branch_count=pruned,
            skeleton_nodes=len(graph),
            reasons=[
                f"Skeleton retains {len(endpoints)} endpoints after pruning "
                f"spurs below {prune_mm:g} mm; anatomy is ambiguous."
            ],
        )

    nodes_px = np.array(
        [[node[1] + origin[0], node[0] + origin[1]] for node in graph],
        dtype=np.float32,
    )
    try:
        nodes_mm = _transform(homography, nodes_px)
    except ValueError:
        return CenterlineResult(status="AMBIGUOUS_SKELETON_TOO_COMPLEX")
    mm_of = {node: nodes_mm[i] for i, node in enumerate(graph)}

    start, end = endpoints
    try:
        sequence = list(_path_nodes(graph, start))
    except ValueError:
        return CenterlineResult(
            status="AMBIGUOUS_SKELETON_RING",
            endpoint_count=2,
            pruned_branch_count=pruned,
            skeleton_nodes=len(graph),
            reasons=["Skeleton path walk did not terminate; medial axis is cyclic."],
        )
    path_mm = sum(
        float(np.linalg.norm(mm_of[b] - mm_of[a])) for a, b in zip(sequence, sequence[1:])
    )

    # Terminal-radius compensation: the medial axis stops at the centre of
    # each end cap, so the tip-to-tip length adds both terminal radii.
    distance_px = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    terminals = np.array(
        [
            [start[1] + origin[0], start[0] + origin[1]],
            [end[1] + origin[0], end[0] + origin[1]],
        ],
        dtype=np.float32,
    )
    terminal_scales = _local_mm_per_px(homography, terminals)
    r_start = float(distance_px[start] * terminal_scales[0])
    r_end = float(distance_px[end] * terminal_scales[1])
    length_mm = path_mm + r_start + r_end

    # Frustum integration along the walked path: series of truncated cones
    # whose radii come from the in-plane distance transform.  This is the
    # slender-axisymmetric model; geometry.py attaches the disclaimers.
    scales_of = dict(zip(graph, _local_mm_per_px(homography, nodes_px)))
    volume_mm3 = 0.0
    for a, b in zip(sequence, sequence[1:]):
        r1 = float(distance_px[a]) * scales_of[a]
        r2 = float(distance_px[b]) * scales_of[b]
        step_mm = float(np.linalg.norm(mm_of[b] - mm_of[a]))
        volume_mm3 += (math.pi / 3.0) * (r1 * r1 + r1 * r2 + r2 * r2) * step_mm

    return CenterlineResult(
        status="OK",
        length_mm=float(length_mm),
        endpoint_count=2,
        pruned_branch_count=pruned,
        terminal_radius_start_mm=r_start,
        terminal_radius_end_mm=r_end,
        skeleton_nodes=len(graph),
        volume_proxy_cm3=volume_mm3 / 1000.0 if volume_mm3 > 0 else None,
    )
