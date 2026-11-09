"""Cross-camera re-identification and global vehicle ids (FR-RI-02 .. FR-RI-04).

A track is identified from the *average* of several crop embeddings (temporal
pooling, as in the UWIPL AI-City Track-2 video ReID). The pooled vector is
compared with every vehicle in the feature gallery using cosine similarity; the
best match above the threshold gives the track its global id, otherwise a new
id (VH-001, VH-002, ...) is created. A stored vehicle whose colour or type
clearly differs from the track's cannot be the match (``AttributeGate``). All
embeddings and attributes are persisted so vehicles are re-identified across
sessions.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


@dataclass
class TrackIdentity:
    """Appearance history of one local track waiting for / holding a global id."""

    camera_id: str
    track_id: str
    embeddings: deque = field(default_factory=lambda: deque(maxlen=12))
    gid: str | None = None
    created_new: bool = False
    verified: bool = False
    label: str | None = None
    first_ts: float | None = None  # when this track first appeared (rules out impossible hand-offs)
    colors: Counter = field(default_factory=Counter)  # colour name votes from its crops
    types: Counter = field(default_factory=Counter)   # detector class votes (confidence-weighted)
    provisional: bool = False  # got a new id before any ReID sample: its first samples decide it normally
    timing_link: int = 0  # id came from timing (overlap / route): its next samples are checked by features

    def add(self, vec: np.ndarray) -> None:
        self.embeddings.append(_normalize(np.asarray(vec, dtype=np.float32)))

    def query(self) -> np.ndarray:
        return _normalize(np.mean(np.stack(self.embeddings), axis=0))


@dataclass
class MatchResult:
    gid: str
    similarity: float
    is_new: bool
    # (camera, track) that held this id on another camera with a lower similarity and has to
    # give it up (it gets re-identified without this id)
    displaced: tuple[str, str] | None = None
    candidates: list[tuple[str, float]] = field(default_factory=list)  # ids that passed, best first
    via: str = "features"  # "features" (cosine similarity) | "overlap" (same moment, overlapping camera)


class CrossCameraMatcher:
    """Feature gallery + global id assignment."""

    def __init__(self, threshold: float = 0.75, gallery_size: int = 30, merge_margin: float = 0.05,
                 db=None, id_prefix: str = "VH-", id_width: int = 3, handoff_threshold: float | None = None,
                 follow_threshold: float | None = None, handoff_window: float = 90.0,
                 follow_min: float = 1.0, follow_margin: float = 0.08, overlaps=None, gate=None,
                 live_gap: float = 2.0, steal_margin: float = 0.05, overlap_link: float = 0.0,
                 overlap_min_sim: float = 0.10, overlap_arrival: float = 6.0, route=None, route_max_seconds: float = 60.0,
                 route_back_seconds: float = 20.0, route_skip: int = 1, route_min_sim: float = 0.10,
                 route_margin: float = 0.08, route_override: float = 0.15, route_fifo: bool = True,
                 route_typical_seconds: float = 15.0, route_learn_min: int = 5, route_feature_min: float = 0.35,
                 seq=None, route_bidirectional: bool = True):
        self.threshold = float(threshold)
        # Sequence ReID (reid.sequence_model): a GRU reads each track's crop features in time order and
        # gives one signature per track; the gallery then holds one signature per (camera, track),
        # recomputed as the track's crops arrive, instead of one vector per crop.
        self.seq = seq
        self._owner_raw: dict = {}  # (camera, track) -> its crop features in time order (GRU input)
        # Driving route (cameras in the order a vehicle passes them, e.g. C11 -> C21 -> ... -> C61).
        # A vehicle that just left the previous camera of the route is expected on this one next.
        self.route = [str(c) for c in (route or [])]
        self.route_max = float(route_max_seconds)    # longest drive from one route camera to the next
        self.route_back = float(route_back_seconds)  # the next camera may see it first (views meet / clock skew)
        self.route_skip = int(route_skip)            # route cameras that may miss the vehicle in between
        self.route_min_sim = float(route_min_sim)
        self.route_margin = float(route_margin)
        self.route_override = float(route_override)
        self.route_fifo = bool(route_fifo)
        self.route_bidirectional = bool(route_bidirectional)  # cars drive the route both ways (in and out)
        self.route_typical = float(route_typical_seconds)
        self.route_learn_min = int(route_learn_min)
        self.route_feature_min = float(route_feature_min)
        self.overlap_link = float(overlap_link)  # seconds; 0 = off (see overlap_candidate)
        self.overlap_min_sim = float(overlap_min_sim)
        self.overlap_arrival = float(overlap_arrival)
        self.track_first: dict[tuple[str, str], float] = {}  # (camera, track) holding an id -> when it appeared
        # One vehicle, one place: an id seen on another camera less than live_gap s ago is in use
        # by that camera's track; a new track takes it only with a clearly higher similarity.
        self.live_gap = float(live_gap)
        self.steal_margin = float(steal_margin)
        self.holder: dict[str, tuple[str, str, float]] = {}  # gid -> (camera, track, similarity it got the id with)
        self.gate = gate  # AttributeGate (colour/type) or None
        self.attrs: dict[str, dict[str, Counter]] = {}  # gid -> {"colors": Counter, "types": Counter}
        self.last_rejections: list[tuple[str, float, str]] = []  # (gid, similarity, reason) of the last match
        # camera pairs whose views overlap (the same vehicle can be live in both at once)
        self.overlaps = {frozenset(map(str, p)) for p in (overlaps or []) if len(p) == 2}
        self.follow_min = float(follow_min)  # 1.0 = off (see _clear_follow_match)
        self.handoff_tolerance = 2.0  # seconds (detection timing jitter)
        self.follow_margin = float(follow_margin)
        # Camera-link (spatio-temporal) prior: a vehicle that just LEFT one camera is expected to
        # appear in another one soon, so within `handoff_window` seconds it may be matched at a
        # lower similarity. Vehicles an operator is following get the lowest threshold.
        self.handoff_threshold = float(handoff_threshold if handoff_threshold is not None else threshold)
        self.follow_threshold = float(follow_threshold if follow_threshold is not None else self.handoff_threshold)
        self.handoff_window = float(handoff_window)
        self.followed: set[str] = set()
        self.last_seen: dict[str, tuple[float, str]] = {}  # gid -> (time, camera)
        self.transitions: dict[str, dict[str, list[float]]] = {}  # from cam -> to cam -> travel times
        self.gallery_size = int(gallery_size)
        self.merge_margin = float(merge_margin)
        self.db = db
        self.id_prefix = id_prefix
        self.id_width = id_width
        self.gallery: dict[str, deque] = {}
        self.active: dict[str, set[tuple[str, str]]] = {}
        self._pending_rows: list[tuple[str, str, float, np.ndarray]] = []
        self._seq = db.next_vehicle_seq() if db is not None else 1
        self._lock = threading.RLock()
        self._matrix: np.ndarray | None = None
        self._matrix_gids: list[str] = []
        self._owner: dict[int, tuple[str, str] | None] = {}  # id(gallery vector) -> (camera, track) it came from

    # --------------------------------------------------------------- gallery
    def query_of(self, track: TrackIdentity) -> np.ndarray:
        """The track's appearance: GRU signature of its crops in time order, or their average."""
        if self.seq is not None:
            return self.seq.signature(list(track.embeddings))
        return track.query()

    def load(self, gallery: dict[str, list[np.ndarray]]) -> None:
        with self._lock:
            for gid, vecs in gallery.items():
                dq = deque(maxlen=self.gallery_size)
                for v in reversed(vecs):
                    if self.seq is not None:
                        v = self.seq.signature([v])  # stored crop feature -> one-crop signature
                    v = _normalize(v.astype(np.float32))
                    dq.append(v)
                    self._owner[id(v)] = None
                self.gallery[gid] = dq
            self._matrix = None

    def load_attributes(self, attrs: dict[str, tuple[str | None, str | None]], weight: float = 3.0) -> None:
        """Stored colour/type of known vehicles (from the vehicles table)."""
        with self._lock:
            for gid, (color, vclass) in attrs.items():
                a = self.attrs.setdefault(gid, {"colors": Counter(), "types": Counter()})
                if color:
                    a["colors"][color] += weight
                if vclass:
                    a["types"][vclass] += weight

    def add_attributes(self, gid: str, color: str | None = None, types: Counter | None = None) -> None:
        with self._lock:
            a = self.attrs.setdefault(gid, {"colors": Counter(), "types": Counter()})
            if color:
                a["colors"][color] += 1
            if types:
                a["types"].update(types)

    def attributes(self, gid: str) -> tuple[str | None, str | None]:
        """(colour, type) currently stored for ``gid`` (None where not yet clear)."""
        a = self.attrs.get(gid)
        if not a or self.gate is None:
            return None, None
        return self.gate.describe(a["colors"], a["types"])

    def _attr_conflict(self, gid: str, track: TrackIdentity) -> str | None:
        if self.gate is None:
            return None
        a = self.attrs.get(gid)
        if not a:
            return None
        return self.gate.conflict(track.colors, track.types, a["colors"], a["types"])

    def _build_matrix(self) -> None:
        gids, rows = [], []
        for gid, dq in self.gallery.items():
            for v in dq:
                gids.append(gid)
                rows.append(v)
        self._matrix_gids = gids
        self._matrix = np.stack(rows) if rows else None

    def add_embedding(self, gid: str, vec: np.ndarray, camera_id: str, ts: float | None = None, persist: bool = True,
                      owner: tuple[str, str] | None = None) -> None:
        """Store one embedding of ``gid`` (``owner``: the (camera, track) it came from)."""
        with self._lock:
            dq = self.gallery.setdefault(gid, deque(maxlen=self.gallery_size))
            raw = _normalize(np.asarray(vec, dtype=np.float32))
            if self.seq is not None:
                # this track's signature is recomputed from all its crops so far and replaces its old one
                if owner is not None:
                    seq_raw = self._owner_raw.setdefault(owner, deque(maxlen=self.seq.max_len))
                    seq_raw.append(raw)
                    old = [x for x in dq if self._owner.get(id(x)) == owner]
                    for x in old:
                        dq.remove(x)
                        self._owner.pop(id(x), None)
                    v = _normalize(self.seq.signature(list(seq_raw)))
                else:
                    v = _normalize(self.seq.signature([raw]))
            else:
                v = raw
            dq.append(v)
            self._owner[id(v)] = owner
            self._matrix = None
            if persist and self.db is not None:
                self._pending_rows.append((gid, camera_id, ts or time.time(), raw))  # the crop feature is stored

    def live_elsewhere(self, gid: str, camera_id: str, now: float) -> bool:
        """Is ``gid`` being seen right now on a camera that does not overlap ``camera_id``?"""
        seen = self.last_seen.get(gid)
        return (seen is not None and seen[1] != camera_id and now - seen[0] <= self.live_gap
                and frozenset((seen[1], camera_id)) not in self.overlaps
                and self.route_steps(seen[1], camera_id) not in (1, -1))  # neighbours' views meet at the edges

    def displace(self, gid: str, camera_id: str, track_id: str, since_ts: float, track: TrackIdentity | None = None) -> None:
        """Take ``gid`` away from the track (camera_id, track_id): its embeddings, colour/type votes
        and live state are removed from that vehicle (in memory and in the database)."""
        key = (camera_id, track_id)
        with self._lock:
            self._owner_raw.pop(key, None)
            dq = self.gallery.get(gid)
            if dq is not None:
                keep = [v for v in dq if self._owner.get(id(v)) != key]
                self.gallery[gid] = deque(keep, maxlen=self.gallery_size)
                if not keep:
                    self.gallery.pop(gid, None)
            self._pending_rows = [r for r in self._pending_rows
                                  if not (r[0] == gid and r[1] == camera_id and r[2] >= since_ts)]
            a = self.attrs.get(gid)
            if a and track is not None:
                a["colors"].subtract(track.colors)
                a["types"].subtract(track.types)
                a["colors"] += Counter()  # drop zero / negative counts
                a["types"] += Counter()
            s = self.active.get(gid)
            if s:
                s.discard(key)
                if not s:
                    self.active.pop(gid, None)
            if self.holder.get(gid, ("", "", 0))[:2] == key:
                self.holder.pop(gid, None)
            self._matrix = None
        if self.db is not None:
            self.db.execute("DELETE FROM embeddings WHERE gid=? AND camera_id=? AND ts>=?", (gid, camera_id, since_ts))

    def flush(self) -> None:
        """Write queued embeddings to the feature database (FR-RI-04)."""
        with self._lock:
            rows, self._pending_rows = self._pending_rows, []
        if rows and self.db is not None:
            self.db.add_embeddings(rows)

    # ------------------------------------------------------------ similarity
    @staticmethod
    def computeCosineSimilarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of two embedding vectors (FR-RI-02)."""
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    compute_cosine_similarity = computeCosineSimilarity

    def similarities(self, query: np.ndarray, top_k: int = 3) -> dict[str, float]:
        """Similarity of ``query`` to every gallery vehicle (mean of its top-k matches)."""
        with self._lock:
            if self._matrix is None:
                self._build_matrix()
            if self._matrix is None:
                return {}
            sims = self._matrix @ _normalize(query.astype(np.float32))
            per: dict[str, list[float]] = {}
            for gid, s in zip(self._matrix_gids, sims.tolist()):
                per.setdefault(gid, []).append(s)
        if self.seq is not None:
            top_k = 1  # one signature per track: the vehicle's best-matching visit decides
        return {g: float(np.mean(sorted(v, reverse=True)[:top_k])) for g, v in per.items()}

    # ---------------------------------------------------- camera-link prior
    def touch(self, gid: str, camera_id: str, ts: float) -> None:
        """Record where/when a vehicle was last seen (called on every detection)."""
        self.last_seen[gid] = (ts, camera_id)

    def threshold_for(self, gid: str, camera_id: str, ts: float, first_ts: float | None = None) -> float:
        """Similarity needed to give ``gid`` to a new track on ``camera_id`` at time ``ts``
        (``first_ts``: when that track appeared)."""
        seen = self.last_seen.get(gid)
        if (first_ts is not None and seen is not None and seen[1] != camera_id and gid not in self.active
                and first_ts < seen[0] - self.handoff_tolerance
                and frozenset((seen[1], camera_id)) not in self.overlaps):
            # The track was already on this camera while the vehicle was still elsewhere:
            # it cannot be the vehicle arriving here, so no hand-off leniency.
            return self.threshold
        if gid in self.active:
            # Visible right now. Only cameras that look at the same place (configured overlaps)
            # can show it at the same time: there the hand-off threshold applies.
            cams = {c for c, _ in self.active[gid]}
            if cams and all(frozenset((c, camera_id)) in self.overlaps for c in cams):
                return self.follow_threshold if gid in self.followed else self.handoff_threshold
            return self.threshold
        if seen is None:
            return self.threshold  # unknown timing: strict
        dt = ts - seen[0]
        if seen[1] != camera_id and 0 <= dt <= self.handoff_window:
            return self.follow_threshold if gid in self.followed else self.handoff_threshold
        if gid in self.followed and 0 <= dt <= self.handoff_window:
            return max(self.follow_threshold, self.handoff_threshold)  # came back into the same camera
        return self.threshold

    def _clear_follow_match(self, gid: str, s: float, thr: float, sims: dict[str, float], own: str | None) -> bool:
        """A followed vehicle that just left another camera may also be matched a little below
        follow_threshold (down to follow_min_similarity) when the candidate is clearly closer to it
        than to any other known vehicle (by follow_margin). Seen in another camera the same car
        often scores 0.5-0.55 because of the change in viewpoint."""
        if gid not in self.followed or thr >= self.threshold or s < self.follow_min:
            return False
        # Other ids that themselves look like the followed vehicle are most likely the same car
        # (seen earlier under another id), so they are not counted as competitors.
        look_alikes = self._look_alikes(gid)
        others = [v for k, v in sims.items() if k not in (gid, own) and k not in look_alikes]
        return s - (max(others) if others else 0.0) >= self.follow_margin

    def _look_alikes(self, gid: str) -> set[str]:
        with self._lock:
            vecs = self.gallery.get(gid)
            if not vecs:
                return set()
            centre = _normalize(np.mean(np.stack(list(vecs)), axis=0))
        return {g for g, s in self.similarities(centre).items() if g != gid and s >= self.follow_min}

    def record_transition(self, from_cam: str, to_cam: str, travel_s: float) -> None:
        if frozenset((from_cam, to_cam)) in self.overlaps:
            return  # same place seen from two cameras, not a route to a next camera
        if from_cam and to_cam and from_cam != to_cam:
            self.transitions.setdefault(from_cam, {}).setdefault(to_cam, []).append(travel_s)

    def expected_next(self, from_cam: str) -> list[dict]:
        """Cameras vehicles leaving ``from_cam`` have been seen in next (learned, most frequent first)."""
        out = [{"camera": c, "count": len(t), "typical_s": round(float(np.median(t)), 1)}
               for c, t in self.transitions.get(from_cam, {}).items()]
        return sorted(out, key=lambda d: -d["count"])

    # ------------------------------------------------------------- matching
    def _blocked(self, gid: str, track: TrackIdentity) -> bool:
        """A vehicle cannot be two different tracks of the same camera at once."""
        for cam, tid in self.active.get(gid, set()):
            if cam == track.camera_id and tid != track.track_id:
                return True
        return False

    def _new_gid(self) -> str:
        gid = f"{self.id_prefix}{self._seq:0{self.id_width}d}"
        self._seq += 1
        return gid

    def matchAcrossCameras(self, track: TrackIdentity, ts: float | None = None,
                           exclude: set[str] | frozenset = frozenset(), force_new: bool = False) -> MatchResult:
        """Assign a global id to ``track`` (FR-RI-03) from its features.

        The track's embedding is compared with every stored vehicle. Candidates are the vehicles
        whose similarity reaches the threshold, whose colour/type does not conflict and that are
        not in use elsewhere (an id live on another camera can only be taken with a clearly
        higher similarity than its current holder has). The candidate with the HIGHEST
        similarity gives its id; with no candidate a new id is created. ``exclude``: ids this
        track may not get (it just lost them to a better match)."""
        if not track.embeddings:
            raise ValueError("track has no embeddings")
        q = self.query_of(track)
        now = ts or time.time()
        with self._lock:
            sims = self.similarities(q)
            self.last_rejections = []
            best_any = -1.0
            cands: list[tuple[float, str, tuple[str, str] | None]] = []
            for gid, s in ({} if force_new else sims).items():
                if gid in exclude or self._blocked(gid, track):
                    continue
                best_any = max(best_any, s)
                thr = self.threshold_for(gid, track.camera_id, now, track.first_ts)
                if s < thr and not self._clear_follow_match(gid, s, thr, sims, track.gid):
                    continue
                reason = self._attr_conflict(gid, track)
                if reason:  # features are close but colour/type say it is another vehicle
                    self.last_rejections.append((gid, s, reason))
                    continue
                displaced = None
                if self.live_elsewhere(gid, track.camera_id, now):
                    h = self.holder.get(gid)
                    if (h is None or h[0] != self.last_seen[gid][1] or h[0] == track.camera_id
                            or s < h[2] + self.steal_margin):
                        self.last_rejections.append(
                            (gid, s, f"it is on camera {self.last_seen[gid][1]} right now"
                                     + (f" (matched there at {h[2]:.2f})" if h else "")))
                        continue
                    displaced = (h[0], h[1])  # this track matches better: the holder gives it up
                cands.append((s, gid, displaced))
            cands.sort(key=lambda c: -c[0])
            og = None if (cands or force_new) else self.overlap_candidate(track, now)
            if og in exclude:
                og = None
            rc = None if (force_new or og is not None) else self.route_candidate(track, now)
            if rc is not None and rc[0] in exclude:
                rc = None
            if rc is not None and cands:
                route_ids = {g for g, _s, _d in self.route_candidates(track, now)}
                if cands[0][1] in route_ids or cands[0][0] >= max(rc[1], 0.0) + self.route_override:
                    rc = None  # the feature match is on the route itself, or clearly better
            if rc is not None:
                rg, rs = rc
                result = MatchResult(rg, max(rs, 0.0), False, via="route:" + self.last_seen[rg][1],
                                     candidates=[(g, s) for s, g, _ in cands])
                seen = self.last_seen.get(rg)
                if seen and rg not in self.active and (self.route_steps(seen[1], track.camera_id) or 0) > 0:
                    self.record_transition(seen[1], track.camera_id, now - seen[0])
            elif cands:
                best_sim, best_gid, displaced = cands[0]
                result = MatchResult(best_gid, best_sim, False, displaced, [(g, s) for s, g, _ in cands])
                seen = self.last_seen.get(best_gid)
                if seen and seen[1] != track.camera_id and best_gid not in self.active:
                    self.record_transition(seen[1], track.camera_id, now - seen[0])
            elif og is not None:
                # no appearance match, but the one car seen right now on an overlapping camera
                result = MatchResult(og, max(sims.get(og, 0.0), 0.0), False, via="overlap:" + self.last_seen[og][1])
            else:
                gid = self._new_gid()
                if self.db is not None:
                    self.db.create_vehicle(gid, int(gid[len(self.id_prefix):]), ts or time.time(), track.camera_id, track.label)
                result = MatchResult(gid, max(best_any, 0.0), True)
            track.gid = result.gid
            track.created_new = result.is_new
            a = self.attrs.setdefault(result.gid, {"colors": Counter(), "types": Counter()})
            a["colors"].update(track.colors)
            a["types"].update(track.types)
            self.active.setdefault(result.gid, set()).add((track.camera_id, track.track_id))
            self.track_first[(track.camera_id, track.track_id)] = track.first_ts if track.first_ts is not None else now
            # the track that created an id holds it firmly; a matched track with its similarity
            if not result.via.startswith("route") or result.gid not in self.holder:
                self.holder[result.gid] = (track.camera_id, track.track_id, 1.0 if result.is_new else result.similarity)
            self._owner_raw.pop((track.camera_id, track.track_id), None)  # re-added in order below
            for v in track.embeddings:
                self.add_embedding(result.gid, v, track.camera_id, ts, owner=(track.camera_id, track.track_id))
        return result

    match_across_cameras = matchAcrossCameras

    def overlap_candidate(self, track: TrackIdentity, now: float) -> str | None:
        """Cameras that watch the same place (reid.overlapping_cameras) see one car at the same
        moment, often at very different sizes (a 20-px car far away on one, 250 px close up on the
        other), where appearance features cannot match. If exactly ONE vehicle is being seen on an
        overlapping camera right now (within overlap_link_seconds) and it is not on this camera
        already and its colour/type do not conflict, it is this track's vehicle."""
        partners, cands, sims = self._overlap_cands(track, now)
        if len(cands) <= 1:
            return cands[0] if cands else None
        # Several vehicles on the overlapping camera(s): clearly closest features ...
        by_sim = sorted(cands, key=lambda g: -sims.get(g, -1.0))
        s0, s1 = sims.get(by_sim[0], -1.0), sims.get(by_sim[1], -1.0)
        if s0 >= 0 and s0 - s1 >= self.route_margin:
            return by_sim[0]
        # ... else the one that came into view there at about the same time as this track
        t0 = track.first_ts
        if t0 is None or self.overlap_arrival <= 0:
            return None
        arr = []
        for gid in cands:
            starts = [self.track_first[k] for k in self.active.get(gid, set())
                      if k[0] in partners and k in self.track_first]
            if starts:
                arr.append((min(abs(s - t0) for s in starts), gid))
        arr.sort()
        if arr and arr[0][0] <= self.overlap_arrival and (len(arr) == 1 or arr[1][0] - arr[0][0] >= self.overlap_arrival / 2):
            return arr[0][1]
        return None

    def _overlap_cands(self, track: TrackIdentity, now: float) -> tuple[set[str], list[str], dict[str, float]]:
        """(overlapping cameras, vehicles seen on them within overlap_link_seconds that may be this
        track's vehicle, the track's similarity to every stored vehicle)."""
        if self.overlap_link <= 0:
            return set(), [], {}
        partners = {c for p in self.overlaps if track.camera_id in p for c in p if c != track.camera_id}
        if not partners:
            return set(), [], {}
        cands = []
        sims = self.similarities(self.query_of(track)) if track.embeddings else {}
        for gid, (ts, cam) in list(self.last_seen.items()):
            if cam not in partners or now - ts > self.overlap_link or self._blocked(gid, track):
                continue
            if any(c == track.camera_id for c, _ in self.active.get(gid, set())):
                continue  # already on this camera (another track)
            if self._overlap_conflict(gid, track):
                continue
            if gid in sims and sims[gid] < self.overlap_min_sim:
                continue  # nothing alike at all: another vehicle
            cands.append(gid)
        return partners, cands, sims

    def route_steps(self, from_cam: str, to_cam: str) -> int | None:
        """Route cameras from ``from_cam`` to ``to_cam`` (1 = the next camera, -1 = the previous one)."""
        if from_cam in self.route and to_cam in self.route:
            return self.route.index(to_cam) - self.route.index(from_cam)
        return None

    def route_candidates(self, track: TrackIdentity, now: float) -> list[tuple[str, float, float]]:
        """Vehicles that can be this track's vehicle by the driving route (reid.route): last seen on
        the previous route camera (or up to route_skip more before it) at most route_max_seconds
        per camera ago, or just now (route_back_seconds) on the next camera. Not on this camera
        already, colour/type not conflicting, features not clearly another vehicle.
        Returns (gid, similarity or -1 when unknown, seconds since it was last seen) per candidate."""
        if not self.route or track.camera_id not in self.route:
            return []
        sims = self.similarities(self.query_of(track)) if track.embeddings else {}
        out = []
        for gid, (ts, cam) in list(self.last_seen.items()):
            if gid == track.gid or cam == track.camera_id:
                continue
            steps = self.route_steps(cam, track.camera_id)
            if steps is None:
                continue
            live = {c for c, _ in self.active.get(gid, set())}
            if live and any(abs(self.route_steps(c, track.camera_id) or 9) != 1
                            or frozenset((c, track.camera_id)) in self.overlaps for c in live | {cam}):
                # Still visible: a camera can only be skipped by a vehicle that left, and a vehicle still
                # on an overlapping camera (same place) is handled by overlap_candidate.
                continue
            dt = now - ts
            if 1 <= steps <= 1 + self.route_skip:
                if not -self.handoff_tolerance <= dt <= self.route_max * steps:
                    continue
            elif self.route_bidirectional and 1 <= -steps <= 1 + self.route_skip:
                # driving the other way (e.g. C21 -> C11 on the way out): same wait as forwards
                if not -self.handoff_tolerance <= dt <= self.route_max * -steps:
                    continue
            elif steps == -1:
                if not -self.handoff_tolerance <= dt <= self.route_back:
                    continue
            else:
                continue
            if self._blocked(gid, track) or any(c == track.camera_id for c, _ in self.active.get(gid, set())):
                continue
            if self._overlap_conflict(gid, track):
                continue
            s = sims.get(gid, -1.0)
            if gid in sims and s < self.route_min_sim:
                continue
            out.append((gid, s, dt))
        return out

    def route_candidate(self, track: TrackIdentity, now: float) -> tuple[str, float] | None:
        """The ONE vehicle the route says this track is: the only candidate; or, with several, the
        one whose features are clearly closest (route_margin); otherwise (route_fifo) the one that
        left the previous camera first - vehicles on one road arrive in the order they left."""
        if self._overlap_cands(track, now)[1]:
            # A vehicle is in view right now on a camera watching the same place: the car is one of
            # those (overlap rule), not an older vehicle from further back on the route.
            return None
        cands = self.route_candidates(track, now)
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0][0], cands[0][1]
        by_sim = sorted(cands, key=lambda c: -c[1])
        if by_sim[0][1] >= self.route_feature_min and by_sim[0][1] - by_sim[1][1] >= self.route_margin:
            return by_sim[0][0], by_sim[0][1]  # (lower similarities are noise: front vs rear of one car ~0.24)
        if self.route_fifo:
            forward = [c for c in cands if (self.route_steps(self.last_seen[c[0]][1], track.camera_id) or 0) > 0
                       or self.route_bidirectional]
            if forward:
                # Waiting vehicles: the one whose time since it left fits the usual drive to this camera
                # best (a car that parked on the way must not take every later arrival's id).
                def miss(c):
                    src = self.last_seen[c[0]][1]
                    return abs(c[2] - self.typical_travel(src, track.camera_id))
                best = min(forward, key=miss)
                return best[0], best[1]
        return None

    def typical_travel(self, from_cam: str, to_cam: str) -> float:
        """Usual seconds from leaving ``from_cam`` to appearing on ``to_cam``: learned from matched
        vehicles (median, once route_learn_min have been seen), else route_typical_seconds per camera."""
        t = self.transitions.get(from_cam, {}).get(to_cam, [])
        if len(t) >= self.route_learn_min:
            return float(np.median(t[-50:]))
        return self.route_typical * max(1, abs(self.route_steps(from_cam, to_cam) or 1))

    def similarity_without(self, query: np.ndarray, gid: str, owner: tuple[str, str]) -> float | None:
        """Similarity of ``query`` to ``gid`` from that vehicle's OTHER samples (not the ones the track
        ``owner`` added itself); None when it has no others."""
        with self._lock:
            vecs = [v for v in self.gallery.get(gid, ()) if self._owner.get(id(v)) != owner]
        if not vecs:
            return None
        s = np.stack(vecs) @ _normalize(np.asarray(query, np.float32))
        k = 1 if self.seq is not None else 3
        return float(np.mean(np.sort(s)[::-1][:k]))

    def link_candidates(self, track: TrackIdentity, now: float) -> dict[str, float]:
        """Every vehicle this track can be by timing - on an overlapping camera right now, or waiting
        for this camera on the route - with its appearance similarity (-1 = no features stored).
        Used to check an id that timing gave before the track had features."""
        out = {}
        _, ocands, sims = self._overlap_cands(track, now)
        for g in ocands:
            out[g] = sims.get(g, -1.0)
        for g, s, _dt in self.route_candidates(track, now):
            out.setdefault(g, s)
        return out

    def gids_conflict(self, a: str, b: str) -> bool:
        """Colour / type of two stored vehicles clearly differ (strict check, as for overlap links)."""
        other = self.attrs.get(b)
        if not other:
            return False
        t = TrackIdentity("", "")
        t.colors, t.types = Counter(other["colors"]), Counter(other["types"])
        return self._overlap_conflict(a, t)

    def gallery_centre(self, gid: str) -> np.ndarray:
        """Mean of a vehicle's stored signatures (unit length)."""
        with self._lock:
            return _normalize(np.mean(np.stack(list(self.gallery[gid])), axis=0))

    def _overlap_conflict(self, gid: str, track: TrackIdentity) -> bool:
        """Stricter colour/type check for an overlap link (no appearance evidence backs it): the
        first readings already count, and car vs motorbike always conflicts."""
        if self.gate is None:
            return False
        a = self.attrs.get(gid)
        if not a:
            return False
        if self.gate.conflict(track.colors, track.types, a["colors"], a["types"]):
            return True
        from .attributes import color_relation  # local: attributes imports nothing from here
        tc = track.colors.most_common(1)[0][0] if track.colors else None
        sc = a["colors"].most_common(1)[0][0] if a["colors"] else None
        if color_relation(tc, sc, self.gate.neutral_max_steps) == "conflict":
            return True
        tt = track.types.most_common(1)[0][0] if track.types else None
        st = a["types"].most_common(1)[0][0] if a["types"] else None
        return bool(tt and st and self.gate.type_group.get(tt, tt) != self.gate.type_group.get(st, st))

    def link_overlap(self, track: TrackIdentity, gid: str, ts: float | None = None) -> MatchResult:
        """Give ``track`` the id of the vehicle seen at the same moment on an overlapping camera."""
        with self._lock:
            track.gid, track.created_new, track.provisional = gid, False, False
            a = self.attrs.setdefault(gid, {"colors": Counter(), "types": Counter()})
            a["colors"].update(track.colors)
            a["types"].update(track.types)
            self.active.setdefault(gid, set()).add((track.camera_id, track.track_id))
            self.track_first[(track.camera_id, track.track_id)] = track.first_ts or ts or time.time()
            self._owner_raw.pop((track.camera_id, track.track_id), None)  # re-added in order below
            for v in track.embeddings:
                self.add_embedding(gid, v, track.camera_id, ts, owner=(track.camera_id, track.track_id))
        return MatchResult(gid, 0.0, False)

    def new_vehicle(self, track: TrackIdentity, ts: float | None = None) -> MatchResult:
        """Give ``track`` a new id before it has any ReID sample (its crops are still too small or
        queued). Its first samples are then checked against the stored vehicles (find_merge), so a
        known vehicle is still recognised and the two records merged."""
        with self._lock:
            gid = self._new_gid()
            if self.db is not None:
                self.db.create_vehicle(gid, int(gid[len(self.id_prefix):]), ts or time.time(), track.camera_id, track.label)
            track.gid, track.created_new, track.verified, track.provisional = gid, True, False, True
            a = self.attrs.setdefault(gid, {"colors": Counter(), "types": Counter()})
            a["colors"].update(track.colors)
            a["types"].update(track.types)
            self.active.setdefault(gid, set()).add((track.camera_id, track.track_id))
            self.track_first[(track.camera_id, track.track_id)] = track.first_ts or ts or time.time()
            self.holder[gid] = (track.camera_id, track.track_id, 1.0)
        return MatchResult(gid, 0.0, True)

    def find_merge(self, track: TrackIdentity) -> tuple[str, float] | None:
        """For a track that got a *new* id, check whether more evidence now shows
        it is an already-known vehicle (prevents duplicate global ids, NFR-R-03)."""
        if not track.gid or not track.created_new:
            return None
        q = self.query_of(track)
        sims = self.similarities(q)
        own_cams = {c for c, _ in self.active.get(track.gid, set())}
        now = time.time()
        best = None
        for gid, s in sims.items():
            if gid == track.gid:
                continue
            thr = self.threshold_for(gid, track.camera_id, now, track.first_ts)
            if thr >= self.threshold and not track.provisional:
                # no hand-off context: need extra evidence (an id given without features is decided
                # by its first samples at the normal threshold, like any new track)
                thr = self.threshold + self.merge_margin
            if s < thr and not self._clear_follow_match(gid, s, thr, sims, track.gid):
                continue
            if self._attr_conflict(gid, track):
                continue
            other_cams = {c for c, _ in self.active.get(gid, set())}
            if own_cams & other_cams:
                continue  # both visible in one camera right now -> two vehicles
            if self.live_elsewhere(gid, track.camera_id, now):
                continue  # it is on another camera right now -> another vehicle
            if best is None or s > best[1]:
                best = (gid, s)
        return best

    def merge(self, src: str, dst: str) -> None:
        with self._lock:
            for v in self.gallery.pop(src, []):
                self.gallery.setdefault(dst, deque(maxlen=self.gallery_size)).append(v)
            self.active.setdefault(dst, set()).update(self.active.pop(src, set()))
            sa = self.attrs.pop(src, None)
            if sa:
                da = self.attrs.setdefault(dst, {"colors": Counter(), "types": Counter()})
                da["colors"].update(sa["colors"])
                da["types"].update(sa["types"])
            src_seen, dst_seen = self.last_seen.pop(src, None), self.last_seen.get(dst)
            if dst_seen and src_seen and dst_seen[1] != src_seen[1]:
                self.record_transition(dst_seen[1], src_seen[1], max(0.0, src_seen[0] - dst_seen[0]))
            if src_seen and (not dst_seen or src_seen[0] > dst_seen[0]):
                self.last_seen[dst] = src_seen
            sh = self.holder.pop(src, None)
            if sh and (dst not in self.holder or (src_seen and dst_seen and src_seen[0] > dst_seen[0])):
                self.holder[dst] = sh
            self._pending_rows = [(dst if g == src else g, c, t, v) for g, c, t, v in self._pending_rows]
            self._matrix = None
        if self.db is not None:
            self.flush()
            self.db.merge_vehicle(src, dst)

    def release(self, gid: str | None, camera_id: str, track_id: str) -> None:
        """Called when a local track ends."""
        if gid is None:
            return
        with self._lock:
            self.track_first.pop((camera_id, track_id), None)
            s = self.active.get(gid)
            if s:
                s.discard((camera_id, track_id))
                if not s:
                    self.active.pop(gid, None)
