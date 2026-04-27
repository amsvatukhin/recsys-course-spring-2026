import json
import pickle
import random
import threading
from collections import defaultdict

import numpy as np
from sklearn.linear_model import SGDRegressor

from .recommender import Recommender

N_FEATURE = 22


class OnlineMFRecommender(Recommender):
    def __init__(self, listen_history_redis, i2i_redis, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback_recommender = fallback_recommender
        self.model = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=5e-6,
            random_state=42,
            learning_rate="invscaling",
            eta0=0.08,
            power_t=0.35,
            average=256,
        )
        self.model_ready = False
        self.training_points = 0
        self.pending = {}
        self.lock = threading.Lock()

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        if not history:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        self._learn_from_feedback(user, prev_track, prev_track_time)
        seen_tracks = {track for track, _ in history}
        candidate_pool = self._build_candidate_pool(
            user, prev_track, prev_track_time, history, seen_tracks
        )
        if not candidate_pool:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        X = np.stack(
            [
                self._dense_features(user, prev_track, prev_track_time, item, anchor, rank, w)
                for item, anchor, rank, w in candidate_pool
            ],
            axis=0,
        )
        scores = self._score_features(X, candidate_pool, prev_track, prev_track_time)
        best_idx = int(np.argmax(scores))
        best_candidate = candidate_pool[best_idx][0]
        with self.lock:
            self.pending[user] = X[best_idx].copy()
        return best_candidate

    def _learn_from_feedback(self, user: int, prev_track: int, prev_track_time: float):
        with self.lock:
            last_row = self.pending.pop(user, None)
        if last_row is None:
            return

        reward = float(max(0.0, min(prev_track_time, 300.0)))
        x = last_row.reshape(1, -1)
        y = np.array([reward], dtype=float)
        with self.lock:
            if self.model_ready:
                self.model.partial_fit(x, y)
            else:
                self.model.partial_fit(x, y)
                self.model_ready = True
        self.training_points += 1

    def _score_features(self, X, candidate_pool, prev_track, prev_track_time):
        prior = np.array(
            [
                self._prior_score(item, anchor, rank, w, prev_track, prev_track_time)
                for item, anchor, rank, w in candidate_pool
            ],
            dtype=float,
        )
        if not self.model_ready or self.training_points < 64:
            return prior
        with self.lock:
            pred = self.model.predict(X)
        return 0.52 * pred + 0.48 * prior

    def _prior_score(self, item, anchor, rank, anchor_weight, prev_track, prev_track_time):
        rank_score = 1.0 / (rank + 1.0)
        aw = float(np.log1p(anchor_weight))
        tl = float(np.log1p(max(prev_track_time, 0.0)))
        ap = 1.0 if int(anchor) == int(prev_track) else 0.0
        return 2.6 * rank_score + 1.05 * aw + 0.32 * tl + 0.95 * ap

    def _dense_features(self, user, prev_track, prev_track_time, item, anchor, rank, anchor_weight):
        r = float(rank)
        aw = float(np.log1p(anchor_weight))
        tl = float(np.log1p(max(prev_track_time, 0.0)))
        ap = 1.0 if int(anchor) == int(prev_track) else 0.0
        u = (user % 64) / 64.0
        pi = (prev_track % 128) / 128.0
        it = (item % 128) / 128.0
        an = (anchor % 128) / 128.0
        v = np.zeros(N_FEATURE, dtype=np.float64)
        v[0] = 1.0 / (r + 1.0)
        v[1] = 1.0 / (r + 4.0)
        v[2] = aw
        v[3] = tl
        v[4] = ap
        v[5] = np.log1p(r)
        v[6] = u
        v[7] = np.sin(2.0 * np.pi * pi)
        v[8] = np.cos(2.0 * np.pi * pi)
        v[9] = np.sin(2.0 * np.pi * it)
        v[10] = np.cos(2.0 * np.pi * it)
        v[11] = np.sin(2.0 * np.pi * an)
        v[12] = np.cos(2.0 * np.pi * an)
        v[13] = (item % 97) / 97.0
        v[14] = (anchor % 97) / 97.0
        v[15] = (prev_track % 97) / 97.0
        v[16] = float(rank < 4)
        v[17] = float(rank < 9)
        v[18] = aw * v[0]
        v[19] = ap * v[0]
        v[20] = tl * v[0]
        v[21] = 1.0
        return v

    def _build_candidate_pool(self, user, prev_track, prev_track_time, history, seen_tracks):
        track_time = defaultdict(float)
        for track, listened_time in history:
            track_time[track] += listened_time
        anchors = list(track_time.keys())
        weights = [track_time[t] for t in anchors]
        max_rank = 20
        max_candidates = 28
        candidates = []
        seen_cand = set()
        tried = set()
        max_attempts = min(24, max(8, len(anchors) * 4))

        head = self._i2i_head_track(seen_tracks, track_time)
        if head is not None:
            t, a, r, w = head
            candidates.append((t, a, r, w))
            seen_cand.add(t)

        for _ in range(max_attempts):
            if len(candidates) >= max_candidates:
                break
            anchor = random.choices(anchors, weights=weights, k=1)[0]
            if len(tried) < len(anchors) and anchor in tried:
                continue
            tried.add(anchor)
            raw = self.i2i_redis.get(anchor)
            if raw is None:
                continue
            items = self._decode_recommendations(raw)
            w = float(track_time[anchor])
            for rank, raw_item in enumerate(items[:max_rank]):
                track = int(raw_item)
                if track in seen_tracks or track in seen_cand:
                    continue
                candidates.append((track, anchor, rank, w))
                seen_cand.add(track)
                if len(candidates) >= max_candidates:
                    return candidates[:max_candidates]
        return candidates[:max_candidates]

    def _i2i_head_track(self, seen_tracks, track_time):
        anchors = list(track_time.keys())
        weights = [track_time[t] for t in anchors]
        work_a = list(anchors)
        work_w = list(weights)
        while work_a:
            anchor = random.choices(work_a, weights=work_w, k=1)[0]
            raw = self.i2i_redis.get(anchor)
            if raw is None:
                idx = work_a.index(anchor)
                work_a.pop(idx)
                work_w.pop(idx)
                continue
            items = self._decode_recommendations(raw)
            for rank, raw_item in enumerate(items):
                track = int(raw_item)
                if track not in seen_tracks:
                    return (track, int(anchor), rank, float(track_time[anchor]))
            idx = work_a.index(anchor)
            work_a.pop(idx)
            work_w.pop(idx)
        return None

    def _decode_recommendations(self, raw):
        try:
            parsed = pickle.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return []

    def _load_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history
