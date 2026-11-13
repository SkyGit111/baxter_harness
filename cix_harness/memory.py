"""Local, provenance-aware episodic memory. No images, coordinates or action replay."""
import hashlib
import json
import re
import sqlite3
import time
import uuid
from pathlib import Path


def geometry_fingerprint(cfg):
    geometry = {k: cfg.get(k) for k in ('arm', 'camera_serial', 'workspace_min_m', 'workspace_max_m',
                'policy_to_base_xyzw', 'contact_guard', 'commissioning_scope')}
    geometry['primitives'] = {k: v for k, v in cfg.get('primitives', {}).items()
                              if k not in ('calibrated', 'notes')}
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]


def scope_key(health, route, model):
    fingerprint = health.get('environment_fingerprint')
    if not isinstance(fingerprint, str) or not fingerprint:
        return None  # Older bridges cannot establish a trustworthy cross-run scope.
    return json.dumps([fingerprint, health.get('arm'), route, model], separators=(',', ':'))


def terms(text):
    words = set(re.findall(r'[a-z0-9]+', text.lower()))
    for run in re.findall(r'[\u4e00-\u9fff]+', text):
        words.update(run[i:i+2] for i in range(max(1, len(run)-1)))
    return words


class EpisodeMemory:
    def __init__(self, path):
        self.path = str(path)
        if self.path != ':memory:': Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=5)
        self.db.execute('''CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY, created REAL NOT NULL, scope TEXT, route TEXT NOT NULL,
            goal TEXT NOT NULL, mode TEXT NOT NULL, status TEXT NOT NULL, summary TEXT NOT NULL)''')
        self.db.execute('CREATE INDEX IF NOT EXISTS memory_scope ON episodes(scope, created)')
        self.db.commit()

    def start(self, scope, route, goal, mode):
        if mode not in ('live', 'preview', 'simulation'): raise ValueError('Invalid memory provenance')
        episode = uuid.uuid4().hex
        with self.db:
            self.db.execute('INSERT INTO episodes VALUES(?,?,?,?,?,?,?,?)',
                            (episode, time.time(), scope, route, goal, mode, 'running', '{}'))
        return episode

    def finish(self, episode, status, verified_steps, failures, recoveries=()):
        if status not in ('verified', 'blocked', 'unknown', 'preview'): raise ValueError('Invalid episode outcome')
        # Persist only a semantic recipe and classified failures, never UV/XYZ/joints/commands.
        recipe = [{k: step[k] for k in ('skill', 'target', 'expect', 'backend') if k in step}
                  for step in verified_steps[-6:]]
        summary = json.dumps({'recipe': recipe, 'failures': list(dict.fromkeys(failures))[-4:],
                              'recovery_attempts': list(dict.fromkeys(recoveries))[-3:]}, ensure_ascii=False)
        with self.db:
            self.db.execute('UPDATE episodes SET status=?, summary=? WHERE id=? AND status=?',
                            (status, summary, episode, 'running'))

    def retrieve(self, goal, scope, limit=2, max_age_days=30):
        if not scope: return []
        query = terms(goal)
        if not query: return []
        rows = self.db.execute('''SELECT id,goal,status,summary FROM episodes
            WHERE scope=? AND mode='live' AND status IN ('verified','blocked') AND created>=?
            ORDER BY created DESC LIMIT 200''', (scope, time.time()-max_age_days*86400)).fetchall()
        ranked = []
        for episode, old_goal, status, raw in rows:
            other = terms(old_goal)
            score = len(query & other)/max(1, len(query | other))
            if score < .15: continue
            summary = json.loads(raw)
            # An incomplete episode is not advertised as a successful recipe.
            item = {'episode': episode[:12], 'outcome': status, 'goal': old_goal[:100],
                    'recipe': summary['recipe'][:3] if status == 'verified' else [],
                    'failures': summary['failures'], 'recovery_attempts': summary.get('recovery_attempts', [])}
            ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in ranked[:limit]]

    def close(self):
        self.db.close()
