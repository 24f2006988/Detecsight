"""The motion history must not grow without bound as track ids climb."""
from app import config
from app.detector import Detector

d = Detector()
cap = config.MAX_TRACKS_PER_SOURCE

for tid in range(cap * 3):
    d._is_moving("drone-01", tid, 0.5, 0.5)

feed = d._history["drone-01"]
print(f"cap                 : {cap}")
print(f"ids pushed          : {cap * 3}")
print(f"entries retained    : {len(feed)}")
print(f"oldest id retained  : {min(feed)}  (expect {cap * 3 - cap})")
print(f"newest id retained  : {max(feed)}")
assert len(feed) == cap, "history is not bounded"
assert min(feed) == cap * 3 - cap, "eviction is not least-recently-seen"

# a second feed must be tracked independently
d._is_moving("helmet-A", 1, 0.5, 0.5)
print(f"feeds tracked       : {sorted(d._history)}")
print(f"stats()             : {d.stats()}")
assert len(d._history["helmet-A"]) == 1
d.reset_source("drone-01")
assert "drone-01" not in d._history
print("PASS: history bounded, LRU eviction, per-feed isolation, reset")
