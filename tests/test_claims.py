import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from claims import STALE_MIN, status, try_claim  # noqa: E402


def test_only_one_of_many_simultaneous_claims_wins(tmp_path):
    out = str(tmp_path / "run")
    results = []
    barrier = threading.Barrier(8)

    def worker(i):
        barrier.wait()
        results.append(try_claim(out, f"w{i}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 1
    assert not os.path.exists(os.path.join(out, ".claiming"))


def test_status_lifecycle(tmp_path):
    out = str(tmp_path / "run")
    assert status(out, "a") == "free"
    assert try_claim(out, "a")
    assert status(out, "b") == "running (a)"
    assert status(out, "a") == "free"  # our own claim: a restarted session may resume it
    assert not try_claim(out, "b")

    old = time.time() - (STALE_MIN + 1) * 60  # the session running it died long ago
    for f in os.listdir(out):
        os.utime(os.path.join(out, f), (old, old))
    assert status(out, "b") == "free"
    assert try_claim(out, "b")
    assert json.load(open(os.path.join(out, "CLAIM")))["worker"] == "b"

    open(os.path.join(out, "FAILED"), "w").close()
    assert status(out, "a") == "failed"
    open(os.path.join(out, "summary.json"), "w").close()
    assert status(out, "a") == "done"  # a finished run wins over an earlier failure


def test_lock_released_between_checks_is_not_a_crash(tmp_path, monkeypatch):
    """Another worker holds the lock, then releases it before we can read its age."""
    import claims

    out = str(tmp_path / "run")
    os.makedirs(os.path.join(out, ".claiming"))

    def vanished(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(claims.os.path, "getmtime", vanished)
    assert claims.try_claim(out, "a") is False
