# dsv41 (v87): per-step DSpark verify width. The drafter always drafts its 5-token block; the scheduler picks how many
# of those drafts the next step verifies (k = 1..5). Every decode request in a step verifies the same k (the async
# scheduler's placeholder list), so the step stays a uniform decode of 1 + k tokens per request and replays the FULL
# graph family captured for that query length (dsv41 qall in the runner). A narrower step drops the attention, dense,
# MoE and all-reduce rows of the drafts least likely to be accepted. Rejection sampling is unchanged, so every output
# token is the target model's (greedy: the same tokens, up to batch-shape numerics).
# Idea from Morrowmake/vllm-cmp170hx adaptive_k.py (Apache-2.0): load- and acceptance-aware verify depth.
#   DSV41_DYNK=""                 off (k = num_speculative_tokens)
#   DSV41_DYNK="fixed:4"          every step verifies 4 drafts (cost measurement)
#   DSV41_DYNK="load:5,5,4,4,3"   k by the number of scheduled requests (the last entry repeats)
#   DSV41_DYNK="accept:3,5"       per step, the k (from the list) that maximises the batch's expected tokens per unit
#                                 of step cost COSTS[R][k]; each request's expected tokens come from its decayed
#                                 per-position acceptance (a position it has not verified lately takes the rate of the
#                                 one before it)
#   DSV41_DYNK_FILE=path          JSON {"mode": "..."} re-read when it changes (dev knob; overrides DSV41_DYNK)
import json
import os
import time

from vllm.logger import init_logger

logger = init_logger("vllm.dsv41.dynk")

# Decode step cost (ms) by scheduled requests R and verified drafts k: v87 MiaAI prose, fixed k, one boot
# (R = 3, 5, 6, 7 interpolated). Prose is where a narrower width can win; its k = 5 steps already skip the MoE rows of
# low-confidence drafts (dsv41_gate), so these are the costs the choice actually trades.
COSTS = {1: {3: 15.4, 4: 16.6, 5: 16.6}, 2: {3: 21.0, 4: 24.5, 5: 24.8}, 3: {3: 25.4, 4: 30.2, 5: 31.0},
         4: {3: 29.7, 4: 35.9, 5: 37.1}, 5: {3: 34.0, 4: 40.2, 5: 41.6}, 6: {3: 38.3, 4: 44.6, 5: 46.1},
         7: {3: 42.5, 4: 48.9, 5: 50.6}, 8: {3: 46.8, 4: 53.2, 5: 55.1}}
PRIOR, STRENGTH, DECAY, HYST = 0.9, 2.0, 0.9, 0.03
LOG_EVERY = 4000
_ST = {"cfg": None, "loaded": False, "src": "", "t": 0.0, "mtime": None, "ever": False, "hist": {}, "n": 0}
_REACH: dict = {}     # req_id -> decayed count of steps that verified position j (j = 0..4)
_ACC: dict = {}       # req_id -> decayed count of steps that accepted position j
_LAST = [None]        # the width the last accept-mode step chose (hysteresis)


def _parse(mode):
    mode = (mode or "").strip()
    if not mode or mode == "off":
        return None
    kind, _, arg = mode.partition(":")
    if kind == "fixed":
        return ("fixed", int(arg))
    if kind == "load":
        return ("load", tuple(int(x) for x in arg.split(",")))
    if kind == "accept":
        return ("accept", tuple(int(x) for x in (arg or "3,5").split(",")))
    raise ValueError(f"DSV41_DYNK: unknown mode {mode!r}")


def _cfg():
    f = os.environ.get("DSV41_DYNK_FILE", "")
    if not f:
        if not _ST["loaded"]:
            _ST["loaded"] = True
            _ST["cfg"] = _parse(os.environ.get("DSV41_DYNK", ""))
            logger.info("dsv41 dynk: mode %r", os.environ.get("DSV41_DYNK", ""))
        return _ST["cfg"]
    now = time.monotonic()
    if not _ST["loaded"] or now - _ST["t"] > 1.0:
        _ST["t"] = now
        try:
            m = os.stat(f).st_mtime
        except OSError:
            m = None
        if not _ST["loaded"] or m != _ST["mtime"]:
            _ST["loaded"], _ST["mtime"] = True, m
            mode = os.environ.get("DSV41_DYNK", "")
            if m is not None:
                try:
                    with open(f) as fh:
                        mode = json.load(fh).get("mode", mode)
                except (OSError, ValueError):
                    pass
            try:
                _ST["cfg"] = _parse(mode)
            except ValueError as e:
                logger.warning("dsv41 dynk: %s (kept %r)", e, _ST["src"])
                return _ST["cfg"]
            _ST["src"] = mode
            _flush_log()
            logger.info("dsv41 dynk: mode %r", mode)
    return _ST["cfg"]


def enabled() -> bool:
    on = _cfg() is not None
    _ST["ever"] |= on
    return on


def pad_k(sched) -> int:
    """Draft count a newly scheduled decode request is padded to: the running requests' current width."""
    k = sched.num_spec_tokens
    if _ST["ever"]:
        k = len(getattr(sched, "_spec_token_placeholders", ())) or k
    return k


def _expected(r: str, k: int) -> float:
    """Expected tokens of a step that verifies k drafts for request r (bonus token included)."""
    reach, acc = _REACH.get(r), _ACC.get(r)
    e, run, c = 1.0, 1.0, PRIOR
    for j in range(k):
        if reach is not None:
            c = (acc[j] + STRENGTH * c) / (reach[j] + STRENGTH)
        run *= c
        e += run
    return e


def _by_acceptance(reqs, R: int, K: int, ks) -> int:
    row = COSTS[min(max(R, 1), max(COSTS))]
    cands = [k for k in ks if k <= K and k in row]
    if not cands:
        return K
    score = {k: sum(_expected(r, k) for r in reqs) / row[k] for k in cands}
    best = max(score, key=lambda k: (score[k], k))
    last = _LAST[0]
    if last in score and last != best and score[best] <= score[last] * (1.0 + HYST):
        best = last
    _LAST[0] = best
    return best


def choose(sched, num_scheduled_tokens) -> int:
    cfg = _cfg()
    K = sched.num_spec_tokens
    if cfg is None:
        return K
    kind, arg = cfg
    reqs = list(num_scheduled_tokens)
    R = len(reqs)
    if kind == "fixed":
        k = arg
    elif kind == "load":
        k = arg[min(max(R, 1), len(arg)) - 1]
    else:
        k = _by_acceptance(reqs, R, K, arg)
    k = max(1, min(K, k))
    h = _ST["hist"].setdefault(R, {})
    h[k] = h.get(k, 0) + 1
    _ST["n"] += 1
    if _ST["n"] >= LOG_EVERY:
        _flush_log()
    return k


def _flush_log() -> None:
    if _ST["n"]:
        logger.info("dsv41 dynk: %d steps, k by requests %s", _ST["n"],
                    {R: dict(sorted(h.items())) for R, h in sorted(_ST["hist"].items())})
    _ST["hist"], _ST["n"] = {}, 0


def observe(sched, req_id: str, num_draft_tokens: int, num_accepted: int) -> None:
    if num_draft_tokens <= 0 or not _ST["ever"]:
        return
    a = min(max(num_accepted, 0), num_draft_tokens)
    reach = _REACH.setdefault(req_id, [0.0] * 5)
    acc = _ACC.setdefault(req_id, [0.0] * 5)
    for j in range(5):
        reach[j] *= DECAY
        acc[j] *= DECAY
    for j in range(min(a + 1, num_draft_tokens, 5)):     # position j was verified: drafts 0..j-1 were accepted
        reach[j] += 1.0
        if j < a:
            acc[j] += 1.0


def forget(req_id: str) -> None:
    _REACH.pop(req_id, None)
    _ACC.pop(req_id, None)
