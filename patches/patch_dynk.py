# dsv41 (v87): hook hybrid/dsv41_dynk.py (per-step DSpark verify width) into the scheduler and the DSpark proposer.
import pathlib
import sys

SITE = pathlib.Path("/usr/local/lib/python3.12/dist-packages")


def sub(rel, old, new):
    p = SITE / rel
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        sys.exit(f"patch_dynk: {rel}: {n} hits for {old[:70]!r}, want 1")
    p.write_text(s.replace(old, new))


S = "vllm/v1/core/sched/scheduler.py"
sub(S, "            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[\n"
       "                len(num_scheduled_tokens)\n            ]\n",
    "            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[\n"
    "                len(num_scheduled_tokens)\n            ]\n"
    "        # dsv41 dynk: per-step DSpark verify width (hybrid/dsv41_dynk.py)\n"
    "        if self.num_spec_tokens > 0 and len(num_scheduled_tokens) > 0 and _dsv41_dynk.enabled():\n"
    "            num_spec_tokens_to_schedule = _dsv41_dynk.choose(self, num_scheduled_tokens)\n")
sub(S, "padded_num_tokens = 1 + self.num_spec_tokens", "padded_num_tokens = 1 + _dsv41_dynk.pad_k(self)")
sub(S, "and num_new_tokens != 1 + self.num_spec_tokens", "and num_new_tokens != 1 + _dsv41_dynk.pad_k(self)")
sub(S, "assert num_new_tokens == 1 + self.num_spec_tokens", "assert num_new_tokens == 1 + _dsv41_dynk.pad_k(self)")
sub(S, "] * self.num_spec_tokens", "] * _dsv41_dynk.pad_k(self)")
sub(S, "                num_rejected = num_draft_tokens - num_accepted\n",
    "                num_rejected = num_draft_tokens - num_accepted\n"
    "                _dsv41_dynk.observe(self, req_id, num_draft_tokens, num_accepted)\n")
sub(S, "        self._dsv41_pp_sent.pop(request.request_id, None)\n        assert request.is_finished()\n",
    "        self._dsv41_pp_sent.pop(request.request_id, None)\n        assert request.is_finished()\n"
    "        _dsv41_dynk.forget(request.request_id)\n")
sub(S, "from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup\n",
    "from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup\n"
    "from hybrid import dsv41_dynk as _dsv41_dynk  # dsv41 dynk\n")

P = "hybrid/dspark_proposer.py"
sub(P, "        assert num_speculative_tokens == self.num_speculative_tokens, (\n",
    "        # dsv41 dynk: the scheduler may verify only the first k drafts of the block (hybrid/dsv41_dynk.py)\n"
    "        assert 1 <= num_speculative_tokens <= self.num_speculative_tokens, (\n")
sub(P, "        return out\n\n    def _dsv41_draft_head_fp8",
    "        if num_speculative_tokens < self.num_speculative_tokens:\n"
    "            out = out[:, :num_speculative_tokens].contiguous()\n"
    "        return out\n\n    def _dsv41_draft_head_fp8")
print("patch_dynk: ok")
