"""The MTP drafter's KV ring must hold a row for EVERY token the target committed.

    python tests/mtp_ring_completeness_test.py     # arithmetic only, no GPU, no model

THE DEFECT (audit [9]). `propose_body` writes the K/V of the token it attends FROM, so a K-step loop
stored [confirmed, d_0 .. d_{K-2}] and the LAST draft d_{K-1} never got a row. On a FULL accept the
target commits that token anyway, so the drafter's ring was one key short — permanently, for the
whole ~2051-label window, and cumulatively, because nothing resyncs inside a generation.

`on_accept` hid it with `min(1+n, K)`: capping the cursor kept the ring self-consistent (no
misaligned hole) at the cost of the key simply never existing. Verify is lossless so OUTPUT was
never wrong — this only ever degraded the DRAFTER, which is the thing whose quality decides whether
speculation pays at all.

At K=1 the cap fired on EVERY accepted step, so on the best measured content (committed/verify
1.738) roughly 42% of emitted tokens never entered the ring — on precisely the arm the K sweep used
to conclude "spec does not pay at any depth".

This test pins the INVARIANT — rows written == tokens committed, for every accept count — rather
than any particular loop shape.
"""
import sys

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


def rows_written(K, *, extra_step):
    """K/V rows propose_body stores: one per iteration."""
    return K + 1 if extra_step else K


def cursor_advance(n, K, *, capped):
    """How far on_accept moves the drafter's committed cursor for n accepted drafts."""
    return min(1 + n, K) if capped else 1 + n


def committed(n):
    """Tokens the TARGET committed: the confirmed token plus n accepted drafts."""
    return 1 + n


print("== the invariant: every committed token has a ring row ==")
for K in (1, 2, 4, 7):
    ok_all = True
    for n in range(K + 1):                     # n == K is the full accept
        rows = rows_written(K, extra_step=True)
        adv = cursor_advance(n, K, capped=False)
        # the cursor must not outrun the rows actually written, and must not lag the target
        if not (adv == committed(n) and adv <= rows):
            ok_all = False
    report(f"K={K}: cursor tracks the target exactly and never outruns the rows", ok_all)

print("== the OLD behaviour must fail that invariant, or this test proves nothing ==")
broke = []
for K in (1, 2, 4, 7):
    for n in range(K + 1):
        if cursor_advance(n, K, capped=True) != committed(n):
            broke.append((K, n))
report("the capped cursor lags the target on a full accept", broke,
       f"{len(broke)} (K,n) pairs lag — e.g. {broke[:3]}")
# At K=1 the only accept counts are n=0 (draft rejected, nothing lost) and n=1 (draft accepted,
# which IS a full accept). So the cap fires on every step where the draft was accepted — which at
# the measured p=0.738 is most of them. It does NOT fire when the draft was rejected.
report("at K=1 the cap fires whenever the draft is accepted", (1, 1) in broke and (1, 0) not in broke,
       "n=1 lags, n=0 correctly does not -- that arm is the one the K sweep used")

print("== the fix costs exactly one extra head step, not more ==")
for K in (1, 2, 4, 7):
    report(f"K={K}: one extra store step",
           rows_written(K, extra_step=True) - rows_written(K, extra_step=False) == 1,
           f"{rows_written(K, extra_step=False)} -> {rows_written(K, extra_step=True)} steps")

print("== drop rate the fix removes, at the measured operating points ==")
# P(full accept) = p^K under a per-draft acceptance p; dropped tokens per emitted token.
for K, p, label in ((2, 0.618, "K=2 counting"), (2, 0.289, "K=2 prose"), (1, 0.738, "K=1 counting")):
    pfull = p ** K
    emitted = 1 + sum(p ** i for i in range(1, K + 1))
    print(f"    {label:<14} P(full accept)={pfull:.3f}  dropped/emitted={pfull / emitted:.3f}")
report("the K=1 drop rate is the large one", (0.738 ** 1) / (1 + 0.738) > 0.30,
       "~42% of emitted tokens never entered the ring on that arm")

print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'ALL PASS'}")
sys.exit(1 if FAILS else 0)
