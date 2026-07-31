#!/usr/bin/env python3
"""Parse tools/seedtail_sweep_raw.txt into the tail x prompt-regime table."""
import re, sys, collections

raw = open(sys.argv[1] if len(sys.argv) > 1 else "tools/seedtail_sweep_raw.txt", errors="ignore").read()
legs = re.split(r"=+ LEG TAIL=(\S+)\s+\S+ =+\n", raw)
# legs[0] = preamble, then (tail, body) pairs
rows = []
for i in range(1, len(legs), 2):
    tail, body = legs[i], legs[i + 1]
    seed_lines = re.findall(r"prompt-prefill draft seed ENABLED \(aux tail=(\S+)\)", body)
    boot = re.search(r"prefill_seed=on\(tail=(\S+?)\)", body)
    reqs = {}
    for m in re.finditer(r"REQ (\w+): prompt_tok=(\d+) completion_tok=(\d+) wall=([\d.]+)s "
                         r"TRUE_tok/s=([\d.]+) finish=(\S+) out_chars=(\d+) md5=(\w+)", body):
        reqs[m.group(1)] = dict(ptok=int(m.group(2)), ctok=int(m.group(3)), wall=float(m.group(4)),
                                tps=float(m.group(5)), finish=m.group(6), md5=m.group(8))
    accs = {}
    for m in re.finditer(r"=== uid=(\d+)\s+steps=(\d+)\s+prompt_len=(\d+)\s+emitted=(\d+)\n"
                         r"\s+ACCEPT-LEN emitted/steps = ([\d.]+)", body):
        accs[int(m.group(1))] = dict(steps=int(m.group(2)), plen=int(m.group(3)),
                                     emitted=int(m.group(4)), acc=float(m.group(5)))
    rows.append(dict(tail=tail, seed_lines=seed_lines, boot=boot.group(1) if boot else None,
                     reqs=reqs, accs=accs, body=body))

UID = {"CODE1": 1, "CODE2": 2, "SHORT1": 3, "SHORT2": 4}   # uid is 0-based; uid 0 = warm
plain = next((r for r in rows if r["tail"] == "none"), None)


def cell(r, tag):
    q = r["reqs"].get(tag)
    a = r["accs"].get(UID[tag])
    if not q:
        return None
    ms = 1000.0 * q["wall"] / a["steps"] if a else None
    return dict(tps=q["tps"], acc=a["acc"] if a else 1.0, steps=a["steps"] if a else None,
                ms=ms, ctok=q["ctok"], wall=q["wall"], md5=q["md5"], finish=q["finish"])


print("PROVENANCE")
for r in rows:
    print(f"  tail={r['tail']:>4}  boot_line_tail={r['boot']}  seed-ENABLED lines={r['seed_lines']}")

for tag in ("CODE1", "CODE2", "SHORT1", "SHORT2"):
    print(f"\n===== {tag} =====")
    pl = cell(plain, tag) if plain else None
    print(f"{'tail':>6} {'accept-len':>11} {'steps':>6} {'ctok':>6} {'wall_s':>7} "
          f"{'tok/s':>8} {'ms/step':>8} {'vs PLAIN':>9}")
    for r in rows:
        c = cell(r, tag)
        if not c:
            print(f"{r['tail']:>6}   (missing)")
            continue
        rel = f"{100*(c['tps']/pl['tps']-1):+.1f}%" if pl else "-"
        print(f"{r['tail']:>6} {c['acc']:>11.3f} {str(c['steps']):>6} {c['ctok']:>6} {c['wall']:>7.2f} "
              f"{c['tps']:>8.2f} {(f'{c[chr(109)+chr(115)]:.1f}' if c['ms'] else '-'):>8} {rel:>9}")

print("\n===== P-BUCKET TABLES (CODE1) =====")
for r in rows:
    m = re.search(r"=== uid=1 .*?(?==== uid=|\Z)", r["body"], re.S)
    if m:
        print(f"\n--- tail={r['tail']} ---\n{m.group(0).strip()}")
print("\n===== P-BUCKET TABLES (SHORT1) =====")
for r in rows:
    m = re.search(r"=== uid=3 .*?(?==== uid=|\Z)", r["body"], re.S)
    if m:
        print(f"\n--- tail={r['tail']} ---\n{m.group(0).strip()}")
