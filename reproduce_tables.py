#!/usr/bin/env python3
"""Regenerate the result tables and headline numbers of

    Accepted but Wrong: Silent Failures of LLM Code Repairs after API Migrations (arXiv:2611.08269)

from data/ only (standard library, no network, no model calls), write them to out/, and compare every table and number
with the values printed in the paper (data/expected.json).

    python3 reproduce_tables.py

Statistics follow the preregistered analysis: unit = documented-change cluster (task values averaged within a cluster);
sign-flip permutation test (exact for <= 16 clusters, else 200,000 Monte Carlo flips, seed 34361, p = (b+1)/(N+1));
whole-cluster bootstrap 95% interval (10,000 resamples, seed 34360); Holm adjustment within each family (main stage:
hypotheses without discordant clusters are left out; large-menu and hard-menu stages: strict Holm, such hypotheses enter
with p = 1).  The seeds are the preregistered ones; they are needed to reproduce the p-values exactly."""
from __future__ import annotations

import gzip, itertools, json, random, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "out"
MODELS = ["GPT-5.6-luna", "Qwen3-8B", "gpt-oss-20b", "qwen3.6-35b-a3b"]
OPEN_WEIGHT = MODELS[1:]
CONFIRMATORY = {"Qwen3-8B", "gpt-oss-20b"}           # by the frozen power analysis
FA_MAX, FR_TOL, CC_TOL, CONFIRM_DEV_FA = 0.10, 0.05, -0.05, 0.15


def jl(name):
    p = DATA / name
    text = gzip.decompress(p.read_bytes()).decode() if name.endswith(".gz") else p.read_text()
    return [json.loads(x) for x in text.splitlines() if x.strip()]


# ------------------------------------------------------------------------------------------------- statistics
def mean(x):
    x = [v for v in x if v is not None]
    return sum(x) / len(x) if x else None


def contrast(a: dict, b: dict) -> dict:
    cs = sorted(set(a) & set(b))
    d = [a[c] - b[c] for c in cs]
    out = {"clusters": len(cs), "mean_a": mean(a[c] for c in cs), "mean_b": mean(b[c] for c in cs), "mean_difference": mean(d),
           "discordant_clusters": sum(1 for x in d if abs(x) > 1e-12)}
    if not cs:
        return out
    rng = random.Random(34360)
    boots = sorted(mean([rng.choice(d) for _ in d]) for _ in range(10_000))
    out["bootstrap_95"] = [boots[250], boots[9749]]
    if out["discordant_clusters"] == 0:
        out["p"] = None
        return out
    obs = abs(sum(d))
    if len(d) <= 16:
        flips = [abs(sum(s * x for s, x in zip(sg, d))) for sg in itertools.product((1, -1), repeat=len(d))]
        out["p"] = sum(f >= obs - 1e-12 for f in flips) / len(flips)
    else:
        r = random.Random(34361)
        N = 200_000
        b = sum(1 for _ in range(N) if abs(sum((1 if r.random() < 0.5 else -1) * x for x in d)) >= obs - 1e-12)
        out["p"] = (b + 1) / (N + 1)
    return out


def holm(rs):
    ps = sorted([(i, r["p"]) for i, r in enumerate(rs) if r.get("p") is not None], key=lambda x: x[1])
    run = 0.0
    for k, (i, p) in enumerate(ps):
        run = max(run, min(1.0, (len(ps) - k) * p))
        rs[i]["holm_p"] = run
    return rs


def holm_strict(rs):
    m = len(rs)
    order = sorted(range(m), key=lambda i: 1.0 if rs[i].get("p") is None else rs[i]["p"])
    run = 0.0
    for k, i in enumerate(order):
        p = 1.0 if rs[i].get("p") is None else rs[i]["p"]
        run = max(run, min(1.0, (m - k) * p))
        rs[i]["holm_p"] = run
    return rs


def f3(x, nd=3):
    if x is None:
        return "--"
    s = f"{x:.{nd}f}"
    return "$-$" + s[1:] if s.startswith("-") else s


def fp(p):
    return "--" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


# ------------------------------------------------------------------------------------------------- data
TASKS = {t["task_id"]: t for t in jl("tasks.jsonl")}
CLUSTER = {t: r["cluster"] for t, r in TASKS.items()}
PROCS = defaultdict(lambda: defaultdict(list))                  # menu -> task -> [procedure records in index order]
for r in sorted(jl("procedures.jsonl.gz"), key=lambda r: (r["menu"], r["task_id"], r["index"])):
    PROCS[r["menu"]][r["task_id"]].append(r)
REPAIRS = defaultdict(lambda: defaultdict(list))                # menu -> model -> [records]
for r in jl("repairs.jsonl.gz"):
    REPAIRS[r["menu"]][r["model"]].append(r)
SELF = defaultdict(dict)
for r in jl("self_tests.jsonl"):
    SELF[r["model"]][r["task_id"]] = r
PROBE = defaultdict(list)
for r in jl("probe.jsonl"):
    PROBE[r["model"]].append(r)
MENU_KEY = {"main": "main_menu", "large": "large_menu", "hard": "hard_menu"}


def affected(menu):
    k = MENU_KEY[menu]
    return [t for t, r in sorted(TASKS.items()) if r[k]["qualified"] and r["class"] != "C" and r[k]["partition"] == "test"]


def by_cluster(values: dict) -> dict:
    cl = defaultdict(list)
    for t, v in values.items():
        if v is not None:
            cl[CLUSTER[t]].append(v)
    return {c: mean(v) for c, v in cl.items()}


def fa(menu, t, get, origin=None):
    w = [get(e) for e in PROCS[menu][t] if not e["hidden_pass"] and (origin is None or e.get("origin") == origin)]
    return sum(w) / len(w) if w else None


def fr(menu, t, get):
    c = [not get(e) for e in PROCS[menu][t] if e["hidden_pass"]]
    return sum(c) / len(c) if c else None


def lev(name):
    return lambda e: e["released"][name]


def repair_rate(rows, metric, arm, part="test"):
    acc = defaultdict(list)
    for r in rows:
        if r["task_id"] in TASKS and r["partition"] == part and r["class"] != "C" and r["arm"] == arm and r["branch"] == "consumer_complete":
            acc[r["task_id"]].append(float(r[metric]))
    return by_cluster({t: mean(v) for t, v in acc.items()})


def count(rows, metric, arm, branch="consumer_complete"):
    return sum(bool(r[metric]) for r in rows if r["partition"] == "test" and r["arm"] == arm and r["branch"] == branch)


def n_cells(rows, arm, branch="consumer_complete"):
    return sum(1 for r in rows if r["partition"] == "test" and r["arm"] == arm and r["branch"] == branch)


def step1_curves(rows):
    wg = [r for r in rows if r["arm"] == "T" and r["branch"] == "none" and r["partition"] == "test" and r["class"] != "C" and "offline_gates" in r]
    wrong = [r for r in wg if not r["step1_hidden_pass"]]
    right = [r for r in wg if r["step1_hidden_pass"]]
    return {g: {"fa": mean(float(r["offline_gates"][g]) for r in wrong) if wrong else None,
                "fr": mean(float(not r["offline_gates"][g]) for r in right) if right else None, "n_wrong": len(wrong)} for g in (wg[0]["offline_gates"] if wg else {})}


def table(name, caption, label, cols, header, rows, env="table", size="\\small"):
    return (f"\\begin{{{env}}}[t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n{size}\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
            + "\n".join(rows) + f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")


T: dict[str, str] = {}      # table name -> LaTeX
N: dict[str, str] = {}      # headline number -> printed value (paper macro name in data/expected.json)

# ------------------------------------------------------------------------------------------------- main stage, model-free
aff = affected("main")
LEVELS = ["typical", "W1", "W2", "W3", "G1", "G2", "G3", "edge"]
LABEL = {"typical": "typical only", "W1": "+ blind writer, k=1", "W2": "+ blind writer, k=2", "W3": "+ blind writer, k=3", "G1": "+ guarded writer, k=1",
         "G2": "+ guarded writer, k=2", "G3": "+ guarded writer, k=3", "edge": "+ author edge input"}
mm = lambda t: TASKS[t]["main_menu"]  # noqa: E731
rows, curves = [], {}
for lv in LEVELS:
    added = mean((mm(t)["writer_levels"][lv]["n_added"] if lv[0] in "WG" else (1 if lv == "edge" and mm(t)["edge_available"] else 0)) for t in aff)
    stale = sum((mm(t)["writer_levels"][lv]["stale_fails_visibly"] if lv[0] in "WG" else mm(t)["stale_fails_visibly"][lv]) for t in aff)
    c = {"fa": mean(fa("main", t, lev(lv)) for t in aff), "fr": mean(fr("main", t, lev(lv)) for t in aff)}
    curves[lv] = c
    order = ["typical", "W1", "W2", "W3", "G1", "G2", "G3", "edge"]
    rows.append((order.index(lv), f"{LABEL[lv]} & {f3(added, 2)} & {f3(c['fa'])} & {f3(c['fr'])} & {stale}/{len(aff)} \\\\"))
n_clusters = len({CLUSTER[t] for t in aff})
T["curve-table"] = table("curve", f"Model-free release-gate adequacy over the complete finite repair space of the {len(aff)} affected test tasks ({n_clusters} clusters), all-consumer-assertions gate.  "
                         "Added: mean visible inputs added to the typical ones.  FA: share of hidden-wrong procedures released; FR: share of hidden-correct procedures blocked (task means).  "
                         "Stale: tasks in which the unrepaired procedure fails visibly.", "tab:curves", "@{}lrrrr@{}", "Visible inputs & Added & FA & FR & Stale", [r for _, r in sorted(rows)])
N["FMFTypFA"], N["FMFEdgeFA"], N["FMFWThreeFA"] = f3(curves["typical"]["fa"]), f3(curves["edge"]["fa"]), f3(curves["W3"]["fa"])
new = [t for t in aff if TASKS[t]["origin"] == "v4"]
m4c = contrast(by_cluster({t: fa("main", t, lev("edge")) for t in new}), by_cluster({t: fa("main", t, lev("typical")) for t in new}))
N["FNewFAEdge"], N["FNewTestClusters"] = f3(mean(fa("main", t, lev("edge")) for t in new)), str(len({CLUSTER[t] for t in new}))
N["FAuditFlagged"] = str(sum(1 for r in TASKS.values() if r["audit_flagged"]))

# task table
tr = []
for tid, r in sorted(TASKS.items(), key=lambda kv: (kv[1]["runtime"], kv[1]["origin"], kv[1]["class"], kv[0])):
    m = r["main_menu"]
    tr.append(f"\\texttt{{{tid}}} & {r['runtime']} & {r['origin']} & {r['class']} & {m['partition']} & {m['n_correct']}/{m['domain_size']} & "
              f"{'yes' if m['stale_fails_visibly']['typical'] else 'no'} & {'yes' if m['stale_fails_visibly']['edge'] else 'no'} \\\\")
hdr = "Task & Runtime & Origin & Class & Split & Correct & typical & edge \\\\\n\\midrule\n"
T["task-table"] = ("\\begin{longtable}{@{}llllllll@{}}\n\\caption{The " + str(len(TASKS)) + " tasks.  Class: S silent change, E explicit error, B no-safe-repair task, C unchanged control.  "
                   "Correct: hidden-correct procedures in the finite domain.  Last two columns: whether the stale procedure fails a visible all-assertions check with typical inputs and with "
                   "the author edge input added (by the edge-input QA rule always yes for affected tasks).}\\label{tab:tasks}\\\\\n\\toprule\n" + hdr + "\\endfirsthead\n\\toprule\n" + hdr
                   + "\\endhead\n" + "\n".join(tr) + "\n\\bottomrule\n\\end{longtable}\n")

# ------------------------------------------------------------------------------------------------- main stage, model repairs
arms, fams, selfs, gates, probes, h1_main, h2_main = [], [], [], [], [], {}, {}
for model in MODELS:
    mr = REPAIRS["main"][model]
    C = lambda metric, arm, part="test": repair_rate(mr, metric, arm, part)  # noqa: E731
    fam = holm([contrast(C("false_acceptance", "E"), C("false_acceptance", "T")), contrast(C("false_acceptance", "S"), C("false_acceptance", "T")),
                contrast(C("false_acceptance", "ES"), C("false_acceptance", "E")), contrast(C("correct_completion", "S"), C("correct_completion", "T"))])
    cc_et = contrast(C("correct_completion", "E"), C("correct_completion", "T"))
    ok = lambda h: h.get("holm_p") is not None and h["holm_p"] < 0.05 and (h["mean_difference"] or 0) < 0  # noqa: E731
    h1_main[model] = bool(ok(fam[0]) and (cc_et["mean_difference"] or 0) >= CC_TOL)
    h2_main[model] = bool(ok(fam[1]) and (fam[3]["mean_difference"] or 0) >= CC_TOL)
    dev_fa = mean(C("false_acceptance", "T", "development").values())
    status = "confirmatory" if model in CONFIRMATORY or (model == "qwen3.6-35b-a3b" and (dev_fa or 0) >= CONFIRM_DEV_FA) else "descriptive"
    cells = [f"{count(mr, 'correct_completion', a)}/{count(mr, 'false_acceptance', a)}/{count(mr, 'false_rejection', a)}" for a in ("T", "S", "E", "ES")]
    arms.append(f"{model} & " + " & ".join(cells) + f" & {count(mr, 'false_acceptance', 'T', 'none')} \\\\")
    fams.append(f"{model} & {status} & " + " & ".join(f"{f3(h.get('mean_difference'))} ({fp(h.get('holm_p'))})" for h in fam) + " \\\\")
    if model == "Qwen3-8B":
        N["FQwenEightHOneB"], N["FQwenEightHOneA"] = f3(fam[0]["mean_b"]), f3(fam[0]["mean_a"])
    # self-tests
    st = SELF[model]
    sim = mean(x for v in st.values() for x in v["max_similarity_to_hidden"])
    sfa = {g: mean(fa("main", t, lambda e, g=g: e["released_self_tests"][model][g]) for t in aff) for g in ("S", "Sraw")}
    sfr = mean(fr("main", t, lambda e: e["released_self_tests"][model]["Sraw"]) for t in aff)
    selfs.append(f"{model} & {sum(1 for v in st.values() if v['kept'])}/{len(st)} & {sum(v['kept'] for v in st.values())} & {sum(v['kept_legacy_binding'] for v in st.values())} & "
                 f"{sum(v['identical_to_hidden'] for v in st.values())} & {f3(sim, 2)} & {f3(sfa['S'])} & {f3(sfa['Sraw'])} & {f3(sfr)} \\\\")
    cv = step1_curves(mr)
    gates.append(f"{model} & {cv['W1']['n_wrong']} & " + " & ".join(f3(cv[k]["fa"]) for k in ("W1", "W3", "G3", "self_Sraw", "self_S")) + f" & {f3(cv['self_Sraw']['fr'])} \\\\")
    # contamination probe
    t1 = {r["task_id"]: r for r in mr if r["arm"] == "T" and r["rep"] == "full" and not r["diff"] and r["order"] == "listed" and r["branch"] == "none"}
    groups = {"cpython": lambda r: r["runtime"] == "cpython", "node": lambda r: r["runtime"] == "node",
              "sqlite-3.53": lambda r: r["runtime"] == "sqlite" and r["release"] == "3.53.0", "sqlite-3.54": lambda r: r["runtime"] == "sqlite" and r["release"] == "3.54.0"}
    cellp, rand = [], []
    for g, f in groups.items():
        pr = [r for r in PROBE[model] if f(r)]
        cellp.append(f"{sum(1 for r in pr if r['hidden_pass'])}/{len(pr)} ({sum(1 for r in pr if t1.get(r['task_id'], {}).get('step1_hidden_pass'))})")
        rand.append(f"{sum(mm(r['task_id'])['n_correct'] / mm(r['task_id'])['domain_size'] for r in pr):.1f}/{len(pr)}")
    probes.append(f"{model} & " + " & ".join(cellp) + " \\\\")
n_arm = n_cells(REPAIRS["main"][MODELS[0]], "T")
T["arms-table"] = table("arms", "Model repairs on the test partition under the all-consumer-assertions gate: correct completions / false acceptances / false rejections per arm "
                        f"(8 cells per task and arm; {n_arm} records per arm).  T typical inputs; S + the model's own self-tests; E + the author edge input; ES both.  "
                        "Last column: false acceptances with no gate (arm T).", "tab:arms", "@{}lrrrrr@{}", "Model & T & S & E & ES & no gate", arms, env="table*")
T["family-table"] = table("family", "Family A4 per model (test clusters; cluster means of per-attempt rates): difference and Holm-adjusted $p$.  H1 FA(E)$-$FA(T); H2 FA(S)$-$FA(T); "
                          "H3 FA(ES)$-$FA(E); H4 CC(S)$-$CC(T).", "tab:family", "@{}llrrrr@{}", "Model & Status & H1 & H2 & H3 & H4", fams, env="table*")
T["selftest-table"] = table("selftests", "Self-tests written by each model from the release note and the visible contract: tasks with at least one valid test, valid tests, tests declared "
                            "legacy-binding, tests identical to a hidden input, mean highest similarity to a hidden input, and the model-free FA of the gate built from the guarded (S) "
                            "and unguarded (raw) self-tests, with the unguarded gate's FR.", "tab:selftests", "@{}lrrrrrrrr@{}",
                            "Model & Tasks & Tests & Binding & =hidden & Sim. & FA (S) & FA (raw) & FR (raw)", selfs, env="table*")
T["step1-gate-table"] = table("step1", "Offline gates applied to the hidden-wrong first-step candidates of arm T (test partition): share released by the blind-writer gate with 1 and 3 "
                              "inputs (W1, W3), the guarded writer gate (G3), and the model's own unguarded (Sraw) and guarded (S) self-tests; last column: share of hidden-correct "
                              "candidates blocked by Sraw.  All other gates blocked no correct candidate.", "tab:step1", "@{}lrrrrrrr@{}",
                              "Model & wrong & W1 & W3 & G3 & Sraw & S & FR Sraw", gates)
T["probe-table"] = table("probe", "Contamination probe: hidden-correct first-step repairs without any release-note text (in parentheses: with the excerpts, typical inputs, full "
                         "representation, no diff view, listed order).  Uniform random choice from each finite domain would be expected to be hidden-correct in "
                         f"{rand[0]}, {rand[1]}, {rand[2]} and {rand[3]} tasks.", "tab:probe", "@{}lrrrr@{}", "Model & CPython & Node.js & SQLite 3.53 & SQLite 3.54", probes)

# ------------------------------------------------------------------------------------------------- large-menu stage
affL = affected("large")
order = ["typical", "edge", "W1", "W2", "W3", "G1", "G2", "G3"]
cL = {lv: (mean(fa("large", t, lev(lv)) for t in affL), mean(fr("large", t, lev(lv)) for t in affL)) for lv in order}
nproc = sum(len(v) for v in PROCS["large"].values())
T["lm-curve-table"] = table("lm", "Large-menu stage, model-free: share of hidden-wrong (FA) and hidden-correct blocked (FR) procedures over the complete large-menu space of the affected "
                            f"test tasks ({nproc} procedures in all tasks), all-consumer-assertions gate, task means.", "tab:lm", "@{}lrr@{}", "Visible inputs & FA & FR",
                            [f"{LABEL[lv]} & {f3(cL[lv][0])} & {f3(cL[lv][1])} \\\\" for lv in order])
N["FLTotal"] = str(sum(r["main_menu"]["fragments"] + r["large_menu"]["fragments_added"] for r in TASKS.values()))
la, h1_large = [], {}
for model in MODELS:
    mr, m4 = REPAIRS["large"][model], REPAIRS["main"][model]
    fam = holm_strict([contrast(repair_rate(mr, "false_acceptance", "E"), repair_rate(mr, "false_acceptance", "T")),
                       contrast(repair_rate(mr, "false_acceptance", "T"), repair_rate(m4, "false_acceptance", "T")),
                       contrast(repair_rate(mr, "correct_completion", "T"), repair_rate(m4, "correct_completion", "T"))])
    cc = contrast(repair_rate(mr, "correct_completion", "E"), repair_rate(mr, "correct_completion", "T"))
    h1_large[model] = bool(fam[0]["holm_p"] < 0.05 and (fam[0]["mean_difference"] or 0) < 0 and (cc["mean_difference"] or 0) >= CC_TOL)
    status = "confirmatory" if model in CONFIRMATORY else "descriptive"
    la.append(f"{model} & {status} & {count(mr, 'correct_completion', 'T')}/{count(mr, 'false_acceptance', 'T')} & {count(mr, 'correct_completion', 'E')}/{count(mr, 'false_acceptance', 'E')} & "
              f"{f3(fam[0]['mean_b'])}$\\to${f3(fam[0]['mean_a'])} ({fp(fam[0]['holm_p'])}) & {f3(fam[1]['mean_difference'])} ({fp(fam[1]['holm_p'])}) & {f3(fam[2]['mean_difference'])} \\\\")
T["la-table"] = table("la", f"Large-menu stage, model repairs (test partition, {n_cells(REPAIRS['large'][MODELS[0]], 'T')} attempts per arm): correct completions / wrong releases under all "
                      "consumer assertions for arms T and E; LH1 wrong releases per attempt T$\\to$E (strict Holm $p$); LH2 change in wrong releases per attempt with typical inputs "
                      "relative to the v4 menu (strict Holm $p$); LH3 change in correct completion relative to the v4 menu.", "tab:la", "@{}llrrrrr@{}",
                      "Model & Status & T & E & LH1 & LH2 & LH3", la, env="table*")

# ------------------------------------------------------------------------------------------------- hard-menu stage
affH = affected("hard")
rows = []
for lv, lab in (("typical", "typical only"), ("W3", "+ blind writer (GPT-5.6-luna), k=3"), ("G3", "+ guarded writer, k=3"), ("edge", "+ author edge input")):
    a, b, h = mean(fa("hard", t, lev(lv)) for t in affH), mean(fr("hard", t, lev(lv)) for t in affH), mean(fa("hard", t, lev(lv), "hard") for t in affH)
    rows.append(f"{lab} & {f3(a)} & {f3(b)} & {f3(h)} \\\\")
    if lv in ("edge", "W3"):
        k = "Edge" if lv == "edge" else "WThree"
        N[f"FHMF{k}FA"], N[f"FHMF{k}Hard"] = f3(a), f3(h)
wr_ok = {}
for w, lab in (("qwen3.6-35b-a3b", "+ blind writer v2 (qwen3.6), k=3"), ("Qwen3-8B", "+ blind writer v2 (Qwen3-8B), k=3")):
    g = lambda e, w=w: e["released_writer_v2"][w]  # noqa: E731
    vals = {t: fa("hard", t, g) for t in affH}
    a, b = mean(vals.values()), mean(fr("hard", t, g) for t in affH)
    rows.append(f"{lab} & {f3(a)} & {f3(b)} & -- \\\\")
    wr_ok[w] = (vals, a, b)
fam_wr = [contrast(by_cluster(wr_ok[w][0]), by_cluster({t: fa("hard", t, lev("typical")) for t in affH})) for w in ("qwen3.6-35b-a3b", "Qwen3-8B")]
aff_main_v2 = affH                  # WR3: same tasks, main menu; qwen3.6 writer (prompt v2) vs qwen3.6's own unguarded self-tests
fam_wr.append(contrast(by_cluster({t: fa("main", t, lambda e: e["released_writer_v2"]["qwen3.6-35b-a3b"]) for t in aff_main_v2}),
                       by_cluster({t: fa("main", t, lambda e: e["released_self_tests"]["qwen3.6-35b-a3b"]["Sraw"]) for t in aff_main_v2})))
holm_strict(fam_wr)
fr_typ = mean(fr("hard", t, lev("typical")) for t in affH)
for h, w, tag in zip(fam_wr, ("qwen3.6-35b-a3b", "Qwen3-8B"), ("WROne", "WRTwo")):
    ok = h["holm_p"] < 0.05 and (h["mean_difference"] or 0) < 0 and wr_ok[w][1] <= FA_MAX and wr_ok[w][2] - fr_typ <= FR_TOL
    N[f"FH{tag}Decision"] = "positive" if ok else "not positive"
    N[f"FH{tag}Holm"] = fp(h["holm_p"])
nproc = sum(len(v) for v in PROCS["hard"].values())
T["hd-table"] = table("hd", "Hard-distractor stage, model-free over the complete hard-menu space of the affected test tasks "
                      f"({nproc} procedures in all tasks): FA and FR task means under all consumer assertions, and FA restricted to hard-origin programs (wrong options that match the "
                      "legacy output on the typical inputs).", "tab:hd", "@{}lrrr@{}", "Visible inputs & FA & FR & FA hard", rows)
N["FHHard"] = str(sum(r["hard_menu"]["fragments_hard"] for r in TASKS.values()))
ha, h1_hard = [], {}
for model in MODELS:
    mr, m5 = REPAIRS["hard"][model], REPAIRS["large"][model]
    fam = holm_strict([contrast(repair_rate(mr, "false_acceptance", "E"), repair_rate(mr, "false_acceptance", "T")),
                       contrast(repair_rate(mr, "false_acceptance", "T"), repair_rate(m5, "false_acceptance", "T"))])
    cc = contrast(repair_rate(mr, "correct_completion", "E"), repair_rate(mr, "correct_completion", "T"))
    h1_hard[model] = bool(fam[0]["holm_p"] < 0.05 and (fam[0]["mean_difference"] or 0) < 0 and (cc["mean_difference"] or 0) >= CC_TOL)
    cv = step1_curves(mr)
    off = [r["offline_writer_v2"] for r in mr if "offline_writer_v2" in r and r["task_id"] in affH and not r["offline_writer_v2"]["step1_hidden_pass"]]
    x = {w: (mean(float(o[w]) for o in off) if off else None) for w in ("qwen3.6-35b-a3b", "Qwen3-8B")}
    sup = "$^{c}$" if model in CONFIRMATORY else ""
    ha.append(f"{model}{sup} & {count(mr, 'correct_completion', 'T')}/{count(mr, 'false_acceptance', 'T')} & {count(mr, 'correct_completion', 'E')}/{count(mr, 'false_acceptance', 'E')} & "
              f"{f3(fam[0]['mean_b'])}$\\to${f3(fam[0]['mean_a'])} ({fp(fam[0]['holm_p'])}) & {f3(fam[1]['mean_difference'])} ({fp(fam[1]['holm_p'])}) & "
              f"{f3(cv['W3']['fa'])} & {f3(x['qwen3.6-35b-a3b'])} & {f3(x['Qwen3-8B'])} & {f3(cv['self_Sraw']['fa'])} \\\\")
T["ha-table"] = table("ha", f"Hard-distractor stage, model repairs (test partition, {n_cells(REPAIRS['hard'][MODELS[0]], 'T')} attempts per arm): correct completions / wrong releases under all "
                      "consumer assertions in arms T and E; HA1 wrong releases per attempt T$\\to$E (strict Holm $p$); HA2 change of typical-input wrong releases relative to the large "
                      "menu.  Last four columns (descriptive, offline): share of each model's hidden-wrong first-step repairs released by the typical inputs plus three inputs of the "
                      "GPT-5.6-luna writer, the qwen3.6 and Qwen3-8B writers (prompt v2), and the model's own self-tests.  $^{c}$ confirmatory family.", "tab:ha", "@{}lrrrrrrrr@{}",
                      "Model & T & E & HA1 & HA2 & luna & qwen3.6 & Qwen3-8B & own", ha, env="table*", size="\\footnotesize")
# replication across model families (positive = preregistered decision rule met; criterion: >= 3 families and every confirmatory one)
for tag, h in (("FEdgeRep", h1_main), ("FSelfRep", h2_main), ("FLRep", h1_large), ("FHRep", h1_hard)):
    N[tag + "Positive"] = str(sum(h.values()))
    met = sum(h.values()) >= 3 and (tag == "FSelfRep" or all(h[m] for m in CONFIRMATORY))
    N[tag + "Met"] = "met" if met else "not met"
N["EdgeInputPositiveAllOpenWeightAllMenus"] = "yes" if all(h[m] for h in (h1_main, h1_large, h1_hard) for m in OPEN_WEIGHT) else "no"

# ------------------------------------------------------------------------------------------------- post hoc, menu-independent view
def median(x):
    x = sorted(x)
    n = len(x)
    return (x[n // 2] if n % 2 else (x[n // 2 - 1] + x[n // 2]) / 2) if n else None


f_ = lambda x, nd: "--" if x is None else f"{x:.{nd}f}"  # noqa: E731
rows = []
for menu, lab in (("main", "earlier menu"), ("large", "large menu")):
    pm = median([len(PROCS[menu][t]) for t in affL])
    cells = []
    for lv in ("typical", "edge", "W3"):
        n = [sum(1 for e in PROCS[menu][t] if e["released"][lv] and not e["hidden_pass"]) for t in affL]
        cells.append(f"{sum(n)} ({f_(mean(n), 2)}) & {sum(1 for x in n if x > 0)}")
    rows.append(f"{lab} & {f_(pm, 0) if float(pm).is_integer() else f_(pm, 1)} & " + " & ".join(cells) + " \\\\")
mrows = []
for model in MODELS:
    v = []
    for menu in ("main", "large"):
        anyfa = defaultdict(bool)
        for r in REPAIRS[menu][model]:
            if r["task_id"] in affL and r["branch"] == "consumer_complete" and r["arm"] in ("T", "E"):
                anyfa[(r["arm"], r["task_id"])] |= bool(r["false_acceptance"])
        v += [sum(anyfa[(arm, t)] for t in affL) for arm in ("T", "E")]
    mrows.append(f"{model} & " + " & ".join(map(str, v)) + " \\\\")
T["menu-free-table"] = ("\\begin{table}[t]\n\\caption{Post-hoc, menu-independent view (" + str(len(affL)) + " affected test tasks).  Top: wrong programs that pass the all-assertions gate on the "
                        "visible inputs, i.e. that the visible inputs cannot tell apart from a correct one (total, mean per task in parentheses), and tasks with at least one such program.  "
                        "Bottom: tasks in which a model released at least one wrong repair in any of its 8 cells under all assertions, on the earlier and the large menu.}\n\\label{tab:menufree}\n\\small\n"
                        "\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n& Programs & \\multicolumn{2}{c}{T} & \\multicolumn{2}{c}{E} & \\multicolumn{2}{c}{W$_3$} \\\\\nMenu & median & wrong & tasks & wrong & tasks & wrong & tasks \\\\\n\\midrule\n"
                        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\n\\medskip\n\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n& \\multicolumn{2}{c}{earlier menu} & \\multicolumn{2}{c}{large menu} \\\\\nModel & T & E & T & E \\\\\n\\midrule\n"
                        + "\n".join(mrows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ------------------------------------------------------------------------------------------------- write and compare
EXP = json.loads((DATA / "expected.json").read_text())
(OUT / "tables").mkdir(parents=True, exist_ok=True)
for name, tex in T.items():
    (OUT / "tables" / f"{name}.tex").write_text(tex)
(OUT / "numbers.json").write_text(json.dumps(N, indent=1) + "\n")
bad = 0
print("== tables")
for name, want in EXP["tables"].items():
    got = T.get(name)
    ok = got == want
    bad += not ok
    print(f"{'MATCH' if ok else 'MISMATCH':9s} table  {name}")
    if not ok and got:
        for i, (a, b) in enumerate(zip(got.splitlines(), want.splitlines())):
            if a != b:
                print(f"           line {i + 1}: got  {a}\n                    want {b}")
                break
print("== headline numbers (abstract)")
for name, (want, what) in EXP["numbers"].items():
    ok = N.get(name) == want
    bad += not ok
    print(f"{'MATCH' if ok else 'MISMATCH':9s} {N.get(name)!s:>14} {what}" + ("" if ok else f"  (paper: {want})"))
print(f"\n{len(EXP['tables'])} tables, {len(EXP['numbers'])} numbers; outputs in out/")
print("RESULT: " + ("ALL MATCH" if bad == 0 else f"MISMATCH ({bad})"))
sys.exit(1 if bad else 0)
