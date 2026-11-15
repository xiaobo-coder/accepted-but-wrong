# Accepted but Wrong: Silent Failures of LLM Code Repairs after API Migrations

Task records, scored repairs and analysis code for the paper *Accepted but Wrong: Silent Failures of LLM Code Repairs
after API Migrations* (Xiaobo Chen, Genliang Zhu, Ziyuan Wang, Chu Wang, 2026). arXiv:
[2611.08269](https://arxiv.org/abs/2611.08269).

After a runtime or API migration (CPython, Node.js, SQLite), a repaired procedure can run cleanly yet return wrong
results. Each of the 73 tasks has a finite repair space: every procedure that can be built from a menu of documented code
fragments. Every procedure was executed on the real old and new runtimes and judged against a withheld contract (hidden
pass). The study asks how many hidden-wrong procedures a release gate lets through when it checks all consumer
assertions on the visible inputs only: the typical inputs, plus an author edge input written from the release note,
writer-generated inputs, or the repairing model's own self-tests. Three menus were frozen in turn: the main menu, a large
menu (1491 fragments), and a hard menu that adds 233 distractors matching the legacy output on the typical inputs. Four
model families repaired every task (8 cells per task and arm).

## Contents

- `data/tasks.jsonl`: the 73 tasks, one per line: requirement, release-note quotes (with URLs), procedure template,
  fragment domains, the unrepaired procedure, the documented change, output schema, the **visible** inputs (typical and
  edge), and per-menu metadata (split, domain size, counts).
- `data/procedures.jsonl.gz`: the complete finite repair spaces of the three menus (4,780 procedures), with the hidden
  verdict and the verdict of every visible-input gate.
- `data/repairs.jsonl.gz`: 37,376 scored model repairs (main menu: arms T, S, E, ES; large and hard menus: arms T, E),
  one line per cell and gate branch.
- `data/self_tests.jsonl`: summary of the self-tests each model wrote (main menu); `data/probe.jsonl`: the
  contamination probe (repairs without any release-note text).
- `data/expected.json`: the values printed in the paper (its result tables verbatim and the headline numbers).
- `reproduce_tables.py`: recomputes all of them from `data/` and compares.

## Reproduce

```bash
python3 reproduce_tables.py      # about 1 minute (Monte Carlo permutation tests); standard library only
```

It writes `out/tables/*.tex` and `out/numbers.json`, prints `MATCH` or `MISMATCH` per item and ends with
`RESULT: ALL MATCH` (exit code 0). Covered: Tables of the tasks, the model-free gate curve, model repairs per arm, the
preregistered hypothesis family per model, self-tests, offline first-step gates and the contamination probe (main
menu); the large-menu curve and repairs; the hard-menu curve and repairs; the post-hoc menu-independent view; and the
abstract's numbers, including the replication of the edge-input effect across model families on all three menus.

Statistics follow the preregistration: the unit is the documented-change cluster; sign-flip permutation test (exact up
to 16 clusters, else 200,000 Monte Carlo flips) with whole-cluster bootstrap intervals and Holm adjustment within each
family. The random seeds are the preregistered ones, so p-values are reproduced exactly.

## Field reference

`tasks.jsonl`

| field | meaning |
| --- | --- |
| `task_id`, `runtime`, `cluster` | task, runtime (`cpython`, `node`, `sqlite`), documented-change cluster (the statistical unit) |
| `class` | `S` silent change, `E` explicit error, `B` no safe repair exists, `C` unchanged control |
| `origin`, `release` | construction round (`v2`, `v3`, `v4`); SQLite release for the post-cutoff tasks |
| `runtime_old`, `runtime_new` | runtime versions before and after the migration |
| `requirement`, `quotes`, `diff` | the requirement text, verbatim release-note quotes with URLs, the documented change |
| `template`, `fields`, `old_procedure`, `schema` | procedure template, fragment domains per field, unrepaired procedure, consumer output schema |
| `visible_inputs` | `typical` inputs and the author `edge` input |
| `audit_flagged` | the blind AI audit flagged the contract as stricter than the requirement |
| `main_menu` | `partition` (`development`/`test`), `qualified`, `n_correct`, `domain_size`, `edge_available`, `stale_fails_visibly` (typical/edge), `fragments`, and per writer level (`W1`-`W3` blind writer, `G1`-`G3` guarded writer) `n_added`, `overlap_with_hidden`, `stale_fails_visibly` |
| `large_menu`, `hard_menu` | split, domain size, fragments added; hard menu: `hard_programs`, `hard_wrong`, `fragments_hard`, `fragments_total`, `writer_v2_inputs_kept` |

`procedures.jsonl.gz`

| field | meaning |
| --- | --- |
| `menu`, `task_id`, `index` | `main`, `large` or `hard`; task; position in the repair space |
| `procedure`, `origin` | the chosen fragment per field; the newest menu part it uses (`main`, `large`, `hard`) |
| `hidden_pass` | the procedure meets the withheld contract on the new runtime |
| `released` | passes the all-consumer-assertions gate on: `typical` inputs; plus `W1`-`W3` blind-writer inputs (GPT-5.6-luna); `G1`-`G3` guarded-writer inputs; the `edge` input |
| `released_self_tests` | main menu: gate on each model's guarded (`S`) or unguarded (`Sraw`) self-tests, and `ES` (edge input + self-tests) |
| `released_writer_v2` | gate on typical inputs plus three inputs of the local writers (prompt v2) |

`repairs.jsonl.gz`

| field | meaning |
| --- | --- |
| `menu`, `model`, `task_id`, `partition`, `class` | menu, repairing model, task, split, task class |
| `arm` | visible inputs given to the model and the gate: `T` typical, `S` + own self-tests, `E` + edge input, `ES` both |
| `branch` | `none` (no gate) or `consumer_complete` (all-consumer-assertions gate) |
| `cell`, `rep`, `diff`, `order` | cell id; representation (`full`, `patch`); diff view shown; fragment order (`listed`, `reversed`) |
| `disposition` | `submitted`, `abstained` or `invalid_output` |
| `correct_completion`, `false_acceptance`, `false_rejection` | released and hidden-correct; released and hidden-wrong (a wrong release); blocked although hidden-correct |
| `released`, `gate_blocked`, `model_abstained` | outcome flags |
| `calls`, `input_tokens`, `output_tokens` | model calls and tokens of the cell |
| `step1_candidate`, `step1_hidden_pass` | the first step was a candidate; its hidden verdict |
| `offline_gates` | arm T, no gate: would the first-step candidate be released by `W1`-`W3`, `G1`-`G3`, `self_S`, `self_Sraw` |
| `offline_writer_v2` | hard menu: the same for the local writers (prompt v2) |

`self_tests.jsonl`: `model`, `task_id`, `status`, `parse_error`, `kept`, `dropped`, `kept_legacy_binding`,
`identical_to_hidden` (tests identical to a hidden input), `max_similarity_to_hidden` (per kept test).
`probe.jsonl`: `model`, `task_id`, `runtime`, `release`, `disposition`, `hidden_pass`.

Models: `GPT-5.6-luna` (commercial, reasoning effort high) and the open-weight `Qwen3-8B` (4-bit), `gpt-oss-20b` and
`qwen3.6-35b-a3b`, served locally. No model-written code was executed: repairs are choices among the authors' fragments.

## Not included (available from the authors on request)

Hidden contracts and hidden inputs (the evaluation key), the verbatim text of commercial-model responses, the runtime
and container harness used to execute the procedures, and the post-hoc analyses that re-read hidden contracts (contract
sensitivity: dropping the disputed checks changed 0 verdicts; robustness and writer-by-repairer matrices). The
prior-evidence studies of the appendix and the free-form and frontier-model stages are in the full audit archive.

## Full audit archive

The complete preregistration archive is attached to this repository's GitHub Release as
`accepted-but-wrong-full-artifact.zip`: every design document and addendum with its freeze manifest, the frozen
per-stage code (builders, runners, evaluators, analyses), all scored outputs, `tools/verify_freezes.py` (checks every
freeze manifest) and `tools/reproduce.py` (re-runs the frozen analyses and regenerates all 2,383 numbers of the paper).
Inside that archive, `paper34` and `p34` are the authors' internal project label.

## License

Code: MIT (`LICENSE`). Data and documentation: CC BY 4.0 (`LICENSE-DATA`). Please cite the paper (`CITATION.cff`).
