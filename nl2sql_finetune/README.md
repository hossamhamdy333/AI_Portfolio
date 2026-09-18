<div align="center">

# Fine-Tuning Qwen2.5-Coder for Text-to-SQL

`transformers` `peft` `bitsandbytes` `trl` `torch` `datasets` `pandas` `scikit-learn` `sqlite3` `matplotlib` `Qwen/Qwen2.5-Coder-1.5B-Instruct`

</div>

---

### Contents

- [Summary](#summary)
- [Problem & motivation](#problem--motivation)
- [Approach](#approach)
- [Data](#data)
- [Results](#results)
- [What I'd do differently / limitations](#what-id-do-differently--limitations)
- [Stack](#stack)

---

## Summary

QLoRA fine-tunes `Qwen/Qwen2.5-Coder-1.5B-Instruct` on schema-constrained
text-to-SQL generation (`b-mc2/sql-create-context`), entirely on a free-tier
16GB T4. The headline numbers, recomputed directly from the two committed
results files rather than just read off the README: exact-match accuracy
goes from 5.00% (zero-shot base) to 81.25% (fine-tuned), and execution-based
valid-SQL rate goes from 92.00% to 97.25%. That first number is real but
misleading on its own. Recomputing it under a quote-character-normalized
scoring function (both notebooks' scoring functions currently leave ' vs "
differences uncorrected, despite claiming otherwise) puts the *base* model's
true semantic accuracy closer to 43%, not 5% — the base model already gets
the SQL logic right most of the time, it just prefers single quotes where
the dataset's gold answers use double quotes. The fine-tuned model's real
accuracy barely moves under the same correction (81.5%, from 81.25%). So the
genuine, fair improvement from fine-tuning is closer to +38 percentage
points than the +76 the raw metric reports — still a large, real result, but
about half the size of the headline, and for a more interesting reason:
most of what fine-tuning bought here is the model learning the dataset's
specific quoting *convention*, on top of a base model that was already
closer to correct than the raw exact-match score gave it credit for.

## Problem & motivation

Text-to-SQL is naturally schema-constrained: the model has to translate a
question into a query using only the tables and columns actually in front
of it, not whatever's statistically likely from pretraining. A naive
fine-tune assumes more training data and more LoRA capacity can only help,
and assumes the eval script that reports "exact match" is actually doing
what its name says. Both assumptions are worth checking directly rather
than trusting.

On the first: the original iteration of this project (3,000 training rows,
LoRA rank 16, per the author's own account — see the note on what's
independently verifiable below) reportedly improved exact match but
regressed valid-SQL rate, with failures traced to the model hallucinating
entirely wrong tables/columns on schemas that didn't closely resemble
anything memorized from the small training slice. `b-mc2/sql-create-context`
makes that failure mode easy to fall into: 92.8% of its 78,577 rows have a
schema string unique to that row (verified directly — see Data), largely
because table names follow a near-unique `table_name_<N>` convention. A
model trained on too small and too repetitive a slice of that schema space
has every incentive to memorize specific (schema, answer) pairs rather than
learn to read an arbitrary schema. The fix that's actually committed here —
12,000 rows instead of 3,000, LoRA rank 32/alpha 64 instead of 16/32, and an
explicit "use only the table and column names given in the schema, never
reference any other table" clause folded into the system prompt — targets
exactly that.

On the second: I didn't take the "exact match" metric at face value.
Re-deriving it from the two committed result parquet files turned up a real
scoring inconsistency between the baseline and fine-tuned evaluation
notebooks (see Approach and Results) that changes how the headline number
should be read, without changing that fine-tuning genuinely helped.

## Approach

### QLoRA config

(`configs/config.yaml`, loaded by all four notebooks so
hyperparameters live in one place, not copy-pasted per notebook): 4-bit
NF4 quantization with double quantization and bf16 compute dtype
(`BitsAndBytesConfig`), LoRA rank 32, alpha 64, dropout 0.1, applied to the
four attention projection matrices only (`q_proj`, `k_proj`, `v_proj`,
`o_proj`) — no MLP modules (`gate_proj`/`up_proj`/`down_proj`), unlike this
portfolio's other LoRA fine-tune (`allam_finetune`), which found MLP
capacity mattered for a harder generative task. Confirmed from the
committed adapter's own `adapter_config.json`: 8,716,288 trainable
parameters out of 1,552,430,592 total — 0.5615% of the model.

### Training

(`03-qlora-fine-tuning-sql.ipynb`, TRL's `SFTTrainer` /
`SFTConfig`): 2 epochs, per-device batch size 4 with 4
gradient-accumulation steps (effective batch 16), learning rate 2e-4,
`max_seq_length=512`, 3% warmup. Training examples use Qwen's chat template
(`<|im_start|>system/user/assistant`) with the schema-adherence instruction
in the system turn and the gold SQL as the assistant turn, so the model
sees the exact same prompt shape at train and inference time. Verified from
the notebook's own saved output: 1,500 total steps (12,000 rows ÷ effective
batch 16 × 2 epochs), training loss fell from 3.08 at step 10 to a
0.44–0.53 plateau by step ~500, ending at 0.4563 at step 1500, over 4 hours
13 minutes on a T4. (The trainer's reported "final train loss" of 0.5266 is
the running average across all 1,500 logged steps, not the loss at the
final step — a normal distinction, but the two numbers aren't
interchangeable.)

**Evaluation, and a real scoring inconsistency between the two eval
notebooks**: both `02-baseline-eval-sql.ipynb` (zero-shot base model) and
`04-finetuned-eval-sql.ipynb` (adapter loaded via `PeftModel.from_pretrained`
on top of the same base) run the same 400-row validation split through
`exact_match` (does the normalized generated SQL string equal the
normalized gold string) and `is_valid_sql` (does the generated SQL actually
parse and execute against an in-memory SQLite instance built from the row's
own `CREATE TABLE` statement — an outcome check, not a string comparison,
so it's unaffected by anything below). The two notebooks' `normalize_sql`
functions are not the same function, though: both lowercase and collapse
whitespace, but only notebook 04's version additionally strips double
quotes around purely-numeric literals (`"2005"` → `2005`). Neither version
touches the far more common case — a generated query using `'text'` where
gold uses `"text"` for an identical string value. I recomputed both
notebooks' scores using notebook 04's actual function applied consistently
to both result sets: baseline moves from 5.00% to 5.25% (the numeric-quote
fix barely matters, because it isn't the dominant mismatch pattern), so the
5.00%-vs-81.25% headline comparison is close to apples-to-apples on this
specific axis — but neither script corrects for the ' vs " pattern that
actually drives most of the gap (see Results).

### Post-processing

: generated output is extracted from markdown code
fences if present, and the two notebooks each implement their own
extraction function (`extract_sql()` in 02, `clean_generated_sql()` in
04) rather than sharing one — similar logic, written twice, a second
instance of the same "eval code isn't shared across notebooks" pattern as
the scoring-function difference above.

## Data

`b-mc2/sql-create-context` — verified by loading it directly: 78,577
(question, schema, SQL) triplets, no missing values, zero exact-duplicate
rows on (question, context, answer). 92.8% of schemas (72,947/78,577) are
unique to their row — the memorization risk described above. Query
complexity, computed on the full raw dataset: 96.3% of gold queries contain
a `WHERE` clause, 29.1% an aggregate (`COUNT`/`SUM`/`AVG`/`MIN`/`MAX`), but
only 2.3% a `JOIN`, 1.5% a `GROUP BY`, and 1.4% an `ORDER BY`. In practice
this dataset — and by extension this whole eval — is dominated by
single-table, filter-only queries; there's little multi-table or grouped
logic for either training or evaluation to exercise. Token length on a
2,000-row sample of the actual training prompt template: 95th percentile
117 tokens, max 209 — comfortably inside `max_seq_length=512`, so
truncation isn't a factor anywhere in this run.

12,400 rows (`train_size` 12,000 + `val_size` 400, both from
`config.yaml`) were sampled with `random_state=42`, then
`train_test_split` (also seeded 42) produced 12,000 train / 400 validation
rows — verified directly from the committed `data/train.parquet` (12,000
rows) and `data/val.parquet` (400 rows), both with columns `question`,
`context`, `answer`. Train and validation mean answer length (10.59 vs.
10.88 words) are close enough that the split doesn't look skewed on that
axis.

### What isn't verifiable from the repo as committed

: the "first iteration
used 3,000 rows and rank 16" part of the story is the author's own account
— no round-1 config, data split, or results file is checked in, only the
final (12,000-row, rank-32) state that `configs/config.yaml` and
`data/train.parquet` currently reflect. I can't independently confirm the
round-1 numbers the way I could confirm everything else in this README.

## Results

Recomputed directly from the two committed result files
(`outputs/results/baseline_results.parquet`,
`outputs/results/finetuned_results.parquet`), not just read from a prior
write-up:

| Metric | Base (zero-shot) | Fine-tuned (QLoRA) | Delta |
|---|---:|---:|---:|
| Exact match (as scored by the repo's own code) | 5.00% | 81.25% | +76.25 pp |
| Valid SQL rate (execution-based) | 92.00% | 97.25% | +5.25 pp |

Valid SQL rate is trustworthy as-is — it's checked by actually executing
the generated query, not by comparing strings, so nothing about the
normalization issue below touches it.

**The exact-match number needs a caveat the repo's own code doesn't
apply.** Of the base model's 380 exact-match misses, 152 (40%) differ from
gold *only* in whether the string literal is single- or double-quoted —
e.g. gold `WHERE result = "113-106"` vs. generated `WHERE result =
'113-106'`, or gold `"East Germany"` vs. generated `'East Germany'`. These
are semantically identical queries; neither notebook's `normalize_sql`
treats them as equal. Scoring the base model's outputs under a
quote-character-normalized comparison instead:

| Metric (quote-character-normalized) | Base (zero-shot) | Fine-tuned |
|---|---:|---:|
| Exact match | ~43.0% (172/400) | ~81.5% (326/400) |
| Corrected delta | | **+38.5 pp** |

The fine-tuned model's own score barely moves under the same correction
(325 → 326 of 400) — of its 75 remaining misses, only 1 was a
quote-character-only difference, because fine-tuning taught it the
dataset's actual convention (double quotes for string literals). So the
real story isn't "the base model can barely write SQL and fine-tuning fixed
that" — it's "the base model was already getting the logic right close to
half the time, and a large share of what fine-tuning bought is learning to
match this dataset's specific formatting convention on top of that,"
alongside a genuine, still-substantial ~38.5-point improvement in outputs
that are correct by both string and execution criteria.

## What I'd do differently / limitations

- **The exact-match metric's claimed normalization doesn't match what
  either notebook's code actually does.** A prior write-up of this project
  described the metric as normalizing "case, whitespace, numeric-literal
  quote style" — the code only ever normalizes case and whitespace
  (both notebooks) plus, in notebook 04 only, double-quoted numeric
  literals. Neither version normalizes the single-vs-double-quote pattern
  that drives 40% of the base model's misses. This is the single biggest
  thing I'd fix before trusting this eval further: one shared, correct
  `normalize_sql` function, imported by both notebooks instead of
  redefined in each, that actually does what it claims.
- **The two eval notebooks each define their own scoring and
  post-processing functions instead of sharing one module.** In this run
  the two `normalize_sql` versions happened to produce nearly identical
  baseline scores (5.00% vs. 5.25%), so the practical damage was small —
  but that's luck, not a property of the setup, and the two separately
  written extraction functions (`extract_sql` vs. `clean_generated_sql`)
  are the same risk in a different place.
- **LoRA targets attention projections only (`q/k/v/o_proj`), not the MLP
  layers.** This portfolio's other LoRA fine-tune (`allam_finetune`) found
  MLP capacity mattered for its (harder, generative) task; whether adding
  `gate/up/down_proj` here would close more of the remaining exact-match
  gap, or just add trainable parameters for no benefit on a task this
  narrow, isn't tested.
- **Both training and evaluation are dominated by single-table, filter-only
  SQL.** Only 2.3% of the full dataset's gold queries contain a `JOIN` and
  1.5% a `GROUP BY`. The reported gains say very little about whether this
  model — before or after fine-tuning — can reliably generate correct
  multi-table or grouped queries, since there's barely enough of either in
  the data to learn from or be scored on.
- **The round-1 (3,000-row, rank-16) regression story isn't independently
  verifiable from this repo.** Only the final, already-fixed configuration
  is checked in; I'm reporting the "why it took two iterations" narrative
  as the author's account, not as something I could re-derive from
  committed artifacts the way I could the round-2 numbers.
- **No comparison against a larger base model's zero-shot performance**
  (e.g., a 7B Qwen2.5-Coder variant). It's an open question how much of
  this 1.5B model's ~38.5-point real improvement is a schema-formatting/
  instruction-following gap that a bigger base model might already close
  for free, versus a genuine SQL-reasoning gain specific to this fine-tune.
- **Single training run, single seed, single train/val split.** There's no
  variance estimate — repeating the fine-tune with a different seed could
  move either headline number by an unmeasured amount.
- **The dataset's own labels are used uncritically.** `sql-create-context`
  is a synthetically assembled dataset (schema/question/SQL triplets, not
  hand-verified by a human against each schema); nothing here checks
  whether a nonzero fraction of "gold" answers are themselves wrong.

## Stack

- `transformers`, `accelerate` for model loading and generation
- `peft` (`LoraConfig`, `get_peft_model`, `PeftModel`) for QLoRA
- `bitsandbytes` (`BitsAndBytesConfig`, 4-bit NF4, double quantization) for
  4-bit base-model loading
- `trl` (`SFTTrainer`, `SFTConfig`), pinned `<0.15.0`
- `torch` (bf16 compute), `torchao` (used at fine-tuned-eval time)
- `datasets` (Hugging Face) for `b-mc2/sql-create-context`
- `pandas` / `pyarrow` for parquet-based train/val/results storage — no
  MLflow or W&B server needed to reproduce or inspect results
- `scikit-learn` (`train_test_split`) for the seeded train/val split
- `sqlite3` (stdlib) for the execution-based valid-SQL check
- `matplotlib` for EDA plots
- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Runs on a single free-tier T4 (Kaggle); each notebook pushes its outputs
  back to the GitHub repo directly from the notebook via a personal access
  token
