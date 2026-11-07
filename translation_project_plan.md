# Translation AI Systems Evaluation — Project Plan (v5)

**DS602 Mini-Project · Compare 2+ AI systems on a given task**

## 1. Title and business question

**Gemini vs gpt-oss (with an open-source NLLB translation model and a copy-source baseline): English-to-Simplified-Chinese translation quality, terminology and numeric fidelity, latency, and cost**

**Scenario.** An education-technology company based in Singapore translates public product pages, help-centre articles, technical instructions, customer-service replies, and business notices from English into Simplified Chinese.

**Business question (course Step 2).** *Which translation system should we deploy for this content, or is no system safe to deploy without human post-editing?*

**Business outcomes and why they matter.**

| Business outcome | Why it matters | Online / proxy signal | Offline metric in this project |
|---|---|---|---|
| Correct meaning | A reversed condition, negation, or amount in a notice can create legal, financial, or support costs | Post-editing rate, complaint rate | chrF (main), optional COMET, BLEU |
| Correct numbers, codes, formatting | Wrong prices, dates, commands, or URLs make content unusable | Correction tickets | Automatic protected-token checks (share of items with all numbers/codes/URLs kept) and format checks |
| Consistent terminology | Inconsistent product and policy terms confuse users | Glossary QA failures | Automatic glossary recall |
| Low operating cost | Translation volume grows with the content library | Monthly translation spend | Cost per request and per 1,000 source characters |
| Acceptable responsiveness | On-demand help-centre translation must feel instant | Page-load abandonment | Latency p50/p95/p99; throughput |

Every offline metric here is a proxy for the business outcome. The project uses several complementary automatic metrics, plus manual review of flagged errors (Section 10), rather than relying on a single score.

## 2. Systems compared (course Step 1)

The project compares two general-purpose LLM families from different vendors against a translation-specialised open-source model. It follows the course's translation option (general LLMs vs "a fine-tuned model") and the brief's rule that "any AI system pair with a clear evaluation task is welcome".

| Key | System | Type | Settings |
|---|---|---|---|
| `gemini` | Google Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) | Proprietary general-purpose LLM | Fixed prompt (Section 6); default temperature (Gemini 3 models are designed for it) and default thinking settings |
| `gptoss` | OpenAI gpt-oss-120b, served by Groq (`openai/gpt-oss-120b`) | Open-weight general-purpose LLM (reasoning model) | Fixed prompt; Groq's default temperature and reasoning effort; only the final answer returned; max 2,048 output tokens |
| `nllb` | Meta NLLB-200-distilled-600M | Open-source, translation-specialised model, self-hosted | Beam 4; runs free on a Colab T4 GPU; no API key |
| `copy` | Copy-source baseline | Floor baseline | Output = English source |

- **Why these systems.** The comparison mirrors the realistic options for a team with **no translation budget**:
  1. a big vendor's free API tier (Gemini 3.5 Flash-Lite, Google's lightest model);
  2. a free hosted open-weight LLM (OpenAI's gpt-oss-120b on Groq, which anyone could also self-host);
  3. a self-hosted translation-specialised model (NLLB-600M).

  Each LLM is used with its vendor's default settings.
- **The systems are deliberately not equivalent.** They differ in size (Flash-Lite is Google's smallest tier; gpt-oss-120b has about 117B parameters, about 5B active per token; NLLB has 0.6B), in whether they reason before answering, and in how they are deployed. Conclusions therefore apply to these specific models and access routes, not to "Google vs OpenAI" in general. A like-for-like vendor comparison (e.g. Gemini Flash) would need a paid tier, because Flash's free tier allows only 20 requests per day.
- **Why a baseline.** It shows the floor of every metric. It also exposes metric weaknesses: copying the source keeps about 98% of protected tokens (numbers survive untranslated), so token fidelity is only meaningful alongside chrF.
- **At least two AI systems even without keys.** If one key is unavailable, `nllb` plus the remaining LLM still meets the brief. Choose systems with `--systems`.
- **What to record.** The script logs the exact model names, returned model identifiers, run date, parameters, and prompt in `run_manifest.json`. Model versions change quickly; if a model is retired, set `GEMINI_MODEL` or `GPTOSS_MODEL` to the current equivalent and report this.
- **Consumer chat apps are not used.** Manual copying prevents fair latency measurement and reproducibility.

## 3. Research questions and hypotheses

- **RQ1.** Which system has the best overall translation quality (chrF, and COMET if computed)?
- **RQ2.** Which system best preserves terminology, negation, conditions, numbers, and formatting?
- **RQ3.** Do rankings change by domain, difficulty, or linguistic phenomenon?
- **RQ4.** Which systems satisfy the pre-registered deployment thresholds?
- **RQ5.** Among systems that pass the thresholds, which has the lowest cost per 1,000 characters and an acceptable p95 latency?

**Hypotheses.**
- **H1.** Both LLMs will outperform NLLB on overall quality, especially on the stress, formatting, and terminology-heavy items.
- **H2.** Gemini and gpt-oss-120b will be close in quality. There is no a-priori direction, so the paired tests decide whether any difference is real.
- **H3.** NLLB will have the lowest latency and cost but will be weakest on formatting, code, and terminology, because it was trained on plain sentences.
- **H4.** gpt-oss-120b reasons before answering, so it will use more output tokens per request than Gemini Flash-Lite; Groq's specialised inference hardware may still give it competitive latency. Efficiency may separate the LLMs more than quality does.
- **H5.** Rankings will differ between the technical, formatting, and stress subsets.

## 4. Test set (course Step 3)

**File:** `translation_eval_dataset.csv`. The file is frozen; `check-data` prints its SHA-256, which you record in the report.

| Split | Items | Use |
|---|---|---|
| `test` | 96 (16 each: academic, business, technical, customer_service, formatting, stress) | All results. Every system runs on all 96 items. |
| `dev` | 6 (one per domain) | Credential pilot and throughput test only. Never used in results, and prompts are not tuned on test items. |

**Columns.**

| Column | Content |
|---|---|
| `id` | Item identifier |
| `split` | `test` or `dev` |
| `domain` | One of the six domains |
| `difficulty` | easy / medium / hard |
| `phenomena` | Rule-based tags: `negation`, `condition`, `number`, `format`, `pronoun_reference`, `contrast`, `plain` |
| `source_en` | English source |
| `reference_zh` | One acceptable Simplified Chinese reference |
| `glossary` | Expected terms, `en=zh;…` |
| `protected_zh` | Must-keep tokens, `;`-separated; `|` separates acceptable alternatives, e.g. `120万|1,200,000` |
| `notes` | Item-specific notes |

**Diversity.** The set mixes six domains, lengths from 67 to 151 characters, multi-line and Markdown inputs, code, URLs, emails, currencies, units, dates, and 16 linguistic stress items (negation scope, conditionals, counterfactuals, pronoun reference, cleft sentences). The stress and formatting subsets are this project's robustness evaluation (optional in the course brief).

**Quality control already applied (v1 → v2).**
1. **A10 duplicated A05 word for word.** It is replaced by a new academic item.
2. **F01 stored line breaks as the literal characters `\n`, unlike F11.** It now uses real line breaks, so formatting checks treat both items the same way.
3. **Six glossary entries were not in their own reference (T02, A09, B08, T09, C10, C11).** Two glossary source terms were also not in the source. All are corrected.
4. **The v1 automatic token check marked correct references as wrong on six items.** Causes: unit conversions (`1.2 million` → `120万`), acronyms the reference correctly translates (`US$` → `美元`, `PIN` → `验证码`, `SGT` → `新加坡时间`), and captured trailing commas (`2,`, `4,`). It is replaced by the curated `protected_zh` column with acceptable alternatives.
5. **Reference self-test.** Every reference must score 100% on its own glossary, protected-token, and format checks. `check-data` enforces this, and all 102 items currently pass.

**Still to do before running systems.**
- Two bilingual team members independently review every reference, especially the new A10 and D01–D06. Record changes in `notes`, then re-run `check-data` and freeze the file.
- Acceptable variants that single references cannot capture:
  - F12: the name *Tan* may be rendered 陈, 谭, or kept as Tan.
  - F09: address order.
  - Unit conversions (B03, B13).

  Automatic metrics will under-score such legitimate variants; check these items by hand during error analysis.

**Known limitations of the test set.**
- **Few easy items.** Only 4 test items are `easy`, and all stress items are `hard`, so difficulty comparisons are descriptive only.
- **Small subgroups.** 96 items give moderate statistical power. Each domain subgroup has only n = 16 and is treated as exploratory.
- **Team-authored items.** The items were written by the team, not sampled from production traffic. This is a diagnostic set, not a representative population sample.

## 5. Experimental controls

- Identical source text for every system; English → Simplified Chinese only.
- One request per item, with no conversation history and no vendor glossary or human editing.
- Identical LLM prompt for Gemini and gpt-oss. Both keep their vendor's default temperature and reasoning settings (Google advises against lowering the temperature for Gemini 3 models; Groq recommends 0.5–0.7 for gpt-oss rather than 0). System order is randomised within each item (seed 602).
- Same Colab session and network window for all API calls. Latency is end-to-end from Singapore; state this in the report.
- **Rate limits:** the script leaves 6 s between calls to each LLM API to stay under the free-tier limits. Groq's free tier allows 30 requests, but only 8,000 tokens, per minute and 1,000 requests per day for gpt-oss-120b; with reasoning, one request can use several hundred tokens. Change this with `GEMINI_MIN_INTERVAL_S` / `GROQ_MIN_INTERVAL_S` (0 on a paid tier). Latency is measured after the wait, so pacing does not inflate it. Expect about 10–15 minutes per LLM for the 96 test items.
- **Retry policy:** up to 4 attempts, waiting the delay the API suggests (otherwise exponential backoff with jitter); the number of attempts is recorded. After the final failure the item is kept with an empty translation, so it scores 0 and **stays in the denominator** of the failure rate.
- The `run` mode is resumable. Only the latest attempt per (item, system) is kept.
- **Pilot first, then freeze.** Run `pilot` on the 6 dev items, inspect the outputs, then freeze all settings before running on the test set.
- **LLM stability check (recommended).** Because both LLMs sample at their default temperature, run both LLMs three times on 12 stratified test items (2 per domain) with the `stability` mode, which reports the share of identical outputs and the chrF range. This shows how much of any Gemini–gpt-oss difference could be run-to-run noise.
- Never hard-code or commit API keys. Use Colab Secrets.
- **Data privacy.** On Gemini's free tier, Google may use prompts to improve its products. The test set contains no confidential or personal data, so this is acceptable here, but it would matter for real company content.

## 6. Fixed LLM prompt

```
Translate the English text below into Simplified Chinese.
Requirements:
1. Preserve the complete meaning, including negation and conditions.
2. Preserve numbers, dates, percentages, currencies, units, URLs, email addresses, code, and Markdown structure.
3. Use clear, professional Simplified Chinese.
4. Output only the translation, with no commentary.

TEXT:
{text}
```

**Fairness note.** Both LLMs receive exactly the same prompt. NLLB is a translation model that takes the raw text, so it receives no instructions; this reflects how each system would really be deployed. Discuss it as a limitation.

## 7. Metrics (course Step 2)

### 7.1 Quality metrics (required)

**Automatic metrics** are computed on all 96 items × all systems:

| Metric | What it measures | Why it matters for the business |
|---|---|---|
| **chrF** (character n-gram F-score, `sacrebleu`) — main automatic metric | Overlap with the reference at character level | Works for Chinese without word segmentation. v1 used chrF++; its word n-grams are meaningless for unsegmented Chinese. |
| **COMET** (`Unbabel/wmt22-comet-da`, optional `--comet` flag, GPU recommended) | Neural metric trained on human judgements; credits paraphrases | Correlates better with human quality than chrF/BLEU |
| BLEU (`tokenize="zh"`), corpus and sentence level | Traditional n-gram overlap | Reported for comparability with the literature only |
| Terminology recall | Share of glossary terms present | Terminology consistency |
| Protected-token fidelity; all-tokens-correct rate | Numbers, codes, URLs, and emails kept (unit-conversion alternatives allowed) | Wrong numbers are high-impact errors |
| Format fidelity | Line breaks, bullets, bold, code spans, and table pipes reproduced | Content must remain publishable |
| Automatic error flags | Failed/empty, untranslated, possible omission (length < 0.6 × reference), possible addition (> 1.6×), token error, terminology miss, format broken | Directs the manual error analysis to likely problems |

**Main quality metric:** chrF, or COMET when computed with `--comet` (recommended, because it credits valid paraphrases that a single reference would penalise).

### 7.2 Cost and efficiency (required)

| Measure | Definition |
|---|---|
| Latency | End-to-end p50, p95, and p99 per request (successful requests), plus a box plot |
| Throughput | Requests per second at concurrency 1 and 4 on the dev items (`throughput` mode), plus sequential throughput from the main run |
| Tokens per request | Input and output tokens for LLMs (Gemini thinking tokens included) |
| Cost | Variable cost per request and per 1,000 source characters |
| Failure rate | Failed requests ÷ all attempted items |

**Prices.** The script defaults are the last-known list prices (USD):
- Gemini 3.5 Flash-Lite (paid tier): $0.30 input / $2.50 output per 1M tokens. The output price includes any thinking tokens.
- gpt-oss-120b on Groq (paid tier): $0.15 input / $0.60 output per 1M tokens. The output count includes reasoning tokens.
- Self-hosted NLLB: an assumed GPU cost of $0.35 per hour

**Verify every price on the official pricing pages on the day you run**, override any that changed via environment variables, set `PRICES_VERIFIED_ON`, and keep screenshots. Gemini's free tier changes what you actually pay, but report list-price costs so the systems remain comparable.

**Expected spend for the full experiment:** about 9,000 source characters (96 requests) per system, so well under US$1 at list prices for both LLMs together, including the pilot, throughput, and stability tests. **Actual spend is US$0**: both LLMs run on free tiers and NLLB runs on Colab's free GPU. Report list-price costs anyway, so the comparison reflects what a paid production deployment would cost.

**Free daily quotas (the binding constraint).** Every call counts against the free quota: pilot 6 + main run 96 + throughput 13 + stability 36 ≈ **150 requests per LLM**.

| API | Free requests per day | Enough? |
|---|---|---|
| Gemini 3.5 Flash-Lite | about 500 | Yes, in one day |
| Gemini 3.x Flash (e.g. 3.8 Flash) | **20** | No: 96 items would take 5+ days. This is why Flash-Lite is used. |
| Groq gpt-oss-120b | 1,000 (and 8,000 tokens per minute) | Yes |

Check your actual limits on the AI Studio rate-limit page (aistudio.google.com/rate-limit) and the Groq console. If a daily quota runs out mid-run, the script stops calling that system, records nothing for the unfinished items, and prints a message. Run the same command the next day to resume.

## 8. Deployment thresholds (fixed before results are inspected)

All thresholds are measured automatically by `evaluate` and saved in `results_decision.csv`.

| Threshold | Value | Why |
|---|---|---|
| Failure rate | ≤ 2% of test items | A system that often returns nothing cannot run unattended |
| Numbers, codes, URLs | ≥ 95% of items with such tokens keep **all** of them | Wrong amounts, dates, or commands are high-impact errors |
| Format fidelity | ≥ 0.95 (mean, on items with formatting) | Content must remain publishable |
| p95 latency | ≤ 5 s | On-demand translation must feel responsive |
| Variable cost | ≤ US$0.05 per 1,000 source characters (US$50 per million), set via `BUDGET_USD_PER_1K_CHARS` | Budget |

**Decision rule** (implemented in `deployment_decision()`):
1. Remove every system that fails any threshold.
2. Among the remaining systems, take the one with the highest main quality metric (chrF, or COMET if computed).
3. If a **cheaper** passing system is **not significantly worse** on that metric (Holm-corrected p ≥ 0.05), recommend the cheaper system instead: a quality difference that could be noise does not justify a higher cost.
4. If no system passes, recommend no unattended deployment and keep human post-editing, and state which threshold blocks each system.

**Caution.** Automatic metrics compare against a single reference, so meaning errors that keep the same words (e.g. a lost "not") can still score well. The manual review in Section 10 checks exactly these cases before the recommendation is finalised.

## 9. Statistics (course Step 4: "bonus for statistical rigour")

- **Paired design.** All systems translate the same items, so every comparison is paired, by item.
- **Effect sizes.** Mean difference with a 95% paired-bootstrap confidence interval (5,000 resamples), for chrF, COMET, terminology, and token fidelity.
- **Tests:**
  - Wilcoxon signed-rank for continuous and ordinal scores
  - Exact McNemar for the binary outcome "all numbers/codes/URLs kept"
  - **Holm correction** across the pairwise comparisons of each metric
- Report effect size with its CI, not p-values alone. Per-system means carry 95% bootstrap CIs.
- Domain, difficulty, and phenomenon breakdowns are descriptive and exploratory (n ≤ 16 per cell).

## 10. Error analysis (course Step 4 and 20% of the grade)

1. `evaluate` writes `error_cases_for_review.csv`: every automatically flagged output plus the 5 lowest-chrF outputs per system.
2. The team reviews each case by hand and fills in `primary_error_type`, `business_impact`, `suspected_cause`, and `mitigation`. Error types: `omission`, `mistranslation`, `negation_or_condition`, `number_date_currency_unit`, `terminology`, `entity`, `register`, `format_code_url`, `unsupported_addition`, `other`. Pay special attention to the stress items (negation, conditions): automatic metrics can miss a reversed meaning.
3. Report:
   - error frequency by system (automatic flags and the manually coded types)
   - rates by domain and by phenomenon (negation, condition, number, format)
   - where each system fails differently (for example, NLLB breaking code spans versus an LLM adding words or explanations)
4. Present **3–4 representative cases** with business impact and mitigation (e.g. glossary feature, placeholder protection for code and URLs, human review for texts that contain conditions). Choose them to represent the error distribution, not because they are funny.

## 11. Pipeline and commands (course Step 4)

**Setup (Colab):**
1. Runtime → Change runtime type → T4 GPU (for NLLB).
2. Upload `translation_evaluation.py` and `translation_eval_dataset.csv`.
3. Add two Colab Secrets (key icon in the left sidebar) and enable notebook access for both:
   - `Gemini_API_Key`: from Google AI Studio (aistudio.google.com/apikey); free tier.
   - `GROQ_API_KEY`: from the GroqCloud console (console.groq.com → API Keys → Create API Key); free, no credit card needed. The key is shown only once.

```python
!pip -q install sacrebleu google-genai groq transformers sentencepiece
# optional: !pip -q install unbabel-comet

# 0. Free rehearsal with simulated outputs (files prefixed DRYRUN_; never report these numbers)
%run translation_evaluation.py run --dry-run
%run translation_evaluation.py throughput --dry-run
%run translation_evaluation.py evaluate --dry-run

# 1. Real experiment
%run translation_evaluation.py check-data        # validate + SHA-256
%run translation_evaluation.py pilot             # 6 dev items: inspect outputs, then freeze settings
%run translation_evaluation.py run               # all 96 test items x all systems (resumable)
%run translation_evaluation.py throughput        # req/s at concurrency 1 and 4
%run translation_evaluation.py stability         # 3 repeats x 12 items per LLM: run-to-run variation
%run translation_evaluation.py evaluate --comet  # metrics, stats, charts, error cases, deployment decision
```

Use `--systems` to choose systems, e.g. `--systems gemini,nllb,copy` if the Groq key is not ready yet.

## 12. Deliverables (course "What to submit")

| Course requirement | This project's file(s) |
|---|---|
| **Evaluation code** (required; Colab preferred) | `translation_evaluation.py`, run in a Colab notebook through the commands above; submit the notebook with its outputs |
| **Result table** (required; point estimates, optional CIs) | `results_summary.csv` (with 95% CIs), `results_pairwise_tests.csv`, `results_decision.csv` |
| **Visualisation** (required; at least 1 chart) | `chart_quality_ci.png`, `chart_latency.png`, `chart_domain_heatmap.png`, `chart_pareto.png`, `chart_auto_errors.png` |
| **Presentation** (required; 10–15 min) | Slides following Section 14 |
| Write-up (optional; 1–2 pages) | Executive summary: problem, method, results, recommendation, limitations |
| Supporting evidence | Frozen dataset + SHA-256, `run_manifest.json`, raw `translation_outputs.csv`, `throughput.csv`, `stability_summary.csv`, `results_per_item.csv`, `error_cases_for_review.csv` (coded), `results_breakdowns.txt`, price screenshots |

## 13. How the project meets the grading criteria (5 × 20%)

| Criterion | Evidence |
|---|---|
| **Business framing** | A real deployment decision with pre-registered thresholds and a budget; the "no deployment" outcome is allowed |
| **Metric selection and justification** | Metric → business-reason table (Section 7); chrF instead of chrF++ for Chinese; optional COMET; complementary checks for numbers, format, and terminology |
| **Code quality and reproducibility** | Single documented script, seeds, resumable runs, manifest with versions/prices/prompt, dataset checksum, reference self-test, dry-run mode |
| **Presentation clarity** | Structure matched to the course timing (Section 14); every chart answers one research question |
| **Error-analysis depth** | Automatic flags → manual error coding → domain and phenomenon patterns → cases with impact and mitigation |

## 14. Presentation structure (10–15 minutes + 5 minutes Q&A)

| Time | Content |
|---|---|
| 1–2 min | Problem statement: business scenario, decision, and why errors are costly |
| 1 min | Systems compared, the baseline, and fairness controls |
| 1–2 min | Test set design and metric justification (the table from Section 7) |
| **3–5 min** | **Results and statistics:** quality with CIs, paired tests, latency/cost, Pareto chart, threshold table and decision |
| 2–3 min | Error analysis: error-type chart, phenomenon breakdown, 3 representative cases |
| 1 min | Recommendation, limitations, and next steps |

**Both students must be able to explain every section of the code.** The course brief states that failing to do so means failing the project.

## 15. Work allocation

- **Student A:** API accounts and Secrets, pilot, running systems, logging, latency, throughput and cost, reproducibility check.
- **Student B:** reference review, freezing the dataset, error-case coding, charts.
- **Both:** statistics, error analysis, slides. Each student re-runs the other's part from the raw files before submission.

## 16. Schedule

| Day | Tasks |
|---|---|
| 1 | Review references and freeze the dataset (`check-data`); create API accounts; dry run |
| 2 | Pilot on the dev items; verify prices; freeze settings; run all systems; throughput |
| 3 | Stability check; `evaluate`; code the error cases |
| 4 | Charts and slides; limitations; write-up |
| 5 | Rehearse (time it); each student explains the other's code; final reproducibility run |

## 17. Reproducibility checklist

- [ ] Dataset frozen; SHA-256 recorded (`check-data`)
- [ ] `run_manifest.json` saved (UTC run date, Python and package versions, models, parameters, prompt, prices, thresholds)
- [ ] Prices verified on the run date (`PRICES_VERIFIED_ON` set; screenshots kept)
- [ ] Raw outputs, statuses, attempts, latencies, and timestamps kept; failed requests stay in the denominator
- [ ] No credentials in any submitted file or notebook output
- [ ] Analysis re-run from the raw files (`evaluate`) by the other student
- [ ] Dry-run files (`DRYRUN_*`) deleted or excluded from the submission

## 18. Limitations

- **Single references and no human scoring.** A single reference penalises legitimate paraphrases, and string-based metrics can miss meaning errors such as a dropped negation. COMET and the manual review of flagged cases mitigate this but do not replace a full human evaluation.
- **Test set.** It is team-authored and diagnostic, with unbalanced difficulty, so results may not generalise to production traffic or to other language directions.
- **Change over time.** API models, prices, and latency change; network conditions from Singapore affect latency.
- **Default settings.** Both LLMs run at their vendors' default temperature and reasoning settings rather than identical settings; the stability check quantifies run-to-run variation.
- **Free-tier conditions.** Free tiers may be slower or more rate-limited than paid tiers, so measured latency and throughput may understate paid performance. On Google's free tier, prompts may be used to improve Google's products (acceptable here because the test set contains no confidential data).
- **Unequal model sizes.** The results compare two specific LLMs of different size classes (Google's lightest tier vs a large open-weight model) on one date, so they say nothing about the vendors in general.

## 19. Change log

### v4 → v5
| Area | v4 | v5 |
|---|---|---|
| Quality evaluation | Blind human rating by two raters (126 outputs each) as the primary outcome | **Removed.** Automatic metrics only (chrF, optional COMET, token/format/terminology checks), plus manual review of flagged error cases |
| Deployment thresholds | Human adequacy, numeric fidelity, critical-error rate + latency + cost | Failure rate, numbers/codes kept, format fidelity, latency, cost (all automatic) |
| Decision rule | Cheapest system passing all thresholds | Best quality among passing systems, unless a cheaper passing system is not significantly worse |
| Code | `human` mode, rating sheets, blinding key | Removed; `evaluate` now writes `results_decision.csv` and prints the recommendation |

### v3 → v4
| Area | v3 | v4 |
|---|---|---|
| Systems | Gemini 3.8 Flash, Claude Haiku 4.5, NLLB, copy | Gemini 3.5 Flash-Lite, **gpt-oss-120b (via Groq)**, NLLB, copy |
| Cost to run | Claude needed prepaid credits | Everything runs on free tiers (US$0) |
| Credentials | `Gemini_API_Key`, `ANTHROPIC_API_KEY` | `Gemini_API_Key`, `GROQ_API_KEY` |
| Gemini model | Gemini 3.8 Flash | **Gemini 3.5 Flash-Lite**: 3.8 Flash's free tier allows only 20 requests per day |
| Code | Gemini-only pacing | Shared free-tier pacing for both LLM APIs; honours the API's suggested retry delay; stops cleanly (and resumes later) when a daily quota runs out |

### v2 → v3
| Area | v2 | v3 |
|---|---|---|
| Systems | Google Cloud Translation, DeepL, GPT-4o, NLLB, copy | **Gemini 3.8 Flash, Claude Haiku 4.5**, NLLB, copy |
| Credentials | Google Cloud project + DeepL + OpenAI keys | Only `Gemini_API_Key` and `ANTHROPIC_API_KEY` |
| Code | Five system callers | Three callers + baseline; free-tier pacing for Gemini; per-system settings logged in the manifest |
| Human evaluation | 168 outputs per rater | 126 outputs per rater (3 AI systems) |
| Hypotheses | MT APIs vs LLM | LLM vs LLM, and LLMs vs a specialised MT model |

### v1 → v2

| Area | v1 | v2 |
|---|---|---|
| Test set size | Plan said 36 items; CSV had 96 | 96 test + 6 separate dev items |
| Data errors | Duplicate item, literal `\n`, 8 glossary errors | Fixed; the reference self-test now passes on all items |
| Systems | 3 APIs, all requiring keys | + NLLB (free open-source model) + copy baseline; optional Gemini/Claude substitutes |
| Token check | Regex marked correct references as wrong | Curated `protected_zh` with alternatives; NFKC normalisation |
| chrF | chrF++ (word n-grams) | chrF (character level, suited to Chinese) + optional COMET |
| Efficiency metrics | p50/p95 only | p50/p95/p99, throughput, tokens per request, billed characters, failure rate |
| Failures | `evaluate` crashed if any request failed | Failures kept and scored 0; failure rate reported |
| Credentials | All three keys needed even to run one system | Per-system lazy clients, a preflight check, Colab Secrets support |
| Statistics | Bootstrap only | + Wilcoxon, McNemar, Holm correction, Clopper–Pearson bound |
| Human evaluation | One sheet covering all 96 items | Two independent rater sheets on 42 stratified items, with agreement statistics, reconciliation rule, threshold decision, and cost per acceptable translation |
| Error analysis | Crosstab only | Automatic flags, a review file with impact/cause/mitigation, and domain and phenomenon breakdowns |
| Prices | Defaulted to 0, so cost was silently 0 | Documented defaults + verification flag |
