# English → Chinese Machine Translation Evaluation

**DS602 Evaluating AI Systems — Mini-Project** · SMU

Which translation system should an education-technology company deploy to translate English help-centre, product, technical, customer-service and business content into Simplified Chinese — or is none safe without human post-editing?

## Systems compared

| Key | System | Type |
|---|---|---|
| `gemini` | Google Gemini 3.5 Flash-Lite | Proprietary general-purpose LLM (free tier) |
| `gptoss` | OpenAI gpt-oss-120b, served by Groq | Open-weight LLM (free tier) |
| `nllb` | Meta NLLB-200-distilled-600M | Open-source translation model, self-hosted |
| `copy` | Copy the English source | Floor baseline |

All LLMs get the same fixed prompt and their vendors' default settings.

## Test set

`translation_eval_dataset.csv` has 96 test items (16 in each of six domains: academic, business, technical, customer service, formatting, stress) plus 6 dev items for the pilot. Every item has a reference translation, expected glossary terms and must-keep tokens (numbers, codes, URLs).

SHA-256: `9b467f32f8ebc4c615b22dbf03706a24b91887fb73500cac0fa67293ba00b69a`

## Main results

| System | chrF (95% CI) | Items with all numbers/codes kept | p95 latency | Cost per 1k chars | Passes all thresholds |
|---|---|---|---|---|---|
| Gemini | 60.9 (56.8–65.1) | 95.3% | 6.57 s | $0.0018 | No (latency) |
| **gpt-oss** | 57.1 (53.3–61.2) | 95.3% | **0.73 s** | **$0.0011** | **Yes** |
| NLLB | 34.2 (30.9–38.1) | 81.4% | 14.6 s | $0.0122 | No |
| Copy baseline | 9.2 | — | — | — | — |

- gpt-oss − Gemini chrF: −3.81, 95% CI [−7.26, −0.27], Holm-corrected p = 0.085 (not significant after correction).
- Both LLMs are far better than NLLB (Holm p < 0.001).
- **Recommendation: deploy gpt-oss-120b.** It is the only system that passes every pre-registered threshold, and it is not significantly worse than Gemini on chrF.

Deployment thresholds (fixed before running): failure rate ≤ 2%, numbers/codes kept on ≥ 95% of items, format fidelity ≥ 0.95, p95 latency ≤ 5 s, cost ≤ US$0.05 per 1,000 source characters.

**Caveats.** n = 96 limits statistical power. NLLB ran on CPU, so its latency and cost are overstated. Costs are list prices; the actual spend was US$0 on free tiers. Prices were not re-verified on the run date.

## Files

| File | Content |
|---|---|
| `translation_evaluation.py` | The full pipeline: data checks, running systems, metrics, statistics, charts, deployment decision |
| `translation_project_plan.md` | Project plan: business question, metrics, thresholds, statistics, error-analysis method |
| `translation_eval_dataset.csv` | Frozen test set |
| `results/results_summary.csv` | All metrics per system, with 95% bootstrap CIs |
| `results/results_pairwise_tests.csv` | Paired tests (Wilcoxon, exact McNemar) with Holm correction |
| `results/results_decision.csv` | Threshold checks and the decision |
| `results/throughput.csv` | Requests per second at concurrency 1 and 4 |
| `results/error_cases_for_review.csv` | Automatically flagged outputs for manual error analysis |

## How to run (Google Colab)

1. Runtime → Change runtime type → T4 GPU (for NLLB).
2. Upload `translation_evaluation.py` and `translation_eval_dataset.csv`.
3. Add two Colab Secrets (key icon in the left sidebar) and enable notebook access: `Gemini_API_Key` (Google AI Studio) and `GROQ_API_KEY` (GroqCloud console).

```python
!pip -q install sacrebleu google-genai groq transformers sentencepiece

%run translation_evaluation.py check-data   # validate the dataset
%run translation_evaluation.py pilot        # 6 dev items
%run translation_evaluation.py run          # all 96 test items x all systems (resumable)
%run translation_evaluation.py throughput
%run translation_evaluation.py stability
%run translation_evaluation.py evaluate     # metrics, statistics, charts, decision
```

Add `--dry-run` to any command to test the pipeline with simulated outputs and no API calls. Never commit API keys.
