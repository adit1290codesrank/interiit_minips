# Feasibility and cost audit of every pipeline and experiment in "Sarvam mini PS research.md"

Scope: compute, money, wall-clock time and human effort for each proposal in Parts B, C, D and E of `/home/user/interiit_minips/reports/Sarvam mini PS research.md`, checked against the resources in the main PS (`/home/user/interiit_minips/Sarvam.pdf`) and the mini PS. Date of audit: 14 Oct 2026.

Conventions used throughout:
- **[A]** = assumption made for this estimate (no source). **[3P]** = price or figure from a third party, not from the vendor's own page. **[PS]** = stated in the problem statement PDF.
- modal.com, docs.sarvam.ai, sarvam.ai, arxiv.org, huggingface.co and dl.fbaipublicfiles.com could not be fetched from this sandbox (DNS failure or proxy 403). Every vendor price below therefore comes from **search-engine extracts of the vendor page** or from third parties, and must be confirmed in the Modal and Sarvam dashboards. No local benchmark could be run (no torch installed, model weights unreachable).
- Exchange rate: **₹86–96.5 per USD** (see Q3). Modal H100: **$3.95/h** (see Q6).

### Shared workload model (used by Q1, Q2, Q3, Q5)

| Quantity | Low | Central | High | Basis |
|---|---|---|---|---|
| Hindi speech rate inside trimmed speech spans | 120 wpm | 140 wpm | 160 wpm | [A] — no Hindi conversational rate study found (see Q1 Gaps) |
| Hindi words in 200 h | 1.44 M | 1.68 M | 1.92 M | 200 h × 60 × wpm |
| Sentence length (Hindi words) | 15 | 12.5 | 10 | brief's 10–15 |
| Sentences in 200 h | ~96 k | **~134 k** | ~192 k | words ÷ length |
| English words / Hindi words | 0.8 | 0.9 | 1.1 | [A] (Hindi splits postpositions and auxiliaries into separate words) |
| English characters per word (incl. space, punctuation) | 5.5 | 5.7 | 6.0 | [A] |
| English characters per source hour | 31.7 k | **43.1 k** | 63.4 k | product of the above |
| English audio per source hour at 150 wpm Bulbul | — | 0.84 h | — | [A] 150 wpm English; ≈168 h of English audio per full rendering |

The real dataset is ≤200 h ([PS §3](file:///home/user/interiit_minips/Sarvam.pdf)); the report's plan is 125 h real + 65 h synthetic + 10 h reserve, so "200 h" is an upper bound for every figure below.

---

## Q1. B3 placement engine: is Hibiki-style MT log-likelihood gain over source prefixes "computationally prohibitive"?

### Takeaway
No. Because one teacher-forced pass over a source prefix scores *every* target token at once, the cost is ≈(|x|+1) passes per sentence, about **1.8 M passes / ~80 M tokens for 200 h**, which is **≈0.5–3 H100-hours (≈$2–11) with a 3B encoder-decoder (MADLAD-3B)** and ≈0.15–1 h with IndicTrans2-1B. Even the critique's own over-count (a pass per prefix *per English word*) is only ≈7–41 H100-h (≈$27–161), inside the $1,000 Modal budget. The critique is wrong by roughly 15× in method and by two orders of magnitude in conclusion.

### Cited Findings
- Hibiki computes contextual alignment by giving MADLAD-3B "progressively longer truncated source input, up to the i-th word", and defines the alignment of a target word as the source position where its log-likelihood jumps most — [Hibiki paper (search extract of HTML)](https://arxiv.org/html/2502.03382).
- Hibiki applies the resulting alignment to synthetic data "either by inserting silences or by synthesizing targets with a voice controlled, alignment-aware TTS" — [hibiki README](https://github.com/kyutai-labs/hibiki).
- An unofficial review notes that Hibiki uses the same MADLAD model to generate targets and to measure alignment, and never validates the alignments against human ones — [Pith review](https://pith.science/paper/2502.03382) (third-party critique, not peer review).
- MADLAD-400-3B-MT is described as a T5 v1.1-style encoder-decoder of ~3B parameters, 32 encoder + 32 decoder layers, d_model 1024, d_ff 8192, Apache 2.0 — [aiindigo review](https://aiindigo.com/blog/comprehensive-review-madlad400-3b-mt-in-2026) [3P]; no vLLM or CTranslate2 throughput figure for MADLAD was found in the same search.
- vLLM on one H100 80 GB with Llama 3.1 8B BF16 is reported at ≈12,500 output tok/s — [Morph vLLM benchmarks](https://www.morphllm.com/vllm-benchmarks) [3P]; a TensorRT-LLM table for Llama 3.1 8B on H100 ranges ~27,000 tok/s (128 in/128 out) to ~3,095 tok/s (2048 in/128 out) — [E2E Networks benchmarks](https://docs.e2enetworks.com/docs/tir/benchmarks/inference_benchmarks/) [3P]. These are *generation* figures; teacher-forced scoring is prefill-only and runs faster per token.
- The report itself already scopes the cost-free version: its "cheapest confirming experiment" for B3 is "No GPU, about one day" on 50 sentences + six clips — [report B3](file:///home/user/interiit_minips/reports/Sarvam%20mini%20PS%20research.md).

### Inferences
**Pass count (correct accounting).** For a sentence with |x| Hindi words and |y| English tokens, run prefixes x≤0 … x≤|x| = |x|+1 encoder passes, each with one teacher-forced decoder pass over the full y; the decoder's per-position log-probs give log p(yᵢ | x≤ⱼ, y<ᵢ) for all i at once. Central: 134 k sentences × 13.5 passes = **1.81 M passes**. Tokens per pass ≈ 25 Hindi subword tokens (prefix, ≤ full sentence) + 20 English tokens = 45 [A] → **≈81 M tokens**. Prefixes of one sentence are batched together, and sentences are bucketed by length, so batches of 256–1,024 sequences are easy.

**FLOPs and time.** Forward cost ≈ 2 × params × tokens [A, standard approximation]. H100 dense BF16 peak ≈ 989 TFLOP/s [A, NVIDIA spec sheet, not fetched]; assumed achieved 50–300 TFLOP/s (5–30% utilisation: HF `transformers` with padding is at the low end).

| Scorer | Tokens | FLOPs | H100-h at 300 / 100 / 50 TFLOP/s | $ at $3.95/h |
|---|---|---|---|---|
| MADLAD-3B (Hibiki's choice) | 81 M | 4.9e17 | 0.45 / 1.4 / 2.7 | **$1.8–10.7** |
| IndicTrans2 1B | 81 M | 1.6e17 | 0.15 / 0.45 / 0.9 | $0.6–3.6 |
| 7B decoder-only LLM (prompt + prefix + target ≈ 90 tokens/pass) via `prompt_logprobs` | 163 M | 2.3e18 | 2.1 / 6.3 / 12.7 | $8–50 |
| **Critique's accounting**: one pass per prefix *per English word* (×15), MADLAD-3B | 1.22 B | 7.3e18 | 6.8 / — / 40.7 | **$27–161** |

Sanity check in tokens/s: at 40% utilisation, compute-bound prefill gives ≈66 k tok/s for 3B and ≈28 k tok/s for 7B, consistent with the order of magnitude of the third-party generation benchmarks above (which are memory-bound and slower). So 81 M tokens ≈ 20 min on one H100 in the ideal case; budget **1–3 H100-h** including data loading, padding waste and a second English version (`en_alt`, ×2 if both are scored → still ≤ $25).

**Multipliers that could matter, and do not change the verdict:** scoring both `en_text` and `en_alt` (×2); re-scoring after retranslation (×1.2) [A]; using a 7B LLM teacher instead (×4–5). Worst plausible: ~$100.

**CPU-only option.** IndicTrans2-200M on a 16-core CPU at an assumed 30–60 passes/s [A] → 1.81 M passes ≈ 8–17 CPU-hours: feasible on a laptop overnight for the calibration subset, not needed for the full set since a GPU run is ~$5.

**Cheaper fallback (MMS_FA + awesome-align/SimAlign + rule overrides).** awesome-align/SimAlign use an mBERT-class encoder (~180M params [A]); one pass per side per sentence: 134 k × 2 × ~30 tokens ≈ 8 M tokens → ~3e15 FLOPs → **minutes on a GPU, ~1–3 CPU-hours** [A]. MMS_FA on the Hindi audio is needed anyway for the dataset's Hindi word timings (see Q5), so it is not an extra cost of B3. The fallback is therefore ~free, but the MT-gain term only costs ~$2–11 more and is computed once, offline, from text alone — it can even be run **before the mid-term**, since it needs only Hindi transcripts and English translations, not Bulbul audio.

**Where the critique has a point.** The cost is not compute but engineering: getting MADLAD/IndicTrans2 prefix scoring right (tokenisation of truncated Devanagari prefixes, the `<2en>` target tag, word↔subword mapping for the "log-likelihood jump") is perhaps 1–2 days of work [A], and the method's validity on Hindi is untested (the Pith critique). Both are reasons to run the report's 50-sentence hand-marked comparison first, not reasons it is unaffordable.

### Gaps
- No measured MADLAD-3B or IndicTrans2-1B scoring throughput on H100 was found; the estimate is FLOP-based. vLLM support for T5-style encoder-decoders was not verified, so assume plain HF batching.
- No source gives Hindi conversational speaking rate in words per minute (a 2024 Speech Prosody paper measures L2 Hindi syllable rates but its numbers were not in the extract — [Gauhati / Speech Prosody 2024](https://gauhati.ac.in/?p=20274)); the 120–160 wpm range is an assumption. English conversational rate ≈ 183 wpm is quoted secondhand — [AUDITORY list](https://auditory.org/postings/2016/335.html).
- Hibiki's exact prompt format for truncated prefixes could not be read from the paper (arxiv blocked).

---

## Q2. B7 counterfactual continuations: calibration subset vs. full dataset

### Takeaway
On the report's calibration subset (6 clips + 100 sentences ≈ 150 sentences) B7 costs **minutes of H100 time (<$1)**; on a 2,000-sentence calibration set ≈ 1 H100-h (≈$2–6). On the full dataset it is **≈33.5 M continuations / ≈1.8 B generated tokens ≈ 41–102 H100-h ≈ $160–400** — feasible on paper but 16–40% of the whole Modal budget, and the report's own text understates it ("millions of generations"; it is tens of millions). The report's plan (calibrate on a subset, fit a cheap per-word risk model for the rest) is the right one.

### Cited Findings
- vLLM, one H100, Llama 3.1 8B BF16 ≈ 12,500 tok/s output — [Morph](https://www.morphllm.com/vllm-benchmarks) [3P]; long-input throughput can drop to ~3 k tok/s — [E2E Networks](https://docs.e2enetworks.com/docs/tir/benchmarks/inference_benchmarks/) [3P, TensorRT-LLM, not vLLM].
- The report's B7: M = 20 continuations per prefix; success threshold ≥30% fewer premature commits at matched LAAL; it already warns "20 continuations × 15 prefixes × tens of thousands of sentences is millions of generations, so run it on a calibration subset" — [report B7](file:///home/user/interiit_minips/reports/Sarvam%20mini%20PS%20research.md).
- Sarvam chat LLM rate limits: default models 60/200/1,000 req/min (Starter/Pro/Business); sarvam-30b and sarvam-105b 40/60/120 req/min — [Sarvam Credits & Rate Limits (search extract)](https://docs.sarvam.ai/api/getting-started/ratelimits).
- Sarvam-30B: ₹2.5 input / ₹10 output per 1 M tokens; Sarvam-105B ₹4 / ₹16; another docs page lists sarvam-105b-conversations at ₹29.28 / ₹73.2; Sarvam-M listed as "Free" on one page — [Sarvam pricing (search extract)](https://docs.sarvam.ai/api/getting-started/pricing.md) (pages conflict).

### Inferences
Per (prefix, continuation): generate the rest of the Hindi (~8 words × ~3 tokens/word in Devanagari for a Llama/Qwen-class tokenizer [A] ≈ 25 tokens) plus its English translation (~20–30 tokens) → **≈55 output tokens** [A]. Prompt ≈ 200 tokens, shared by all 20 samples of a prefix (use `n=20` sampling with prefix caching, so the prompt is prefilled once per prefix, not 20 times).

| Scope | Sentences | Prefixes (×12.5) | Continuations (×20) | Output tokens | H100-h at 12.5 k / 5 k tok/s | $ |
|---|---|---|---|---|---|---|
| Report's calibration (6 clips + 100 sents) | ~150 | ~1.9 k | 37.5 k | 2.1 M | 0.05 / 0.1 | **< $1** |
| Larger calibration | 2,000 | 25 k | 500 k | 27.5 M | 0.6 / 1.5 | $2–6 |
| Full 200 h | ~134 k | 1.68 M | **33.5 M** | **1.84 B** | 41 / 102 | **$160–400** |

Add: lemma/embedding agreement scoring (a small sentence-embedding model over 33.5 M short strings ≈ a few H100-h [A]) and alignment of each hypothetical translation (awesome-align, cheap). Use 8B-class models; a 30B+ model would multiply the full-dataset cost 3–4× and push it past the budget.

**Via Sarvam's API instead:** the calibration subset is 1.9 k requests if all 20 samples are requested in one call (≈ 47 min at 40 RPM for sarvam-30b) and costs ≈ ₹40 (7.5 M in + 2.1 M out tokens at ₹2.5/₹10). The full dataset is **infeasible via API** on rate limits alone: 1.68 M requests at 40–120 RPM is 10–29 days of continuous calls, and 33.5 M single-sample calls would take ~1–2 years.

**Recommendation that follows:** run B7 on 2,000 stratified sentences (≈$5), fit a per-word overturn-risk model on cheap features (POS, position relative to the verb group, entity/negation/repair flags, MT-gain), and apply it to the rest. Full-dataset B7 should only be bought after the mid-term and only if the calibration result clears the ≥30% threshold.

### Gaps
- Devanagari tokens per word for the candidate LLM (Qwen, Llama, Gemma, Sarvam-30B) were not measured; this drives the full-dataset cost roughly linearly.
- No source gives vLLM throughput with `n=20` parallel sampling for short outputs; 12.5 k tok/s is a generic figure.

---

## Q3. Sarvam API: Bulbul v3 English synthesis for all training pairs, ASR round-trips, translation and rate limits

### Takeaway
At the documented **₹30 per 10,000 characters (beta) for bulbul:v3**, one full English rendering of 200 h of source costs about **₹26 k ≈ $270–300** (range $200–440). USD 1,000 of Sarvam credit (≈ ₹86–96.5 k, ≈ 29–32 M characters) therefore pays for **about two full renderings plus all the Hindi-side Sarvam usage**, not for "multiple paces/temperatures" across the whole set. The binding constraint is **time, not money**: Starter's bulbul:v3 limit of 30 requests/min makes one sentence-level pass ≈ 74 h of continuous calls. Saaras ASR (₹30/h) and Mayura/Sarvam-Translate (₹20/10 K chars) are cheap.

### Cited Findings
- **Bulbul v3: ₹30 per 10K characters, labelled beta pricing; Bulbul v2: ₹15 per 10K; billed per character, rounded up per request; every plan starts with ₹1,000 free credits; Starter is pay-as-you-go** — [Sarvam pricing docs (search extract)](https://docs.sarvam.ai/api/getting-started/pricing.md); [sarvam.ai/api-pricing (search extract)](https://sarvam.ai/api-pricing). A third-party blog dated Aug 2026 reports the same ₹30/10K beta rate — [invideo blog](https://invideo.io/blog/sarvam-bulbul-indian-tts/) [3P].
- **Speech-to-text: ₹30 per hour of audio, billed per second, rounded up per request; diarization adds ₹45/h; transcription + translation ₹30/h** — [Sarvam pricing docs (search extract)](https://docs.sarvam.ai/api/getting-started/pricing.md); a reseller lists Saaras v3 at ₹53/h with a ~35% markup — [CallMissed](https://www.callmissed.com/models/saaras-v3) [3P]; a competitor's blog headlines "₹30/hour base rate (2026)" — [smallest.ai](https://smallest.ai/blog/sarvam-ai-speech-to-text-pricing) [3P]. No separate Saarika price was found.
- **Mayura v1 and Sarvam-Translate v1: ₹20 per 10K characters each**, rounded up per request — [Sarvam pricing (search extract)](https://docs.sarvam.ai/api/getting-started/pricing.md).
- **Rate limits (Starter / Pro / Business):** TTS REST 60/200/1,000 req/min, **but bulbul:v3 on Starter is 30 req/min**; TTS WebSocket 60/200/1,000 concurrent (bulbul:v3 30 concurrent on Starter); STT REST 60/100/4,000 req/min; STT WebSocket 20/100/100 concurrent; STT batch 20/100/500 req/min; translation 60/200/1,000 req/min; limits are per account, shared across keys, and "subject to change" — [Sarvam Credits & Rate Limits (search extract)](https://docs.sarvam.ai/api/getting-started/ratelimits). The extract notes that doc versions disagree on some cells.
- PS: "USD 1,000 of Sarvam API credits, usable on all Sarvam APIs"; "Sarvam API credits are shared before the mid-term"; English targets must be "bulbul:v3, voice shubh, language_code="en-IN", 24 kHz. Pace and temperature are free, but declare them"; Bulbul "does not return" word timings — [PS §0, §3 Part B, §11 FAQ](file:///home/user/interiit_minips/Sarvam.pdf).
- USD→INR: an undated live-quote page shows ≈ ₹96.5/USD with a 52-week range of ≈ 86–97 — [Pluang](https://pluang.com/en/tools/currency-converter/usd-inr) [3P, undated].

### Inferences
**English characters to synthesise.** 200 h × 43.1 k chars/h = **8.6 M chars** (range 6.3–12.7 M) for one rendering of `en_text`. Cost = chars ÷ 10,000 × ₹30:

| Scenario | Chars | ₹ | USD (₹96.5 / ₹86) | Share of credit |
|---|---|---|---|---|
| 1 pace, `en_text` only, central | 8.6 M | 25.9 k | $268 / $301 | ~27–30% |
| 1 pace, high case | 12.7 M | 38.0 k | $394 / $442 | ~40–44% |
| 2 paces (or 2 temperatures) on everything | 17.2 M | 51.7 k | $536 / $601 | ~54–60% |
| 1 pace full + 2nd pace on 25% + 15% re-synthesis overhead for B3 step 6 chunk re-renders [A] | 12.0 M | 36.2 k | $375 / $420 | ~38–42% |
| 3 full paces | 25.9 M | 77.6 k | $804 / $902 | ~80–90% (leaves nothing for the rest) |
| `en_alt` also rendered (×2 on any row) | — | — | — | doubles the row |

Credit capacity: ₹86–96.5 k ÷ ₹3 per 1 k chars ≈ **28.7–32.2 M characters** in total. Whether the credit is loaded as USD or as an INR amount is not stated in the PS; the FX range changes capacity by ~12%.

**Other Sarvam usage (pre-mid-term data engine):**
- Bulbul hi-IN as one synthetic Hindi voice: say 20 of the 65 synthetic hours × ~42 k Devanagari chars/h [A: 140 wpm × ~5 chars/word] = 0.84 M chars ≈ **₹2.5 k ≈ $26–29**; ×3 for regenerations ≈ $85.
- Saaras round-trip on synthetic Hindi: 65–90 h (incl. regenerations) × ₹30 = **₹2.0–2.7 k ≈ $21–31**. Saaras drafts on all 125 h real (only as a cross-check, since IndicVoices ships L1 transcripts): ₹3.75 k ≈ $40. Saaras/ASR check on 168 h of Bulbul English per rendering: ₹5.0 k ≈ $55 (optional; an open English ASR on Modal is cheaper).
- Mayura or Sarvam-Translate as a second-opinion translation: all 200 h ≈ 8.4 M Hindi chars × ₹20/10 K = **₹16.8 k ≈ $175–195**; only the 65 h synthetic/entity slice ≈ ₹5.5 k ≈ $57–64. (Billing basis — input vs output characters — not confirmed; assumed input.)
- Sarvam-30B for script generation or prefix translation: tens of millions of tokens at ₹2.5/₹10 per M is ≤ ₹500 ≈ $5, even using the higher ₹73.2/M 105B-conversations rate ≈ $10.

**Suggested Sarvam budget (central):** English rendering 1 pace + 25% second pace + overhead ≈ $400 · Hindi Bulbul voice ≈ $50 · Saaras round-trips and spot drafts ≈ $70 · Mayura on the entity slice ≈ $60 · LLM ≈ $10 · pilot and mistakes reserve ≈ $100 → **≈ $690, leaving ≈ $300 headroom**. Verdict: sufficient, provided the team does **not** render every pace × temperature combination over all 200 h, and pilots on 1 h (≈ 43 k chars ≈ ₹130) first to measure the true chars/hour.

**Rate-limit arithmetic (the real constraint).**
- One request per sentence: 134 k requests ÷ 30 RPM = **74 h (3.1 days) of continuous calls** per rendering on Starter; at Pro's 200 RPM, 11 h. One request per record (≈ 36 k records if clips average ~20 s [A]) = 20 h on Starter. Two paces double these.
- Per-record rendering cuts requests but forces re-segmentation of the English audio per sentence by forced alignment, and the B3 engine wants per-sentence or per-chunk renders. **Ask Sarvam on Discord for a higher bulbul:v3 limit**, or use the TTS WebSocket's 30 concurrent streams on Starter.
- Saaras REST at 60 RPM: ~23 k synthetic clips (65 h at ~10 s [A]) = 6.5 h per round trip. Fine.
- English rendering must happen **after the mid-term** (it builds training pairs, PS §3 Part B), when the clock is shortest; 3–6 days of API time must be scheduled from day 1 after the reveal, or pre-rendered before the mid-term (the PS does not forbid rendering English early, since training pairs are not part of the audited Part A dataset — confirm with organisers).

### Gaps
- Official Sarvam pages could not be opened; prices and limits come from search extracts that themselves note conflicting doc versions. Confirm in the Sarvam dashboard.
- Bulbul v3's maximum characters per request, whether pace/temperature affect price (assumed not), and whether billing counts Unicode code points or bytes for Devanagari were not found.
- Whether "beta pricing" for bulbul:v3 will change before the post-mid-term rendering window is unknown — a rise to v2×2 already happened once; budget a 20–50% contingency.
- No source on whether competition credits are denominated in USD or INR.

---

## Q4. Synthetic Hindi TTS (~65 h) with open TTS on H100

### Takeaway
Rendering ~90 h of audio (65 h kept + ~35% regenerations + pilot) costs roughly **5–30 H100-hours (≈ $20–120)** depending on the model and on batching; IndicF5/F5-style flow matching is in the 5–15 h range, Indic Parler-TTS 5–30 h. Not a budget risk; the risk is engineering time (batching, voice prompts, licence-clean voices) and the gate's rejection rate.

### Cited Findings
- No measured real-time factor for `ai4bharat/indic-parler-tts` was found. A GPU vendor's guide reports ≈ 0.3× RTF for English Parler-TTS on an RTX 3050 (a ~50-word paragraph in 5–8 s) — [GigaGPU](https://gigagpu.com/?p=33067) [3P vendor test]; Parler-TTS lists SDPA, `torch.compile`, streaming and batching as speed-ups — [Beam Parler-TTS guide](https://www.beam.cloud/blog/parler-tts) [3P]; CPU inference is described as impractically slow — [aiindigo Indic Parler-TTS tutorial](https://aiindigo.com/tutorials/getting-started-with-indic-parler-tts-high-fidelity-multilingual-speech-synthesi) [3P].
- F5-TTS (IndicF5's base) reports RTF 0.15 at 16 NFE on 10 s of speech — [F5-TTS summary](https://www.emergentmind.com/topics/f5-tts) [secondary summary of the paper]; an accelerated 7-step variant reaches RTF 0.030 on an RTX 3090 — [Fast F5-TTS / EPSS, arXiv 2505.19931](https://arxiv.org/abs/2505.19931v2); the F5-TTS project cites RTF 0.0394 on an L20 GPU in a client-server setup — [F5-TTS project summary](https://www.sourcepulse.org/projects/1877241) [3P].
- The report's gate regenerates up to 3× then drops, and holds 10 h of reserve for regeneration — [report C1, C3](file:///home/user/interiit_minips/reports/Sarvam%20mini%20PS%20research.md).

### Inferences
Audio to generate: 65 h × (1 + ~0.3 average regeneration) + ~5 h pilots/calibration (B4's 200 amounts + 100 phone numbers per voice) ≈ **90 h** [A].

| Model | Assumed effective RTF on H100 | H100-h for 90 h | $ |
|---|---|---|---|
| Indic Parler-TTS, unbatched HF `generate` | 0.2–0.3 [A, from 0.3 on RTX 3050 and an H100 being faster but unbatched AR decoding being launch-bound] | 18–27 | $71–107 |
| Indic Parler-TTS, batched 16–64 + SDPA/compile | 0.03–0.06 [A] | 3–5 | $11–21 |
| IndicF5 (32 NFE default) | 0.05–0.15 [A, from 0.15 at 16 NFE on older GPUs] | 5–14 | $18–53 |
| Veena / Svara (LLM + codec TTS) | unknown; assume 0.05–0.2 with vLLM-style batching [A] | 5–18 | $18–71 |
| Bulbul hi-IN | API, see Q3 | 0 | ≈ $26–85 Sarvam credit |

A mix of 3–4 voices ≈ **10–20 H100-h ≈ $40–80** is a realistic planning number. Run several models in parallel Modal containers to finish in an afternoon. These figures exclude Modal CPU/RAM charges and cold-start time [A: add ~10–15%].

### Gaps
- No primary RTF for Indic Parler-TTS, IndicF5, Veena or Svara on an H100; all numbers are extrapolated. A 10-minute benchmark per model on Modal (< $1 each) settles it.
- The real regeneration rate under the two-ASR entity gate is unknown; the B4 calibration experiment measures it.

---

## Q5. ASR round-trip validation (2 ASRs) on synthetic Hindi + alignment of all audio

### Takeaway
All of it fits in **≈ 3–10 H100-hours (≈ $12–40)** plus ≈ $30 of Saaras if Saaras is one of the two ASRs. Hindi MMS_FA alignment of all 190 h (needed for the dataset regardless of B3) is ~1–2 H100-h on GPU but would take days on a laptop CPU, so run it on Modal.

### Cited Findings
- STT ₹30/h — [Sarvam pricing (search extract)](https://docs.sarvam.ai/api/getting-started/pricing.md).
- A user benchmark of a wav2vec2 model (claimed 300M) on CPU processed ~5,500 s of audio in 1,280 s on 1 CPU and 750 s on 2 CPUs (RTF ≈ 0.23 and 0.14 by our arithmetic), with a 1B model far slower — [IPED issue #1214](https://github.com/sepinf-inc/IPED/issues/1214) [3P; CPU model not stated; model attribution uncertain].
- The report's round-trip pair is IndicConformer-600M + Saaras or Omnilingual 7B LLM-ASR; English alignment uses MFA english_us_arpa + MMS_FA cross-check — [report C2, C3](file:///home/user/interiit_minips/reports/Sarvam%20mini%20PS%20research.md).
- WhisperX advertises ~70× real-time batched transcription with large-v2 on GPU — [WhisperX README](https://github.com/m-bain/whisperX) (claim recalled from the README; not re-fetched in this audit).

### Inferences
**MMS_FA compute model.** The MMS_FA bundle is ~1.18 GB fp32 ≈ 295 M parameters (wav2vec2-large class). Forward cost ≈ 2 × 300 M × 50 frames/s ≈ 3e10 FLOP per audio-second, plus attention and the conv front end ≈ 4–5e10 FLOP/s of audio [A]. 190 h = 684 k s → ≈ 3e16 FLOP → **minutes on an H100**; real wall time is I/O-bound, budget **1–2 H100-h**. On a 4–8-core laptop at ~100–300 GFLOP/s effective [A] the RTF is ≈ 0.15–0.5 → **28–95 CPU-hours** for 190 h, so do it on Modal (GPU or many CPU containers).

| Job | Audio | Engine | Estimate | $ |
|---|---|---|---|---|
| ASR #1 round trip on synthetic Hindi | ~90 h | IndicConformer-600M CTC on GPU, RTF ~0.005–0.01 [A] | 0.5–1 H100-h | $2–4 |
| ASR #2 round trip | ~90 h | Saaras API | ₹2.7 k | **≈ $30 Sarvam** |
| ASR #2 alternative | ~90 h | Omnilingual 7B LLM-ASR, batched, RTF ~0.02–0.05 [A] | 2–5 H100-h | $8–20 |
| Hindi word timings, all records | ~190 h | MMS_FA (+ cross-aligner on flagged clips) | 1–2 H100-h | $4–8 |
| English word timings per Bulbul rendering | ~168 h | MMS_FA on GPU + MFA on CPU containers | ~1 H100-h + CPU | $4–10 |
| Second pace (if rendered) | +168 h | same | +1 H100-h | +$4–10 |

Total ≈ **5–10 H100-h ≈ $20–40** plus ~$30 of Saaras. MFA is CPU-only; 168 h on 16 Modal CPU cores is a few hours [A].

### Gaps
- No GPU throughput figures for IndicConformer or Omnilingual 7B were found; no Hindi MMS_FA CPU benchmark exists. A one-clip timing run on Modal and on the laptop would replace these assumptions.

---

## Q6. LoRA fine-tuning cost, Modal's H100 price, and evaluation cost

### Takeaway
The report's arithmetic is correct: at 12 k tok/s and 9 × 12.5 tokens per audio-second, one epoch over 200 h is **1.875 H100-h ≈ $7.40 at $3.95/h**. Modal's H100 list price of **$0.001097/s ≈ $3.95/h** is confirmed by several third parties but not by modal.com itself (unreachable). The dominant post-mid-term cost is **evaluation wall-clock**, not training: a protocol-faithful 200-session dev eval takes ≈ 2 H100-h (≈ $8) per checkpoint.

### Cited Findings
- Modal H100 at **$0.001097/s ≈ $3.95/h**, per-second billing — [Beam pricing breakdown (June 2026 snapshot)](https://www.beam.cloud/blog/modal-pricing-explained) [3P]; [Spheron](https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/) [3P]; an Aug 2026 comparison puts the same H100 at $3.95/h on Modal vs $5.49 on Replicate and $6.50 on Baseten — [devtoollab](https://devtoollab.com/blog/best-ai-inference-platforms) [3P]. Starter tier includes $30/month of free credits; Team is $250/month + compute — [morphllm](https://www.morphllm.com/modal-pricing) [3P].
- Multipliers (sources conflict): region pinning 1.25× (US/EU/UK/AP) up to 2.5× — [Blaxel](https://blaxel.ai/blog/modal-pricing-alternatives-guide), [morphllm](https://www.morphllm.com/modal-pricing); 1.5–1.75× — [usagepricing](https://usagepricing.com/blueprint/modal), [Spheron](https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/); 1.15× broad / 1.75× narrow — [flexprice](https://flexprice.io/pricing-index/modal). Non-preemptible is 3×, which morphllm says applies only to CPU functions — [morphllm](https://www.morphllm.com/modal-pricing) [3P]. An aggregator headline of "$0.92 an hour" for Modal H100 is unsupported — [keenable](https://select.keenable.ai/r/nvidia-h100-cloud-cost-real-price-per-gpu-hour/MYEY81nBL3-DIQrqXC_g-g).
- moshi-finetune: ≈ 12 k tokens/s on 1×H100 for Moshi 7B LoRA, peak 39.6 GB; tokens = steps × GPUs × batch × duration_sec × 9 × 12.5 — [moshi-finetune README](https://github.com/kyutai-labs/moshi-finetune) (via `research_notes/.../finetune_and_latency.md`).
- Final eval protocol: one H100 80 GB, 16 CPU cores, 128 GB RAM; sessions 10–60 s, one at a time with a 2 s gap; frames at real time — [PS §4–§6](file:///home/user/interiit_minips/Sarvam.pdf).
- "Part of the Modal credits comes about two weeks after the PS release, and the rest after the mid-term" — [PS §11](file:///home/user/interiit_minips/Sarvam.pdf).

### Inferences
**LoRA (re-derived).** 12,000 ÷ (9 × 12.5 = 112.5) = 106.7 audio-s per wall-s → 720,000 s ÷ 106.7 = 6,750 s = **1.875 h → $7.41**. The 2,000 × 16 × 100 s config = 3.2 M audio-s ≈ 8.3 h ≈ **$33** per run. Caveats the report should state: (i) training pairs are longer than their Hindi audio (English trails by the lag, so +10–30% [A]); (ii) the base's architecture is unknown — a 3B base is ~2–3× cheaper, a full fine-tune or a larger base 3–5× dearer; (iii) Modal also bills CPU and RAM and idle container time, and region pinning (if used) adds 1.15–1.75× — leave region unpinned. Planning figure: **$40–60 per full LoRA run** including overheads.

**Evaluation.** Per session ≈ 30 s average audio + ~3 s flush + 2 s gap ≈ 35 s [A]. A 200-session dev set ≈ 7,000 s ≈ **1.9 H100-h ≈ $7.7** per checkpoint in protocol-faithful real-time replay; add English ASR for ASR-BLEU (Whisper on the same GPU, minutes). 20 checkpoints × 1 eval ≈ $155; with a 2-arm × several-seed design and the base-checkpoint baseline, ≈ $200–350. Cheaper: an offline, faster-than-real-time replay (simulated clock) for screening, keeping real-time replay for the final 3–5 candidates.

**Modal budget plan ($1,000 ≈ 253 H100-h).** Split between tranches is not stated; assume 50/50 [A].

| Phase | Item | H100-h | $ |
|---|---|---|---|
| Pre-mid-term (~$500) | Synthetic Hindi TTS (Q4) | 10–20 | 40–80 |
| | Round-trip ASR + Hindi alignment (Q5) | 3–7 | 12–28 |
| | B3 MT-gain scoring (Q1) | 1–3 | 4–12 |
| | B7 calibration 2,000 sents (Q2) | 1–2 | 4–8 |
| | LLM translation of ~134 k sentences × 2 versions with a local 8B (≈ 134 k × ~120 out tokens ≈ 16 M tokens ≈ 0.5 h [A]) or via API | 0.5–2 | 2–8 |
| | ASR-proxy ratio sweep (B1: small Hindi ASR on 10 h mixes × 3) | 5–15 [A] | 20–60 |
| | Hibiki-Zero 3B rehearsal on 10–20 h | 3–10 [A] | 12–40 |
| | Debugging, cold starts, mistakes (×1.3) | — | ~60 |
| | **Subtotal** | **~30–70** | **~$150–300** |
| Post-mid-term (~$500 + leftovers) | English alignment (1–2 renderings) | 1–2 | 4–10 |
| | Base-checkpoint baseline eval | 2 | 8 |
| | 6–8 LoRA runs | 50–65 | 240–320 |
| | 15–25 dev evals | 30–50 | 120–200 |
| | **Subtotal** | **~85–120** | **~$370–540** |

Verdict: the budget is adequate with ~$150–450 slack; it would not support full-dataset B7 *and* many runs of a full fine-tune at the same time.

### Gaps
- modal.com/pricing was unreachable; confirm the H100 rate, the CPU/RAM per-second rates and whether region/non-preemptible multipliers apply to GPU functions.
- The tranche split of Modal credits and the exact release dates are not in the PS.
- The base's real training throughput is unknowable before the reveal.

---

## Q7. Human effort

### Takeaway
Roughly **25–40 person-hours for the mini PS** (mostly Part D labelling, recording and writing code) and **~40–70 person-hours of human checking for the main data engine**, provided IndicVoices' native Level-1 transcripts are used as-is with scripted rule fixes. Re-transcribing 105 h of real audio by hand (hundreds of person-hours) is not feasible and is not what the report proposes.

### Cited Findings
- IndicVoices ships native Level-1 verbatim transcripts — [IndicVoices README](https://github.com/AI4Bharat/IndicVoices) (as cited [V] in the report B1); the report still calls for a spot check of 50 L1 transcripts for English-in-Devanagari and numbers-as-words, and lists five places where a human is needed (fillers, number spoken form, loanword spelling, sentence segmentation, boundary checks) — [report B1, C2](file:///home/user/interiit_minips/reports/Sarvam%20mini%20PS%20research.md).
- The report's audit: listen to 1–2% of accepted clips stratified by voice and entity type; B4 human listening to 50 clips; B3 hand-marking safe points for 50 sentences + 6 clips; B7 100 sentences; scale the 30-boundary check to ~200 boundaries — [report B3, B4, C3, C7](file:///home/user/interiit_minips/reports/Sarvam%20mini%20PS%20research.md).
- Part D requires 6 clips of 20–40 s, Level-1 transcripts, 30 hand-checked boundaries, two translations, and a checker with accuracy and failure cases — [Mini PS Part D](file:///home/user/interiit_minips/Inter%20IIT%20Tech%20Meet%2015.0_%20Sarvam.ai%20Mini%20PS.pdf).
- Human inter-labeller disagreement on word boundaries is ~16 ms mean in Buckeye — [arXiv 2403.19509](https://arxiv.org/pdf/2403.19509) (as cited in the report, [S: re-check]).

### Inferences
All per-item rates are [A] (no source on Hindi transcription or labelling throughput was found).

| Task | Size | Rate [A] | Person-hours |
|---|---|---|---|
| **Mini PS** | | | |
| Plan and record 6 clips; convert | 6 × 20–40 s | — | 1–2 |
| Level-1 transcripts by hand (3–4 min audio) | ~3 min | ~15–25× real time for verbatim with fillers | 1–1.5 |
| 30-boundary blind check in Audacity | 30 | 1–2 min each + setup | 1–1.5 |
| Gold entities/negations/repairs for 6 clips | ~50 items | ~1 min each | 1 |
| Human verdicts on (entity × 4 outputs) | ~200 pairs | ~20 s each | 1–1.5 |
| Code (aligner, checker, sim, plots, run_clip, RESULTS.md) and report writing | — | — | 20–30 |
| **Mini PS total** | | | **~25–40** |
| **Main PS data engine** | | | |
| IndicVoices L1 spot check | 50 transcripts | ~1–2 min each | 1–2 |
| MUCS (12 h) English→Devanagari and numbers-as-words fixes | LLM rewrite + review of a 5% sample | 3× real time | ~2 |
| FLEURS (5 h) normalised read text → L1 conventions | scripted + 5% review | 3× real time | ~1 |
| Human audit of accepted clips | 1–2% of 190 h = 1.9–3.8 h audio | 3× real time | 6–12 |
| B4 TTS calibration listening | 50 clips | ~1 min | 1 |
| Translation review queue (COMETKiwi bottom 5–10%, polarity mismatches) | e.g. 2–5% of 134 k sentences reviewed at 20–30 s each [A] | — | 15–40 |
| B3/B7 hand-marked safe points | ~250 sentences | ~2 min each | 8–10 |
| 200-boundary stratified aligner check (C7), Hindi + English | 200 | ~1.5 min | 5 |
| Real speaker-disjoint dev set (30–50 entity clips from friends, with consent) | 50 clips | ~15 min each incl. transcription and gold | 12 |
| **Main PS total** | | | **~50–85** |

Gold entity labels for the synthetic slice cost **zero human time by design** (generated with the spoken form, B4). The translation review queue is the item most likely to blow up: if the gates route 10% of sentences to review, it is ~100 person-hours, so set thresholds by queue capacity, not only by quality.

### Gaps
- Whether IndicVoices L1 transcripts already satisfy the PS's extra rules (English in Devanagari without bracketed Latin, numbers as words) is unknown until the 50-transcript check; if heavy rewriting is needed, effort rises sharply.
- No published throughput for Hindi verbatim transcription or boundary labelling was found; all rates are assumptions.

---

## Q8. Part E-ML regime (2): prefix translation via LLM — number of calls, and whether call latency matters

### Takeaway
Regime (2) is an **offline simulation**: the report computes lag on the Hindi playback timeline from word timings and fixed (or Bulbul-measured) word durations, with an optional fixed compute delay δ, and caches every call. API latency (500–1,000 ms/call) never enters the plotted latency, so the critique's "non-viable online" objection does not apply. The report does, however, **undercount the calls**: it is ~1,000–2,500 calls, not "a few hundred" — still < $1 and < 1 h.

### Cited Findings
- Report E-ML: ready time = end of Hindi word min(k+i−1, J) "plus an optional compute delay δ (e.g. 0.2 s)"; regime (2) translates each prefix "given the English already committed, and commit only word i"; "Cache every call to JSON so the interview demo runs offline; with six clips and k ∈ {1, 2, 3, 5, 7, ∞} it is a few hundred calls"; regime (3) is "an analysis tool, not a deployable component" — [report E-ML](file:///home/user/interiit_minips/reports/Sarvam%20mini%20PS%20research.md).
- The mini PS asks to "Simulate a wait-k pipeline on your clips … give each English word a fixed duration or measure it from TTS audio" — [Mini PS Part E](file:///home/user/interiit_minips/Inter%20IIT%20Tech%20Meet%2015.0_%20Sarvam.ai%20Mini%20PS.pdf).
- The submitted system is the fine-tuned checkpoint itself; external APIs "must not be part of the submitted system" — [PS §3 Part B, §10](file:///home/user/interiit_minips/Sarvam.pdf).
- Sarvam LLM limits 40–60 RPM on Starter — [Sarvam rate limits (search extract)](https://docs.sarvam.ai/api/getting-started/ratelimits).

### Inferences
Six clips × 20–40 s at 120–160 wpm ≈ 240–640 Hindi words ≈ 20–53 sentences (central ≈ 420 words, 35 sentences of ~12 words, ~11 English words each). Per sentence and finite k, calls ≈ number of English words committed while the source is unfinished (≈ |x|−k+1, capped at |y|) + 1 flush call for the remainder: k = 1, 2, 3, 5, 7 → 12 + 11 + 10 + 8 + 6 = 47, plus 1 call for k = ∞ → **≈ 48 calls/sentence → ≈ 1,700 calls (range ~1,000–2,500)**. Tokens ≈ 1,700 × (~400 in + ~40 out) ≈ 0.7 M in + 0.07 M out → **< $1** at any mainstream API rate (≈ ₹2.4 at Sarvam-30B's ₹2.5/₹10). Time: 14–28 min sequential at 0.5–1 s/call; ≈ 43 min at a 40 RPM limit; minutes with 10-way concurrency.

**Interview implication:** for one *new* 30 s clip, regime (2) is ≈ 6 sentences × 48 ≈ 280 calls ≈ 2.5–5 min sequential and requires network. Either run it with bounded concurrency, run it with IndicTrans2-200M locally (≈ 280 × 0.3–1.5 s ≈ 1.5–7 min on CPU, see Q9 — and IndicTrans2 cannot condition on committed English without forced-prefix decoding), or show regimes (1) and (3) live and regime (2) from cache. Suggest correcting "a few hundred calls" to "about 1,000–2,500 calls" in the report.

If the team wants the plot to reflect a cascade's real compute delay, δ can be swept up to 0.5–1.0 s — but that models a system the PS forbids submitting, so the default δ ≈ one 80 ms frame to 0.2 s is the relevant one.

### Gaps
- Actual clip lengths and sentence counts will set the call count; recompute once the clips exist.

---

## Q9. Part D on a laptop CPU: MMS_FA, IndicTrans2-200M, and total `run_clip.py` runtime

### Takeaway
Part D runs comfortably on a laptop CPU: MMS_FA alignment of all six clips takes **≈ 0.5–1.5 min** plus a 5–15 s model load; IndicTrans2-200M translation of ~36 sentences ≈ **0.2–1 min**; the interview `run_clip.py` on one new 30 s clip is **≈ 30–90 s of compute** (excluding regime (2) LLM prefix calls), dominated by model loading. The human step — typing a Level-1 transcript — is the longest part (~5–10 min).

### Cited Findings
- MMS_FA weights ≈ 1.18 GB; 16 kHz input; 20 ms emission frames; plan pre-downloads weights so the demo runs offline — [report D0, D3](file:///home/user/interiit_minips/reports/Sarvam%20mini%20PS%20research.md).
- CPU wav2vec2 benchmark: ~5,500 s of audio in 1,280 s (1 CPU) / 750 s (2 CPUs), RTF ≈ 0.14–0.23 — [IPED issue #1214](https://github.com/sepinf-inc/IPED/issues/1214) [3P; hardware unspecified].
- IndicTrans2 distilled models have ~20% of the parameters and "~36% reduced inference time"; test-set decoding went from ~52 s to ~33 s — [AI4Bharat IndicTrans2 blog](https://ai4bharat.iitm.ac.in/blog/indictrans2); the model card recommends a GPU with flash_attn — [HF model card](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M). No CPU sentences/s figure was found.
- The report's IndicTrans2 call uses `num_beams=5, max_length=256`, ~0.8 GB fp32 — [report D5](file:///home/user/interiit_minips/reports/Sarvam%20mini%20PS%20research.md).

### Inferences
- **MMS_FA on CPU.** ~4–5e10 FLOP per audio-second (Q5) at ~100–300 GFLOP/s effective on 4–8 laptop cores [A] → RTF 0.15–0.5, consistent with the IPED 0.14–0.23. Six clips × ~30 s = 180 s → **27–90 s**; a single 40 s clip in one pass ≈ 6–20 s. One-pass attention over 2,000 frames needs only a few GB of RAM [A]. CTC Viterbi (`forced_align`) and uroman are < 1 s per clip. This laptop sandbox (4 vCPU Xeon @ 2.1 GHz) could not test it (no torch, weights blocked).
- **IndicTrans2-200M on CPU.** ~20 output tokens × 5 beams through a ~100 M-param decoder ≈ 2e10 FLOP per sentence; small-matrix CPU efficiency ~20–100 GFLOP/s [A] → **0.3–1.5 s per sentence** → 36 sentences ≈ 10–55 s (+5–10 s load). Batching all sentences of a clip helps.
- **`run_clip.py` on one new 30 s clip:** imports + model loads 15–30 s; ffmpeg < 1 s; MMS_FA 5–15 s; IndicTrans2 ~6 sentences 2–9 s; LLM translation (online, cached thereafter) 5–30 s; checker < 1 s; wait-k regimes (1) and (3) + matplotlib 2–5 s → **≈ 30–90 s**. Adding regime (2): + 2.5–5 min online (Q8). Recommendation: warm models once at the start of the interview (a `--serve` or notebook mode) and show regime (2) from cache.

### Gaps
- No measured CPU runtime for MMS_FA or IndicTrans2-200M on a laptop; time `make clip` once on the actual laptop and record the numbers in RESULTS.md.

---

## Q10. Timeline: what fits before the mini PS deadline vs. before the mid-term

### Takeaway
Neither PDF gives dates, so the timeline is relative. **Before the mini PS deadline**, do only CPU/cheap-API work: Part D, Part E-ML, the B3 hand-marked comparison and B7 on ~150 sentences — all fit in ~25–40 person-hours and < $5. **Before the mid-term**, the data engine (manifest count, TTS, gates, Hindi alignment, translations, MT-gain scoring, audits) fits in ~$150–300 of the first Modal tranche plus ~$150–250 of Sarvam credit. **After the mid-term**, the critical path is Bulbul English rendering (rate-limited to days) and real-time evaluation, not training GPU-hours.

### Cited Findings
- Modal credits: part ~2 weeks after PS release, rest after the mid-term; Sarvam credits before the mid-term; base checkpoint, inference code and reference client only after the mid-term; sample records and a validator "before the mid-term (release date to be finalised)" — [PS §11](file:///home/user/interiit_minips/Sarvam.pdf).
- Mini PS: individual work; 4-page PDF + repo with RESULTS.md; live run on a new clip in the interview — [Mini PS §2](file:///home/user/interiit_minips/Inter%20IIT%20Tech%20Meet%2015.0_%20Sarvam.ai%20Mini%20PS.pdf).
- After the mid-term, "you may not add new records or hours" — [PS §3 Part B](file:///home/user/interiit_minips/Sarvam.pdf).
- Modal Starter includes $30/month free credits — [morphllm](https://www.morphllm.com/modal-pricing) [3P]; Sarvam plans start with ₹1,000 free credit — [Sarvam pricing (search extract)](https://docs.sarvam.ai/api/getting-started/pricing.md).

### Inferences
**Before the mini PS deadline (likely before any competition credit arrives):** Part D (recording, L1, MMS_FA on CPU, 30 boundaries, two translations, checker) and Part E-ML (all three regimes, cached) — CPU only, LLM calls < $1, IndicTrans2 local. B3's no-GPU 50-sentence experiment and B7 on ~150 sentences fit within free tiers (Modal's $30/month, Sarvam's ₹1,000 ≈ 333 k Bulbul characters or ~33 h of STT). Do **not** promise anything requiring the full dataset.

**Mini PS → mid-term (Modal tranche 1 + Sarvam credits):**
1. Week 1: IndicVoices manifest/speaker-count (1 CPU day), 50-transcript L1 check, B4 TTS calibration (≈ $5), Modal and Sarvam pilots on 1 h to measure chars/hour and RTF.
2. Weeks 2–3: script generation, open-TTS rendering (10–20 H100-h), two-ASR gates, Hindi alignment of all records (1–2 H100-h), LLM translations, gates, review queue.
3. Week 3–4: MT-gain scoring (1–3 H100-h), B7 on 2,000 sentences, ASR-proxy ratio sweep, Hibiki-Zero rehearsal, JSONL + hours ledger + reproducibility rebuild, human audit (~10 h), mid-term report.
Optional and permitted-if-confirmed: pre-render Bulbul English for the final dataset so the post-mid-term clock is not spent waiting on 30 RPM.

**After the mid-term:** day 0 base-checkpoint eval; days 1–4 Bulbul English rendering (if not pre-rendered: 74 h of calls per pass on Starter) and English alignment; placement engine + exporter (CPU); then 6–8 LoRA runs (~8 h each) interleaved with real-time evals (~2 h each). Each "train + eval" cycle is ~10–12 h wall-clock on one GPU, so parallel Modal containers matter more than budget.

**What is not realistic:** full-dataset B7 *and* many full fine-tunes; rendering every pace × temperature over 200 h; hand-correcting real-data transcripts at scale; a 200-session real-time eval after every training run without parallel containers.

### Gaps
- Mini PS deadline, mid-term date, final submission date and the Modal tranche split are not stated in either PDF; the plan above is relative and must be re-anchored when dates are announced.
- Whether rendering Bulbul English training targets before the mid-term is allowed (they are not part of the audited dataset) should be confirmed with the organisers.
