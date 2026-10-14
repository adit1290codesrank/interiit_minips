# Fix the framing; the engineering already holds

> **Status: all 17 checklist items have been applied to the master report** ("Hold the commitment, not the sentence"). Line numbers below refer to the master report *before* those fixes, and quotes of the old title and wording are kept as the record of what was wrong.

The Sarvam mini PS study material is built on the right technology for October 2026, and most of it is feasible, but it needs fixes before the interview. The stack is a Hibiki-family streaming target, training data that teaches the model when to speak, MMS_FA alignment, wait-k analysis and Bulbul v3 targets, and no newer open system replaces it. The plan is feasible within the PS budgets: Modal needs about $150–300 before the mid-term and $370–540 after it, and Sarvam credit about $690 of $1,000. The real time constraint is Bulbul's rate limit, not GPU-hours. All the arithmetic in the report recomputes exactly. Where the report fails is consistency and coverage:

- **Title and opening example.** The title and the opening example contradict the report's own A5 anticipation analysis and its B3 placement rule.
- **Missing PS rules.** Part C omits three format rules from PS §3 Part A and §5.2.
- **Data plan.** The 105 h IndicVoices plan rests on a Hindi subset the primary paper puts at about 65 h train plus 5 h test.
- **Sources.** The source appendix lists 36 of about 99 cited sources.
- **Prefix-translation simulation.** E-ML regime (2) is not labelled as an offline simulation.

The external AI critique was written against an older version, and commit 6b06d3e already fixed four of its points: the parser rewrite, removal of "fifty million", Hibiki-family wording and the passive-voice caution. Of its eleven claims as they stand today:

- **Valid and fixed:** one (the parser bugs).
- **Correct but already covered:** two (observation adding; torchcodec).
- **Half right:** three. The slogan, the IndicVoices conversations and telephone bandwidth each point at a real gap, though not the one named.
- **Wrong:** five (the timing mandate proving Hibiki, crore→million, cost of MT-likelihood placement, flat TTS prosody and the audio KPI, and E-ML latency). The E-ML objection sits next to two real issues.

The audit then found about forty further issues. All eight parser failure families it found have since been fixed (seven in commit 986559a, decimals (दशमलव) in the follow-up), and 59 test phrases now pass [L]. The rest are mostly one-sentence edits, which a prioritised checklist below orders by interview risk.

**Evidence tags.** Tags are kept from the audit notes:
- **[V]**: checked against primary code, a wheel, a README or the PS PDFs.
- **[L]**: tested locally in this audit.
- **[S]**: a search-engine extract or secondary source, to re-check before quoting.
- **[E]**: an estimate or inference by the audit, with its assumptions stated in the notes.

The proxy blocked arXiv, Hugging Face, the ACL Anthology and Sarvam's docs, so every paper-level and vendor-price number is [S]. Line numbers ("L*n*") refer to the master report at HEAD 986559a.

## Verdict by part: sound arithmetic, five framing and coverage faults

The audit re-ran every computation in the report, with exact fractions where relevant. All of them hold:
- AL = 21/8 = 2.625 and the off-by-one 59/16 = 3.6875.
- The k = 1..9 table, the oracle AL of 41/16, LAAL 3.3, DAL 3 and AP 51/72.
- The 787.5 → 937.5 ms time-domain example.
- 643 frames = 51.44 s, and 3,840-byte frames.
- The 200 h split and the points tables.
- 1.875 H100-h ≈ $7.41 per LoRA epoch, rebuilt from the moshi-finetune README formula of tokens = steps × GPUs × batch × duration × 9 × 12.5 ([moshi-finetune](https://github.com/kyutai-labs/moshi-finetune)) [V].

The wait-k code reproduces SimulEval's `ALScorer` line for line ([SimulEval latency_scorer.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)) [V]. The echo server, run verbatim against a protocol-faithful client, returned `/health` 200, echoed exactly `valid_samples` × 2 bytes and sent one `session_complete` [L]. So the numbers can be copied into the handwritten appendix as they stand.

The weaknesses are in what the report says and leaves out, not in what it computes.

| Part | Feasible | Internally consistent | Valid | Current (Oct 2026) |
|---|---|---|---|---|
| **A** Understanding | Yes | **No.** The title "Hold the verb, speak everything else" and the L3 example ("speak 'Yesterday I… my old bank account' immediately") contradict A5's "got … not heard: anticipation" (L89) and B3's monotone `cummax` (L165). The A3 table's own rule numbers clash with the PS's rules 1–6 (L54) | **Mostly.** All arithmetic recomputes [L]. Three overclaims: "36 … are lost by committing too early" (L3/L16); "risks 18 points" (L75); the brief's "k + i" is "off by one" with no "if i is 1-indexed" hedge (L3, L597) | Yes. Add YAAL/LongYAAL alongside AL/LAAL |
| **B** Research | Yes. MT-likelihood placement costs about $2–11 [E]. B7 is affordable on a calibration subset only | **Partly.** `en_alt` is defined three ways (B2 L151, C5 L293, D5 L437). B6's PAD gate needs a Hindi verb signal that the model's English text stream does not carry (L197). Kathbath sits in different buckets in B1 and C1. The 25% entity-TTS share lies outside its own 0–20% sweep | **Valid with refinements.** Reframe the hold rule around *commitments*. Add question force, tags, post-verbal नहीं and कल tense. The causative is not free deferral | **Mostly.** Add GRPO latency RL (Hibiki-Zero stage 2). Add a caveat to the Hibiki-Zero rehearsal plan |
| **C** Dataset | **At risk.** R1's 105 h of IndicVoices Hindi exceeds the ~65 h + 5 h the primary paper reports [S]. Bulbul rendering is rate-limited to about 74 h of calls per pass [S/E] | **No.** Clips ≤ 60 s, `source.model`/`source.voice` for synthetic records, synthetic `start`/`end`, the §10 reproducibility re-run and `model_card.md` are all missing | **Partly.** Real-vs-synthetic **bandwidth shortcut**. No wideband telephone branch. Speed perturbation breaks word timings. External noise corpora are a grey area under the "only Part A" rule | **Partly.** Pin `bulbul:v3` explicitly (Bulbul V4 renames Shubh). Saaras V4 has superseded v3 |
| **D** Build | Yes. `run_clip.py` takes about 30–90 s on a laptop CPU [E] | **Partly.** D7 headings (L481–484) use different numbers from the D sections; NEXT_STEPS mismatches. The D0 "12 phrases" slip is already fixed in 986559a | **Two code defects.** The D1 `ffprobe` glob fails with more than one clip [L]. The D3 sketch drops digit words instead of interpolating them [L] | **Mostly.** torch is at 2.14.1, so "all current" (L377) is false; pin torch 2.11 to match torchaudio 2.11.0. Saaras V4 |
| **E** Track | Yes. Regime (2) is about 1,000–3,000 cached LLM calls, under $1 [E] | **Partly.** δ means three different things (A4, B3, E-ML) | **One labelling fault.** Regime (2) is not marked as an offline simulation, so it can be read as a banned cascade. The observation-adding formula is not convex | Yes. DeepFilterNet3 is stale on PyPI, as the report already says |
| **Parser** | Yes | Matches the docs after 986559a | **Fixed.** At 6b06d3e it passed its 37 tests but failed 27 of 65 adversarial probes, 19 of them undocumented. The suite (59 phrases + 4 tests) now passes, and spot probes of पाँच सौ करोड़, सवा लाख, पौने एक बजे and दो दशमलव पाँच करोड़ return correct values [L]. The remaining probe differences are documented limits | n/a |

## The external critique, point by point: five wrong, three half right, three already handled

The critique was aimed at the pre-6b06d3e text. That commit rewrote the number parser and added 37 tests. It replaced the claim that the scorer accepts "fifty million" with "Keeping 'crore' is safe either way" (L73). It changed the base-model wording to "most likely Hibiki-family … never 'proven'" (L37). And it added "Do **not** use the passive as a deferral device" (B2, L151). A grep finds no leftovers of the old claims [V]. The table rules on each claim against today's text.

| # | Critique claim | Status vs. 6b06d3e | Verdict now | Evidence | Action |
|---|---|---|---|---|---|
| 1 | "Hold the verb" is a syntactic fallacy; use passive, topicalisation, fronted adverbials | Passive caution added | **Half right.** The slogan and the L3 example *are* wrong: canonical English cannot voice the object before the verb, so "Yesterday I… my old bank account" strands the object. But B2 already prescribes fronted adverbials, causatives and topicalisation. Rejecting the passive as the default is *correct*: it deletes the agent, and the intent KPI scores "who does what to whom" | He et al. 2016 document passivisation in human Ja→En interpreting, which optimises fluency at low lag ([ACL N16-1111](https://aclanthology.org/N16-1111/)) [S]. The intent KPI text is in [PS §7](file:///home/user/interiit_minips/Sarvam.pdf) [V]. Indian English uses left dislocation about 9× more than British English (1.42 vs 0.16 per 1,000 words) ([BSU](https://journals.bsu.by/index.php/philology/en/article/view/3574)) [S] | Retitle; rewrite L3; add one sentence to B2; widen the hold rule (S1-1, S2-13) |
| 2 | The English word-timing mandate proves the base is Hibiki, not Hibiki-Zero | Wording already fixed (L37) | **Wrong.** Hibiki-Zero keeps Moshi's time-aligned inner-monologue text stream. What it drops is word-level *source↔target* alignment, not target-side word times | `hibiki_zero/inference.py` emits one text token per 80 ms frame, filters PAD/EPAD ids `(0, 3)` and truncates audio at the text-EOS index ([hibiki-zero](https://github.com/kyutai-labs/hibiki-zero/blob/main/hibiki_zero/inference.py)) [V]. moshi-finetune's `Interleaver` takes `(word, (start, end), speaker)` ([interleaver.py](https://github.com/kyutai-labs/moshi-finetune/blob/main/finetune/data/interleaver.py)) [V] | None required. Optional footnote: the mandate narrows the base only to "a model with a time-aligned target text stream" |
| 3 | IndicVoices multi-speaker conversations violate "one speaker per clip" | Not addressed | **Half right.** Conversations were recorded as two people on one telephone bridge, so whole files do break the rule. But segments carry `speaker_id`, and AI4Bharat's own chunker skips `speaker_id == 1`. Per-speaker segments therefore comply. The bigger problem is the hours | IndicVoices paper on the telephone bridge ([arXiv 2403.01926](https://arxiv.org/html/2403.01926)) [S]; the official chunker ([utils_dataset_create_iv.py](https://github.com/AI4Bharat/IndicVoices/blob/master/data_prep_scripts/utils_dataset_create_iv.py)) [V] | Use per-speaker segments only, with a backchannel/overlap screen. Re-plan R1 hours (S1-9) |
| 4 | Converting crore→million violates the dataset rules and risks entity points | "fifty million" claim removed | **Wrong against today's report.** The report keeps lakh/crore in the dataset and in speech. Its checker only *accepts* "50 million" as value-equivalent, behind a unit-preservation flag (L73, L471). Converting *would* break PS §3, so keeping units weakly dominates | "Amounts keep the speaker's units (lakh, crore)" ([PS §3](file:///home/user/interiit_minips/Sarvam.pdf)) [V]. The FAQ shows equivalence only within one unit system (25,000 / 25k) [V]. Entity metrics in speech-LLM ST are often exact match ([arXiv 2512.16378](https://arxiv.org/pdf/2512.16378)) [S] | None. Never append a conversion either: "reading out alternatives the speaker did not offer" is a §10 integrity risk [V] |
| 5 | Hibiki-style MT-likelihood placement is computationally prohibitive | Not addressed | **Wrong.** One teacher-forced pass scores every target token, so the cost is (\|x\|+1) passes per sentence. That is about 1.8 M passes and 81 M tokens for 200 h: **0.5–3 H100-h, about $2–11** with MADLAD-3B. Even the critique's own over-count costs only $27–161 | Hibiki's contextual alignment uses progressively longer source prefixes ([arXiv 2502.03382](https://arxiv.org/html/2502.03382)) [S]; FLOP-based estimate [E] | Add one cost sentence to B3. The real cost is 1–2 days of engineering and untested validity on Hindi, so run the 50-sentence check first |
| 6 | Flat open Indic TTS prosody degrades the audio-quality KPI | Not addressed | **Wrong mechanism.** Listeners rate "how clean and natural the **English** speech sounds", and the English is fixed to Bulbul v3 `shubh`. Flat *Hindi* prosody widens the synthetic–real *input* gap instead, which hits translation, intent, entity and latency (52 points) | [PS §3, §9.1](file:///home/user/interiit_minips/Sarvam.pdf) [V]; the Translatotron 2/CVSS warning on synthetic source input ([arXiv 2203.13339](https://arxiv.org/pdf/2203.13339)) [S] | Re-attribute the risk. Add input-side rate/temperature variation. List the true audio-KPI drivers (S2-11) |
| 7 | Missing telephony 8/16 kHz → 24 kHz handling | Not addressed | **Wrong as stated, right about the gaps.** C4 (L280) and E-SP (L574) already specify 24 → 8 kHz, band-pass, G.711/GSM/AMR-NB/Opus, loss, and back to 24 kHz. Missing: a 16 kHz wideband branch, a prose resampling step for real corpora, and the bandwidth shortcut | Measured: the 8 kHz resample alone imposes the 3.4–4 kHz upper edge; no codec removes energy below 300 Hz [L]. phonesim's profiles ([phonesim](https://github.com/deepmark/phonesim)) [V] | C4 rewrite (S2-8, S2-9) |
| 8 | Add observation adding at 10–20% | Already present before the critique | **Already covered; the fixed ratio is unsupported.** E-SP has "enhanced + 0.1–0.3 × noisy" (L574). Iwamoto et al. show SAR rising monotonically with the weight, not a fixed 10–20%. The report's formula is also not convex | [arXiv 2201.06685](https://arxiv.org/abs/2201.06685) [S]; later work tunes the weight on validation data ([arXiv 2509.21833](https://arxiv.org/pdf/2509.21833)) [S] | Convex form, weight tuned on a real dev set, and "no denoising" as the default (S3) |
| 9 | The parser has 9 bugs | **Fixed in 6b06d3e** (the status box lists seven bug classes, counting the import fix) | **Valid against the old version.** The audit then found 8 *further* failure families in the rewrite. These were found by the audit and **fixed in the follow-up commit 986559a; re-run `python prototypes/test_hindi_number_parser.py`** (57 phrases + 4 tests pass) [L] | Probe script and results in the consistency note | Re-run the tests; see the parser box below |
| 10 | E-ML prefix translation has prohibitive LLM latency | Not addressed | **Wrong as a latency objection.** Regime (2) is an offline simulation on the Hindi playback timeline with cached calls, so API latency never enters the plotted lag. Two real neighbouring faults: it is not labelled offline (S1-5), and "a few hundred calls" should be about 1,000–3,000 | The mini PS asks for a simulation with fixed or TTS-measured word durations ([Mini PS Part E](file:///home/user/interiit_minips/Inter%20IIT%20Tech%20Meet%2015.0_%20Sarvam.ai%20Mini%20PS.pdf)) [V]; call count [E] | Add the offline label and the corrected count |
| 11 | torchaudio 2.11 needs torchcodec + FFmpeg, which breaks a slim Docker image | Already in the report (L175, L377) | **Correct but low impact.** `torchaudio.load` raises `ImportError` without torchcodec, and the wheel declares no Requires-Dist. But the server receives raw `pcm_s16le` over WebSocket and decodes no files | torchaudio 2.11.0 wheel; torchcodec 0.17.0 links `libavcodec.so.61` and tries FFmpeg majors 9→4 [V] | One line in D0/E-Dev: "the serving image does no file I/O; data tools use soundfile" |

**Parser: found by the audit, fixed in the follow-up commit.** Re-run `prototypes/test_hindi_number_parser.py`. The failure families were:
- सौ or हज़ार before करोड़ truncated the amount: पाँच सौ करोड़ → 500, off by 10⁷.
- सवा सौ → 100, and सवा लाख/सवा करोड़ → nothing.
- करोड without the nukta was not a multiplier, so पाँच करोड → 5.
- Attached quotes and brackets hid numbers.
- वन/टू homographs gave false positives (वन विभाग → AMT 1).
- पौने एक बजे → 0:45 instead of 12:45.
- Distributive दो दो सौ split into AMBIG + 100.

All now return correct values [L]. The one residual found while writing this report, "दो दशमलव पाँच करोड़" yielding two wrong amounts (2 and 50,000,000), is also fixed: decimals are now parsed ("दो दशमलव पाँच करोड़" → 25000000), and the case is in the test suite [L].

## New issues the audit found, by severity

The severity scale:
- **S1:** misleads in the interview, or breaks or omits a PS rule.
- **S2:** an inconsistency or validity gap a careful reader will catch.
- **S3:** a wording, typography or citation slip.

The proposed text is ready to paste.

### S1 — fix before submitting

| ID | Location | Problem | Proposed fix text |
|---|---|---|---|
| S1-1 | Title L1; L3; B2 L151; Conclusion L595 | Under the report's own A5/B3 rules, "my old bank account" cannot precede "करवा दिया" when the target is the PS gloss "Yesterday I got my old bank account closed". The causative "got" is itself a small anticipation, because causation is settled only at करवा (word 8). The title also says only the verb is held, while the body holds negations, amounts, names and repairs | **Title:** "Hold the commitment, not the sentence" (or keep the title and add the subtitle "verbs, polarity, amounts, names and repairs wait; time, topic and context go early"). **L3:** "speak 'Yesterday I…' at once, hold 'got my old bank account closed' until 'करवा दिया' is heard, and hold an amount until the repair's unit word plus one beat of silence. If B2's experiment shows the naturalness cost is acceptable, a left-dislocated target ('Yesterday, my old bank account — I got it closed') lets the object go early too." **B2 add:** "The causative defers the lexical verb, but its auxiliary 'got' still waits for 'करवा' under B3; only left dislocation lets the object precede the Hindi verb." **L595:** "waits only at the verb" → "waits only from the verb onwards" |
| S1-2 | Parser (status box L10, §D6) | Eight failure families (above) | **Fixed in 986559a.** Re-run the tests. Optionally reword "Number parser (fixed)" to "rewritten; 59 test phrases pass; known limits in §D6" |
| S1-3 | Appendix L603–640 | The mini PS says to "List every source you used". The appendix has 36 rows, but the body cites about 99 distinct sources, 57 of them absent from the appendix (Simul-COMET, TransLLaMa, NaturalFlow, WhisperX, IndicXlit, DeepFilterNet3, torchmetrics, PHINC, and others) | Generate the 4-page PDF's appendix table automatically from every link the PDF actually cites. In NEXT_STEPS Step 4 add: "and make sure every source cited in the PDF is in the table" |
| S1-4 | C intro L213; C3 table; C6 diagram | PS §3 Part A and §5.2 rules are missing, and the mid-term gate gives zero points for breaking them | Append to L213: "Each clip is at most 60 s. No source field may be empty. Synthetic records give `source.model` and `source.voice` (the generating model and voice), with `source.start` = 0 and `source.end` = the clip duration when the file is the clip (PS §5.2). Everything is declared in `model_card.md` and `data_card.md`, and the data pipeline must be re-runnable on request, since a top-five team that cannot reproduce it is disqualified (PS §10)." Add a "§5.2 schema" row to C3 |
| S1-5 | E-ML regime (2), L524 | Uses an external LLM on gold transcripts with no "offline" label, so it reads like a banned cascade | After regime (2) add: "Like (3), this is an offline simulation on gold transcripts, run to measure how prefix translation fails; an external LLM translating the input would be a banned cascade in the submitted system." Add to regime (1): "offline re-timing of a full-sentence translation" |
| S1-6 | L3, L16, L75, L597 | Overclaims an interviewer can puncture. Intent and entity points can be lost to mistranslation too, and KPIs are relative to other teams and the base (PS §9.1) | L3/L16: "36 of the 70 KPI points are for intent and entities, which committing too early puts at risk". L75: "risks an entity error, the most heavily weighted kind". L3/L597: "is off by one against STACL if i is 1-indexed" |
| S1-7 | B6 (4), L197; A4 L77 | The PAD/EPAD bias is gated "before the clause verb", but in a Moshi/Hibiki model the text stream is the *English* target. A Hindi verb signal would need an outside recogniser, which is a cascade (the report's own rule 8) | "(4) sweep a PAD/EPAD logit bias β, gated off while the model's own text stream is inside an unfinished number; gate on the Hindi verb only if the released base exposes a source-text stream, and confirm with the organisers" |
| S1-8 | B2 L151, C5 L293, D5 L437, C6 L321 | `en_alt` is "fluent" in B2 and C6 but "literal with the repair" in C5 and D5 | One definition: "`en_text` = chunk-monotonic + corrected meaning; `en_alt[0]` = fluent + corrected meaning. The literal, repair-preserving rendering ('five lakh… no no, five crore') is a Part D deliverable only (or an explicitly labelled `en_alt[1]`) and never a training target." C5's guard becomes "fluent and monotonic targets both corrected; literal-with-repair never in `en_text`" |
| S1-9 | B1 L137; C1 R1 L221; L3 | R1 allocates **105 h** to IndicVoices Hindi. The v1 paper's Table 6 gives **65 h train / 287 speakers and 5 h test / 63 speakers**, and the Lahaja paper also says 65 h ([arXiv 2403.01926](https://arxiv.org/pdf/2403.01926); [arXiv 2408.11440](https://arxiv.org/html/2408.11440v1)) [S]. After dropping the other speaker's segments and filtering, about 50–65 h may remain [E]. The report already flags the count as its biggest unknown (L3, L341), but the 200 h table presents 105 h as a plan | C1 R1: "IndicVoices Hindi, **≤ 105 h, planned at 55–65 h** pending the HF-card count (v1 paper: 65 h train + 5 h test); per-speaker segments only." Add R3b "Kathbath (CC0, 150 h read) 20–40 h". Add "Vaani Hindi only if its official HF card says CC BY 4.0 (a derived subset says CC BY-NC)" ([Vaani](https://vaani.iisc.ac.in/); [arXiv 2603.28714](https://arxiv.org/html/2603.28714v2)) [S]. Say the real share may fall toward ~50% if the count is short |

### S2 — inconsistencies and validity gaps

| ID | Location | Problem | Proposed fix |
|---|---|---|---|
| S2-1 | B1 L137 vs C1 L223–224 | Kathbath is "Common Voice or Kathbath 3 h" in B1 but an optional add-on to R3 in C1 | Use one assignment in both places (with S1-9, Kathbath becomes R3b) |
| S2-2 | C1 L225 vs B4 L185, C7 L341 | 50 h = 25% entity-TTS, but the sweep stops at 0/5/10/20% | "Sweep 0/10/20/30%; 25% is a prior to be revised by the sweep" |
| S2-3 | A4 L75, B3 L165, E-ML L500, C5 L297 | δ is a 200–500 ms look-ahead in A4, swept over {0.5, 1, 2} s in B3, and a compute delay in E-ML | Rename the E-ML compute delay *c*; sweep δ ∈ {0.25, 0.5, 1, 2} s |
| S2-4 | A3 L54 table | The report's rows 1–10 clash with the PS's rule numbers ("rule 3, 4, 5 or 6") | Label the rows R1–R10 and add a "PS rule" column |
| S2-5 | D1 L397 | `ffprobe … clips/24k/*.wav` exits 1 with two or more files [L] | `for f in clips/24k/*.wav; do printf '%s,' "$f"; ffprobe -v error -show_entries format=duration -of csv=p=0 "$f"; done` |
| S2-6 | D3 `align_mms.py` L409–426 | Words with an empty romanisation (such as "98765") are silently dropped, although the comment promises interpolation, and the "every word timed" assert would then fail [L] | Add the interpolation (split the gap between neighbouring kept words), or reject digits during transcript validation |
| S2-7 | B5 L193 | The Hibiki-Zero rehearsal assumes moshi-finetune works. Its last commit (Jul 2025) predates Hibiki-Zero and has no Hibiki loader; Hibiki uses conditioning tensors. "8 GB … 12 GB" is the *inference* VRAM figure, and there is no Hindi input [V] | "Expect to patch the loader and conditioning; the VRAM figure is for inference; the rehearsal tests plumbing, not Hindi quality" ([moshi run_inference.py](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/run_inference.py)) |
| S2-8 | C4 L280; C6 L308 | **Bandwidth shortcut.** The real buckets (IndicVoices, MUCS at 16 kHz, FLEURS; about 122 of 125 real hours) are empty above 8 kHz after upsampling, while the TTS buckets are full-band, so bandwidth labels "synthetic" almost perfectly. There is also no prose step for resampling real corpora [E/V] | Add to C4: "Resample every real source to 24 kHz mono s16le with soxr, recording the native rate in `data_card.md`. Band-limit at least 30% of synthetic clips through 16 kHz (random 4–8 kHz low-pass) so bandwidth cannot identify the source. Do not resample at inference: feed the 24 kHz input unchanged." |
| S2-9 | C4 L280; E-SP L574 | No wideband branch. A fixed 300–3400 Hz band-pass suits PSTN only; the low edge is about 100 Hz on mobile. Codec lag shifts every word time | "Add a 16 kHz branch (G.722, AMR-WB, Opus-WB). Use a randomised 2nd-order high-pass (250–320 Hz PSTN, ~100 Hz mobile, 50–80 Hz wideband) instead of a fixed band-pass. Randomise the resampler kernel. Measure and remove each codec's delay by cross-correlation. Note that AMR needs an AMR-enabled ffmpeg, and that SpeechBrain `CodecAugment` and `torchaudio.io` are broken on torchaudio ≥ 2.9" [V] |
| S2-10 | C4 L280 | Recommends 0.9/1.0/1.1 speed perturbation (an ASR result), which breaks Hindi word times and English placement in timed S2ST pairs | "Apply tempo change only at data-build time, before computing timings, or skip it" |
| S2-11 | C4/C5, B6 | The audio-quality KPI's real drivers are absent | List them: silence-insertion hard cuts (insert only at natural ≥ 50–100 ms gaps, with 5–10 ms fades, or synthesise chunk-wise with Bulbul); Bulbul pace and temperature; English alignment accuracy; audio-token temperature; stream underruns; causal loudness with a limiter; no time-stretch. Noisy source with clean target also keeps the output clean [E]. Hibiki's silence insertion causes "hard cuts" ([pith summary of 2502.03382](https://pith.science/paper/2502.03382)) [S] |
| S2-12 | C4 L280 | Under "Only your Part A dataset may be used for training", MUSAN, DNS or babble made of other people's speech is a grey area. ESC-50 is CC BY-NC ([ESC-50 LICENSE](https://github.com/karolpiczak/ESC-50/blob/master/LICENSE)) [V] | "Harvest noise from VAD-negative stretches of Part A recordings; build babble from 3–8 Part A speakers; use self-recorded or generated noise and pyroomacoustics or OpenSLR 26/28 (Apache 2.0) RIRs; ask the organisers before using MUSAN/DNS" |
| S2-13 | A4 step 3; B3 step 3 (L165) | Holding until "the Hindi verb group" misses intent cues that arrive at or after it. Polar questions (47 of 50 carried final rising intonation, and only 6 used क्या, [Syed & Dash](https://glowlinguistics.org/asia11/wp-content/uploads/sites/3/2017/11/v1-syed-dash.pdf) [S]), tags such as "…ना?", post-verbal or emphatic नहीं, and कल (yesterday or tomorrow) resolved by tense | "Negators and auxiliaries wait for the Hindi verb group plus one token or the clause boundary. Render intonation-only questions and tags as declarative + 'right?'. Voice 'yesterday/tomorrow' for कल only after the ergative -ने or the tense auxiliary. Keep a से-intermediary in causatives; use the passive only when the Hindi is passive or agentless" |
| S2-14 | B2 L151 | The default "Yesterday I… [~1.6 s] …got my old bank account closed" creates the mid-clause pause that listeners penalise. Left dislocation fills the pause with content | Allow left dislocation more freely than "rare" for an en-IN voice, pending a 20-pair AB listening test. Add the experiment "fluent words + safe placement vs chunk-monotonic words + safe placement", which tests whether timing alone captures most of the lag gain |
| S2-15 | C2 L241–253; D2 L402; C6 L316 | Saaras `verbatim` returns English words in **Latin** ("…नौ आठ चार zero नौ…") ([Saaras docs](https://docs.sarvam.ai/api/getting-started/models/saaras.md)) [S], but Level 1 requires Devanagari. IndicConformer trains on *normalised* targets, so it is not a verbatim oracle [V] | Add a Latin→Devanagari transliteration step, plus an assert that no Latin remains |
| S2-16 | C1 L230 | The "IndicVoices-R … adds no hours [V]" tag sits on an interpretation | "[V] for provenance; the hour treatment is our reading of PS §3, so ask the organisers" |
| S2-17 | Whole report | Bulbul **V4** went live alongside V3 (changelog 6 Oct 2026). In V4, "Shubh" is renamed "Shubh – Hinglish Ad Voice" ([Sarvam changelog](https://docs.sarvam.ai/changelog/2026/10/6.md)) [S]. An accidental V4 call yields non-compliant targets | "Pin model `bulbul:v3`, voice `shubh`, `en-IN`, 24 kHz in code, and assert them in the manifest builder" |
| S2-18 | A2/E metrics | The metric set is one generation behind. IWSLT 2026 assigns latency regimes by LongYAAL and reports XCOMET-XL ([IWSLT 2026 findings](https://aclanthology.org/2026.iwslt-1.39.pdf)) [S]. COMPASS finds standalone MOS predictors do not predict listener preference ([arXiv 2606.03241](https://arxiv.org/pdf/2606.03241)) [S] | Report YAAL next to AL/LAAL, and xCOMET next to COMET/chrF. Add an internal-pause metric (count and duration of silences > ~250 ms). Treat UTMOS as a soft proxy |

### S3 — small slips (one line each)

These cost little to fix and remove "your report says two things" moments:
- **§ citation.** L37 cites PS §5.1 for the 10–60 s session length; that is §4.
- **Quotation.** L69 shows a paraphrase in quotation marks ("25,000 and 25k both count"). Quote it exactly.
- **G3.** L33 abridges G3. Add "or the final dataset no longer meets Part A".
- **Percentages.** L228 should read 62.5 / 32.5 / 5.
- **CER band.** C3 L263 leaves 10–15% CER undefined. Add "10–15%: regenerate; > 15%: drop after 3 tries". Tag the "4–5% CER" claim at L259 as [S] and name the system (Saarika v2.5).
- **Name match.** It uses `partial_ratio ≥ 85` in D6 but edit distance ≤ 0.2 in the C3 gate. Unify the two, or explain the difference.
- **Thresholds.** The 100 ms (Hindi) vs 80 ms (English) aligner flags need one sentence of explanation.
- **Cross-reference.** B3 L175 points to E-SP about `torchaudio.load`. Change it to "the removal of `torchaudio.backend` breaks DeepFilterNet".
- **D7 headings.** L481–484 use D2/D3/D4. Use the section numbers D3/D5/D6.
- **Package versions.** L377's "all current on PyPI" is false for torch (2.14.1). Write "torch pinned to 2.11 to match torchaudio 2.11.0", and add `--index-url https://download.pytorch.org/whl/cpu`, because the default wheel pulls CUDA 13 and a 4.7 GB venv [L].
- **Compute formula.** Show the 9 × 12.5 tokens-per-second formula next to the 1.9 H100-h figure.
- **Display words.** The D3 split regex keeps "…". Strip `[।॥?,.!…;:"“”'()]`.
- **Observation adding.** Write it in convex form, (1 − α)·enhanced + α·noisy with α tuned over {0.1, 0.2, 0.3, 0.5}, and make "no denoising at inference" the default.
- **Untagged claims.** Tag the B1 licence exclusions, the "20–50 voices" estimate, "50–100× below LoRA rates", "quantisation is not a latency lever", the "~18 s dropped words" claim and the Sarvam mayura/translate capabilities. State in the L5 legend that tag suffixes are free text.
- **README.** The folder tree omits `research_notes/Sarvam mini PS final audit/`.
- **NEXT_STEPS Step 5.** Replace "Saarika v2.5" with "Saaras v3 (~19.3%, 10-language aggregate)", and "within-20 ms" with "within-25 ms".
- **D6 table.** Its wording on single English digit words and "three or more small numbers" is loose.
- **ffmpeg filters.** The `highpass`/`lowpass` filters default to 2 poles, so call the E-SP telephone band "approximate".
- **Licences.** Note that the MMS_FA and ctc-forced-aligner default weights are CC-BY-NC. A cautious team can make NeMo NFA + IndicConformer (MIT) the production aligner and keep MMS_FA for Part D and cross-checks ([ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)) [V].

## Technology currency in October 2026: update six names, switch nothing structural

The core architecture call stands. No open-weight system is at once Hindi→English, streaming and speech-out. The Moshi-family duplex models (Hibiki, Hibiki-Zero) are the only structurally relevant open bases, and the 80 ms / 24 kHz framing is a Mimi signature ([moshi README](https://github.com/kyutai-labs/moshi/blob/main/README.md)) [V]. Hibiki-Zero (Feb 2026, ICML 2026 oral) is still the right rehearsal proxy. It is 3B parameters, built on `moshi>=0.2.13`, and accepts `--lora-weight` [V]. It covers Fr/Es/Pt/De→En with no Hindi, and its weights are CC BY-NC-SA ([hibiki-zero](https://github.com/kyutai-labs/hibiki-zero); [arXiv 2602.11072](https://arxiv.org/html/2602.11072v1)) [S].

The frontier moved in two directions without displacing this choice:
- **Closed duplex systems:** Seed LiveInterpret 2.0, Zh↔En only, about 3 s latency ([arXiv 2507.17527](https://arxiv.org/html/2507.17527v2)) [S]; Google's real-time S2ST, about 2 s and trained on time-synchronised data ([Google Research](https://research.google/blog/real-time-speech-to-speech-translation/)) [S]; Gemini 3.5 Live Translate ([multilingual.com](https://multilingual.com/google-launches-gemini-3-5-live-translate-real-time-speech-capabilities/)) [S].
- **Speech-LLM thinker–talker designs:** Qwen3-LiveTranslate (API only), SimulS2ST-Omni ([arXiv 2607.19810](https://arxiv.org/pdf/2607.19810)) [S] and SimulS2S-LLM.

Both families independently confirm the report's thesis that *time-aligned training data sets the policy*.

| Component | Report uses | Newer as of Oct 2026 | Switch? |
|---|---|---|---|
| Latency-policy learning | Placement rule in training pairs | Hibiki-Zero's stage-2 **GRPO** RL optimises latency with a BLEU reward ([ICML oral](https://icml.cc/virtual/2026/oral/71089)) [S]. Allowed: reward models are permitted, and rollouts add no records (PS FAQ) [V] | **Add** as a post-reveal option in B5/B6; keep the placement rule as the primary lever |
| Sarvam ASR | Saaras v3 (L402, L624) | **Saaras V4**, announced 24 Aug 2026, GA 2 Sep: an audio encoder plus a 3B hybrid state-space LM, five modes including verbatim ([Sarvam blog](https://www.sarvam.ai/blogs/introducing-saaras-v4)) [S] | **Update the name**, test V4 verbatim on the Part D clips, and transliterate Latin tokens |
| Sarvam TTS | Bulbul v3 `shubh` (PS-mandated) | **Bulbul V4** live; v3 called "stable"; no v3 deprecation found ([changelog](https://docs.sarvam.ai/changelog/2026/10/6.md)) [S] | **Do not switch**; pin v3 explicitly |
| Second-opinion ASR | Omnilingual (< 40 s caveat), IndicConformer | Omnilingual **v2** CTC/LLM checkpoints plus `omniASR_LLM_Unlimited_*_v2` for unlimited length; Apache-2.0 ([omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr)) [V]. Nemotron 3.5 ASR (600M streaming, hi-IN, Jun 2026) [S]. Qwen3-ASR lists Hindi but publishes no per-language WER [V] | **Switch to Omnilingual v2**; make the 40 s caveat model-specific. Others optional |
| Forced alignment | torchaudio 2.11 MMS_FA + uroman | Still the newest; `MMS_FA` and `forced_align` exist in the 2.11.0 wheel [V]. Qwen3-ForcedAligner still covers 11 languages without Hindi [V]. No 2026 aligner reports Hindi boundary accuracy [S] | **Keep** |
| MT | LLM + IndicTrans2-200M | No IndicTrans3; the last update was Jan 2025 RoPE long-context variants ([IndicTrans2](https://github.com/AI4Bharat/IndicTrans2)) [V]. **Sarvam-Translate** has open weights, built on Gemma3-4B ([Sarvam blog](https://www.sarvam.ai/blogs/sarvam-translate)) [S] | **Add** Sarvam-Translate as the second open system |
| Open Hindi TTS | Indic Parler-TTS, IndicF5, Veena, Svara | **Chatterbox Multilingual V3** with a Hindi pack, zero-shot cloning ([chatterbox](https://github.com/resemble-ai/chatterbox)) [V]; its MIT licence is confirmed only via a secondary source [S]. Kokoro (Apache, two Hindi voices, no cloning) [V]. Maya1 succeeds Veena [S]. CosyVoice3 has no Hindi; Fish/OpenAudio stay excluded [V] | **Add Chatterbox** for voice diversity, behind the two-ASR gate |
| Data | IndicVoices-led | **Vaani**: about 31K h of audio, about 2.0K h transcribed, licence conflicting ([arXiv 2603.28714](https://arxiv.org/html/2603.28714v2)) [S]. No new clean, real, code-mixed Hindi corpus [S] | Vaani **only if** the official card says CC BY 4.0 |
| Telephone augmentation | ffmpeg chain, phonesim | phonesim 0.2.0 (MIT) [V]. SpeechBrain `CodecAugment` and `torchaudio.io` are **broken** on torchaudio ≥ 2.9 [V] | Use phonesim or the ffmpeg CLI only |
| Metrics | AL/LAAL/Start/EndOffset/ATD, ASR-BLEU/COMET, UTMOS | **YAAL/LongYAAL** + XCOMET-XL (IWSLT 2026) [S]. Original UTMOS generalises across languages better than UTMOSv2 ([arXiv 2509.04830](https://arxiv.org/pdf/2509.04830)) [S] | **Add** YAAL, xCOMET and a pause metric |
| Packages | torch 2.11.*, torchaudio 2.11.0 | torch 2.14.1 exists, but torchaudio's latest is 2.11.0. torchcodec is at 0.17.0 [V] | **Keep the pin**; fix the wording |

## Feasibility and cost: the budget fits, the Bulbul rate limit does not

Vendor prices below come from search extracts of Sarvam's pricing pages ([pricing](https://docs.sarvam.ai/api/getting-started/pricing.md); [rate limits](https://docs.sarvam.ai/api/getting-started/ratelimits)) and from third-party snapshots of Modal's H100 rate of $0.001097/s ≈ $3.95/h ([Beam](https://www.beam.cloud/blog/modal-pricing-explained)) [S]. Confirm both in the dashboards. The workload model assumes 140 wpm Hindi, 12.5-word sentences and about 134k sentences in 200 h, so every figure is [E].

| Item | Size | Compute / time | Cost | Verdict |
|---|---|---|---|---|
| B3 MT-likelihood placement (MADLAD-3B) | 1.8 M prefix passes, 81 M tokens | 0.5–3 H100-h | **$2–11** (critique's accounting: $27–161) | Feasible; can run before the mid-term from text alone |
| B7 counterfactual continuations | 150 sentences / 2,000 / full 134k | minutes / ~1 h / 41–102 H100-h | <$1 / $2–6 / **$160–400** | Calibration subset only; full run only after the mid-term, and only if it clears the ≥ 30% threshold |
| Bulbul v3 English rendering | 8.6 M chars per pass (6.3–12.7 M) | **74 h of continuous calls per pass** at Starter's 30 RPM; 11 h at Pro's 200 RPM | ₹26k ≈ **$270–300** per pass at ₹30/10k chars (beta) | Money is fine for about two passes; time is the constraint. Ask for a higher limit, or pre-render before the mid-term if the organisers allow it. Three full paces would use 80–90% of the credit |
| Whole Sarvam plan (central) | Rendering + Hindi voice + Saaras + Mayura + LLM + reserve | — | **≈ $690 of $1,000** | Sufficient if paces × temperatures are not all rendered |
| Synthetic Hindi TTS | ~90 h incl. regenerations | 10–20 H100-h | $40–80 | Feasible |
| Two-ASR round trip + Hindi alignment | 90 h synthetic, 190 h all | 5–10 H100-h | $20–40 + ~$30 Saaras | Feasible; run MMS_FA on Modal, not on a laptop (28–95 CPU-h) |
| LoRA fine-tune | 200 h epoch | 1.875 H100-h/epoch | $7.41/epoch; **plan $40–60 per run** | Feasible; architecture unknown until the reveal |
| Protocol-faithful eval | 200 sessions | ≈ 1.9 H100-h per checkpoint | ≈ $7.7 | Eval time, not training, dominates after the mid-term |
| Modal totals | — | pre-mid-term ~30–70 H100-h; post ~85–120 | **$150–300 + $370–540** | $150–450 slack; cannot fund full B7 *and* many full fine-tunes |
| Part D on a laptop CPU | 6 clips | MMS_FA 27–90 s; IndicTrans2 10–55 s; `run_clip.py` **30–90 s** per new clip | $0 | Feasible; warm the models before the interview |
| E-ML regime (2) | 6 clips × k ∈ {1,2,3,5,7,∞} | ~1,000–3,000 calls; 15–45 min sequential | **< $1** | Feasible from cache; in the interview, show regime (2) cached |
| Human effort | — | Mini PS **25–40 person-h**; main data engine **50–85 person-h** | — | The translation review queue can blow up to about 100 h at 10% routing, so set thresholds by queue capacity |

## A prioritised fix checklist

The order follows interview exposure first, then PS-rule risk, then effort. Items 1–9 take an afternoon. Items 10–17 take a day.

| # | Do this | Where | Est. effort |
|---|---|---|---|
| 1 | Re-run `python prototypes/test_hindi_number_parser.py` (expect "59 phrases and 4 tests passed") | prototypes | 15 min |
| 2 | Retitle; rewrite the L3 example and the "36 points lost" sentence; fix "risks 18 points" and "off by one" (S1-1, S1-6) | L1, L3, L16, L75, L595, L597 | 30 min |
| 3 | Add the B2 causative/left-dislocation sentence and the widened hold rule: questions, tags, कल tense, post-verbal नहीं, से-intermediary (S1-1, S2-13) | B2 L151, B3 L165, A4 | 30 min |
| 4 | Add the §5.2/§3/§10 rules to C intro, C3 and C6 (S1-4) | L213, C3, C6 | 20 min |
| 5 | Label E-ML regimes (1) and (2) offline; correct the call count (S1-5) | L524 | 10 min |
| 6 | Single `en_alt` definition everywhere (S1-8) | B2, C5, D5, C6 | 15 min |
| 7 | Re-gate the PAD bias on numbers only (S1-7) | B6 L197, A4 L77 | 5 min |
| 8 | Re-plan R1 as 55–65 h pending the count; add Kathbath R3b; set the Vaani condition; fix the Kathbath bucket (S1-9, S2-1) | B1, C1, L3 | 30 min |
| 9 | Pin `bulbul:v3`/`shubh` explicitly; update Saaras v3 → V4 and add Latin→Devanagari transliteration (S2-15, S2-17) | C2, C6, D2, appendix row 20 | 20 min |
| 10 | Rewrite C4: real-source resampling, bandwidth-shortcut fix, wideband branch, randomised high-pass and resampler, codec-delay compensation, tempo only at build time, Part A-only noise (S2-8 to S2-10, S2-12) | C4, E-SP | 1 h |
| 11 | Re-attribute the TTS-prosody risk; list the audio-KPI drivers; make the observation-adding formula convex with "no denoise" as default (S2-11, S3) | C4/C5, E-SP | 30 min |
| 12 | Fix the `ffprobe` loop and add digit interpolation in `align_mms.py` (S2-5, S2-6) | D1, D3 | 30 min |
| 13 | Add the B5 rehearsal caveat and the GRPO option (S2-7) | B5, B6 | 10 min |
| 14 | Add YAAL, xCOMET and a pause metric (S2-18) | A2, E-ML, D7 | 15 min |
| 15 | Reconcile δ, the entity-TTS sweep, rule labels and the D7 headings (S2-2 to S2-4) | as listed | 30 min |
| 16 | Build the appendix from every link cited in the 4-page PDF (S1-3) | Appendix; NEXT_STEPS Step 4 | 1 h |
| 17 | Apply the S3 list, the README tree and the NEXT_STEPS mismatches | throughout | 1 h |

## What to say in the interview on the contested points

| If asked | Say |
|---|---|
| "English can't put the object before the verb, so isn't 'hold the verb' a fallacy?" | "Right for canonical clauses, which is why I hold the *commitment*. Subject, time and settled context go early; the verb and object go as one chunk after 'करवा दिया'. Left dislocation ('my old bank account — I got it closed') is my tool for long objects, and Indian English uses it far more than British English." |
| "Why not the passive, like human interpreters?" | "Interpreters optimise fluency at low lag. This KPI scores who did what to whom, and 'the account was closed' deletes the causer of करवाया. I use the passive only when the Hindi itself is passive or agentless." |
| "Doesn't the word-timing rule prove the base is Hibiki?" | "No. Hibiki-Zero's inference code still emits a padded, frame-aligned text stream, so it needs English word times too. It dropped Hindi↔English word alignment, not target timing. I say 'most likely Hibiki-family', and my pair store keeps both word- and sentence-level maps." |
| "Would you say 'fifty million'?" | "No. The dataset must keep lakh/crore, the FAQ only shows same-unit equivalence, and exact-match entity scorers exist, so 'crore' can never score worse. I never append a conversion, because that is reading out an alternative the speaker did not offer." |
| "Isn't MT-likelihood alignment too expensive?" | "One teacher-forced pass scores every target token, so it is about 1.8 M passes for 200 h, a few H100-hours and under $11. The real cost is engineering and validating it on Hindi, which is why I hand-mark 50 sentences first." |
| "Won't flat Hindi TTS hurt audio quality?" | "Audio quality is rated on the English, which is Bulbul shubh by rule. Flat Hindi hurts generalisation to real speech, so I keep real speech the majority and vary TTS rate, temperature and prompts. The audio-KPI risks I manage are hard cuts from silence insertion, pace, and stream underruns." |
| "How do you handle 8 kHz phone audio?" | "The input is already 24 kHz, so I never resample at inference. Robustness comes from training-time narrowband and wideband codec chains. I band-limit synthetic clips too, so bandwidth can't tell the model 'this is TTS'." |
| "Do you denoise?" | "Not by default. A denoiser adds 10–40 ms and artifacts the model never saw in training. I'd enable an SNR-gated denoiser with observation adding only if a real noisy dev set shows a KPI gain." |
| "Your E-ML uses an LLM: isn't that a cascade?" | "It's an offline simulation on gold transcripts, to measure how prefix translation fails. Nothing external translates the input in the submitted system." |
| "IndicVoices conversations have two speakers." | "Yes, it's a single telephone bridge. I keep only the metadata speaker's segments, as AI4Bharat's own chunker does, and screen for backchannels. The bigger issue is that Hindi is about 65 h train in the v1 paper, so I planned a Kathbath top-up." |

## Conclusion

The audit changes where the risk sits. The student feared the technical core: the parser, the base-model inference and the cost of alignment. That core held up, the parser after two rounds of fixes. The remaining danger lies in the first page and in the PS's bookkeeping rules. An interviewer who reads only the title and the opening example will find the report contradicting its own A5 table. A mid-term auditor who reads only Part C will find no `source.model`, no 60 s cap and an IndicVoices hour count the primary paper does not support. Both are cheap to fix, and both matter more than any of the critique's computational objections.

The critique's lasting value lies in what it brushed against without naming. Under the critique of "hold the verb" lies an intent class the hold rule misses: questions and tags marked only at the clause end. The telephony point leads to a bandwidth shortcut that could quietly split real and synthetic data into two training regimes. And Bulbul V4's rename of Shubh, which arrived after the report was written, is the most likely way a compliant plan could produce non-compliant targets. Fix those three and the report shifts from "plausible plan" to "plan that anticipated its own failure modes", which is the case to make in the interview.
