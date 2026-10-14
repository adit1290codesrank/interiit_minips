# Consistency, validity and code audit of the Sarvam mini PS report and repository

Scope: `reports/Sarvam mini PS research.md` (the "report"; 637 lines, 16,865 words), `README.md`, `NEXT_STEPS.md` (untracked, excluded through `.git/info/exclude`), `prototypes/`, checked against `Sarvam.pdf` (the "PS") and the mini PS PDF (both extracted with `pdftotext -layout`). The report text audited is the one at commit 6b06d3e. The four later commits (1c1d2d2…1e63594) only add files under `research_notes/Sarvam mini PS final audit/`, and `git diff --stat 6b06d3e HEAD` shows no change to the report, README or prototypes. This audit changed no repository file. All scratch code is in `/tmp/claude-0/-home-user-interiit-minips/c41df105-99ef-5d9b-b959-5c09e7324606/scratchpad/audit/` (the "scratch dir").

Line references "L*n*" point to the report unless stated otherwise. Severity:
- **S1:** would mislead in the interview, or breaks or omits a PS rule.
- **S2:** an internal inconsistency a careful reader will notice.
- **S3:** minor wording, typography or citation slips.

Links to `file://` paths are the evidence: the repository files themselves, or a scratch script whose output is quoted.

---

## Overall: what must be fixed, ranked by severity

### Takeaway
All the headline numbers recompute exactly: AL 2.625 and 3.6875, the k = 1..9 table, oracle AL 41/16, LAAL 3.3, DAL 3, AP 51/72, the 787.5 → 937.5 ms time-domain example, 643 frames / 51.44 s, 3,840-byte frames, the 200 h split, the points tables and the compute budget. The wait-k code and the echo server also run correctly. The serious problems are elsewhere:
1. The title and opening example contradict the report's own A5 and B3 rules.
2. The number parser, described as "fixed", silently returns wrong values for "सौ/हज़ार + करोड़" amounts, "सवा सौ / सवा लाख", and करोड spelled without the nukta.
3. The 36-row source appendix does not list "every source used", as the mini PS requires; about 55 more sources are cited in the body.
4. Part C omits three Part A format rules: clips at most 60 s, `source.model`/`source.voice` for synthetic records, and synthetic start/end values.
5. Regime (2) of E-ML is not labelled as offline analysis.

### Cited Findings (master list, most severe first; details in the sections below)

**S1-1. The title and opening example contradict A5 and B3.**
- **What the report says:** the opening paragraph (L3) says to "speak 'Yesterday I… my old bank account' immediately, but hold the English verb".
- **The conflict:** the target used everywhere, the PS's own gloss "Yesterday I got my old bank account closed", puts "got" before "my old bank account". A5 marks "got" as "not heard: anticipation" (L89). B3 sends "negators and auxiliaries … to the Hindi verb group" and makes placement monotone with `cummax` (L165). Under the report's own rule, therefore, "my old bank account" cannot be spoken before "करवा दिया". A5's oracle in fact holds "got my old bank account closed" until word 8 (L118).
- **Title:** "Hold the verb, speak everything else" also conflicts with the report's thesis that verbs, negations, amounts, names and repairs are all held (L3: "holds only verbs, negations and entity spans"; A4).
- **Fix:** see the title/thesis section below.

**S1-2. Number-parser bugs (new, not in the "Known limits").** These were found with an adversarial probe, [probe_parser.py](file:///tmp/claude-0/-home-user-interiit-minips/c41df105-99ef-5d9b-b959-5c09e7324606/scratchpad/audit/probe_parser.py).
- **"सौ" or "हज़ार" before a larger unit is truncated:**
  - पाँच सौ करोड़ → `AMT 500` (should be 5,000,000,000)
  - सौ करोड़ → `100`
  - एक सौ करोड़ → `100`
  - डेढ़ सौ करोड़ → `150`
  - एक हज़ार करोड़ → `1000`
- **सवा followed by a multiplier fails:** सवा सौ → `AMT 100` (should be 125); सवा लाख and सवा करोड़ → no entity at all.
- **The spelling करोड (no nukta) is not a multiplier:** पाँच करोड → `AMT 5`.
- **Why it matters:** B4 samples amounts "log-uniformly from ₹10 to ₹100 crore" (L179), and D6 claims सौ is a multiplier and सवा a fraction (L441). The top of the planned amount range would therefore be verified wrongly, and silently.

**S1-3. The source appendix is not "every source you used".** The mini PS says "List every source you used in a table in the appendix" (mini PS p.1).
- The appendix has 36 rows (L600–637). A script that reduces each body link to its paper/repo/package found **99 distinct cited sources in the body, 57 of them not in the appendix**. A few of those are the same project under a second URL, for example `pypi:ctc-forced-aligner` and `pypi:indictranstoolkit`.
- Body-only citations include: Salesky 1906.00556, Chen-adjacent Simul-MuST-C, Simul-COMET, TransLLaMa 2402.04636, OTTAWA 2406.01919, NaturalFlow 2606.13121, 2610.17782, 2501.04927, Gaido EMNLP 2021, IndicXlit, Omnilingual-ASR, WhisperX, IndicConformer, DeepFilterNet3 2305.08227, Le Roux 1811.02508, torchmetrics, python-soxr, Spheron, keenable, MarkTechPost, AIKosh, fhai50032 and PHINC.
- NEXT_STEPS Step 4 says "pick 10–15 of the 36 … the ones you have actually opened". That is fine only if the 4-page PDF then cites nothing else.
- **Fix:** generate the appendix from every link actually cited in the submitted PDF.

**S1-4. Part A/§5.2 rules missing from the dataset design (Part C).**
- The C intro (L213) lists the record contents but omits three rules:
  - "Clips of at most 60 seconds" (PS §3 Part A). The only "60 s" in the report is the 10–60 s session length (L37).
  - "For synthetic Hindi … source.model and source.voice give the generating model and voice" (PS §5.2). Neither field appears anywhere in the report.
  - For synthetic records, `source.start`/`source.end` are "0 and the clip duration when the file is the clip" (PS §5.2).
- PS §10 Reproducibility ("Sarvam may ask any top-five team to re-run part of its training or data pipeline … cannot be reproduced leads to disqualification") is also never mentioned.
- The mid-term gate gives zero points for breaking §3 Part A or §5.2 (PS §8).
- **Fix:** add these rules to C intro, C3 ("Hindi word timings" row → add a "§5.2 schema" row: clip ≤ 60 s, all source fields non-empty, synthetic `model`/`voice` present) and the C6 diagram.

**S1-5. E-ML regime (2) is not labelled as offline analysis.**
- Regime (3) is explicitly "an oracle analysis policy … not a deployable component" (L521).
- Regime (2), "translate the Hindi prefix … with the LLM", carries no such label. It uses an external LLM on *gold* Hindi transcripts, which would be a banned cascade (PS §3 Part B) if anyone read it as a system design.
- **Fix:** see the regime (2) section below.

**S1-6. Overclaims an interviewer can puncture.**
- **"36 (intent 18, entity 18) are lost by committing too early" (L3; heading L16 "36 of 70 points ride on not speaking too early").** Intent and entity points can also be lost to mistranslation, mis-hearing or dropped content. The KPIs are also "scored relative to the other teams and the unchanged base checkpoint" (PS §9.1), so the points are not an absolute pool that committing early forfeits. Suggested wording: "36 of the 70 KPI points are for intent and entities, which committing too early puts at risk."
- **"speaking it early saves perhaps 300 ms and risks 18 points" (L75).** One entity error does not risk the whole 18-point component. Say "risks an entity error, the most heavily weighted kind".
- **"even an off-by-one in the brief itself" (L594) and "the mini PS's own 'k + i' hint is off by one" (L3).** These are stronger than A5's own hedge, "It matches STACL only if *i* starts at 0" (L112). Say "is off by one if i is 1-indexed".

**S1-7. The decode-time PAD-bias gate relies on information the checkpoint may not have.**
- B6 (4) gates the bias "off while the model's *own* text stream is inside an unfinished number or before the clause verb" (L197). A4 (5) says the same thing for numbers (L77).
- In Hibiki/Moshi-style models the inner-monologue text stream is the *target* (English) text. "Before the [Hindi] clause verb" therefore cannot be observed without a Hindi transcript. An outside Hindi recogniser would be a cascade, and the report itself says so in rule 8 (L65).
- **Fix:** drop "or before the clause verb", or say it applies only if the base exposes a source-text stream, and confirm with the organisers.

**S1-8. The contents of `en_alt` are defined three different ways.**
- B2: "The **fluent** version goes in `en_alt`… Both encode only the corrected meaning" (L151).
- C5: "Training target = corrected meaning; literal only in `en_alt`" (L293), where "literal" means "five lakh, no no, five crore".
- D5: the literal version "keeps every disfluency and the repair ('five lakh… no no, five crore')" (L437).
- The C6 diagram follows B2: "{monotonic en_text, fluent en_alt}" (L321).
- **Fix:** state once that `en_text` = monotonic + corrected and `en_alt[0]` = fluent + corrected. The repair-preserving literal version belongs only to the Part D deliverable (or to `en_alt[1]`, explicitly labelled). Change C5's guard to "fluent and monotonic targets both corrected; literal-with-repair never in `en_text`".

**S2 and S3 items.** These are listed in the sections below, with fixes:
- D0 "12 prototype phrases" is stale.
- The D1 `ffprobe` glob fails with more than one file.
- B1 and C1 assign Kathbath to different buckets.
- The planned entity-TTS share (25%) lies outside its own sweep (0–20%).
- δ is overloaded, and the δ ranges differ.
- The A3 table's rule numbers clash with the PS's rule numbers.
- D7's headings use different numbers from the D sections.
- The B3 → "Part E-SP" cross-reference points at the wrong thing.
- The D3 sketch drops words instead of interpolating them.
- "all current on PyPI" is false for torch.
- The C3 CER band has a gap, and its 4–5% CER is untagged.
- Name matching uses two different metrics.
- "a few hundred calls" underestimates the call count.
- A2 cites §5.1 for the 10–60 s session length.
- A3 puts a paraphrase in quotation marks.
- NEXT_STEPS mentions Saarika v2.5 and 20 ms figures that the report does not contain.
- The D6 table's DIG and "small numbers" wording is loose.
- The IndicVoices-R "adds no hours" interpretation is tagged [V].
- The observation-adding formula is not convex.

### Inferences
- The serious risks are all *wording* or *coverage* problems in a report whose arithmetic is sound, plus one real code defect (the parser). All can be fixed in an afternoon.
- The parser defect matters most in practice: a checker that silently passes "सौ करोड़" as 100 would corrupt both the C3 TTS gate and the D6 accuracy numbers, and the 37 tests do not catch it.

### Gaps
- The MMS_FA weights could not be downloaded (`dl.fbaipublicfiles.com` returns HTTP 403 through the proxy), so the alignment sketch was run with randomly initialised weights. This tests the API and data flow, not alignment quality.
- arXiv, Hugging Face and download.pytorch.org were unreachable (`curl` code 000), so [S] paper numbers were not re-checked. That was outside this audit's scope anyway.

---

## 1. Do the PS-derived claims match the PS text?

### Takeaway
Almost every quoted rule matches the PS word for word, and all points tables sum correctly. The deviations are three missing Part A/§5.2 rules (S1-4), one wrong section citation, and one paraphrase shown in quotation marks.

### Cited Findings
- **Points: correct.**
  - Mini PS: A 10 + B 40 + C 25 + D 10 + E 15 = 100 ([mini PS](file:///home/user/interiit_minips/Inter%20IIT%20Tech%20Meet%2015.0_%20Sarvam.ai%20Mini%20PS.pdf), p.1–2). The report's compression table (L579–586) uses the same values, and its page budget sums to 0.5 + 1.6 + 1.1 + 0.35 + 0.45 = 4.0 pages.
  - Final KPIs: 12 + 18 + 18 + 16 + 6 = 70. Mid-term: 12 + 3 + 5 = 20. Final report + demo: 5 + 5. Grand total 100 (PS §9.1).
  - The A1 table (L22–31) reproduces all ten components: 12, 3 + 5, 18, 18, 16, 12, 6, 5 + 5 = 100.
  - Appendix row 1 states "KPIs (12/18/18/16/6)", which is correct.
  - NEXT_STEPS "Parts B and C carry 65 of the 100 points" is correct (40 + 25).
- **Matching verbatim or near-verbatim (checked against the PS PDF):**
  - Bulbul "bulbul:v3, voice shubh, language_code="en-IN", 24 kHz", with pace and temperature declared (PS §3 Part B; report L20, L66).
  - The cascade definition (PS §3 Part B; L20, L65).
  - Gates G1–G3 and "final-round KPI points (70) are zero" (PS §8; L33).
  - The protocol rules: Playback, No take-backs, Finishing (30 s), Keeping up (1 s, "because your server is not reading"), Format (one `session_complete`, nothing after it), Timing (accept within 10 s), and failure on "rule 3, 4, 5 or 6, or closes early" (PS §5.1; L54–63).
  - The container rules: port 8000, `/ws`, `/health` 200 within 300 s (PS §5.1; L41).
  - The hour-counting rules (PS §3; L69, L230).
  - The FAQ answers on corrections, numbers, English words and external models (PS §11; L64, L69, L73, L203).
  - The §10 permission for Sarvam API outputs (L139).
  - Causality (L69) and scorer integrity (L67).
- **§4 cited as §5.1 (S3).** "one connection is one session of 10–60 s ([PS §5.1])" (L37). The 10–60 s session length is in PS §4 (Language and conditions, "Session Length 10 to 60 seconds"). Fix: cite §4 for the length and §5.1 for "One connection is one session".
- **Paraphrase in quotation marks (S3).** L69 has "25,000 and 25k both count". The PS says "Equivalent written forms of the same number, such as 25,000 and 25k, both count as correct" (PS §11). Fix: quote it exactly, or drop the quotation marks.
- **G3 abridged (S3).** L33 describes G3 only as "records or hours were added after the mid-term". PS §8 G3 also requires that "The final Part A dataset meets §3 Part A". Fix: add "or the final dataset no longer meets Part A".
- **Missing rules (S1-4 above).**
  - Clip ≤ 60 s.
  - `source.model`/`source.voice` for synthetic records.
  - Synthetic `source.start` = 0 and `source.end` = duration.
  - The top-five reproducibility re-run.
  - "Declare everything in model_card.md and data_card.md" (PS §10). Only `data_card.md` appears, in the C6 diagram (L302).
- **The mini PS's own Part C/D slip is handled correctly.** The mini PS's ML hint says "Use your Part C word timings … your Part C entity checker", although both are built in Part D. The report notes this as "one wording slip" (L493).
- **The PS gloss is used correctly.** The PS's English gloss "Yesterday I got my old bank account closed" (PS §2) is the 8-word target in A5. The Hindi sentence has 9 words. Both counts check out.

### Inferences
- The missing §5.2 synthetic fields are the most consequential omission. Part C's synthetic share is 65 h, and every one of those records would fail the "No source field may be left empty" or `model`/`voice` rules if the manifest builder followed only the report.

### Gaps
- The PS PDF's rule bullets are not numbered in the text layer. The numbering 1–6 is inferred from "breaks rule 3, 4, 5 or 6" plus the order of the bullets. This is the same inference the report makes.

---

## 2. Do numbers that appear in several places agree, and do cross-references resolve?

### Takeaway
The 200 h split, the AL values, the source count (36), the test count (37 phrases plus 4 tests) and the CER and name-match thresholds agree everywhere they appear, with one exception: D0 still says "12 prototype phrases". Several smaller sets of numbers drift apart: Kathbath's bucket, the entity-TTS share versus its sweep, the δ ranges, and rule numbering.

### Cited Findings
- **200 h split: consistent.**
  - Real: 105 + 12 + 5 + 3 = 125. Synthetic: 50 + 15 = 65. Reserve: 10. Total 200 (L3, L128, L137, L219–228; research note [data_sources_licences.md](file:///home/user/interiit_minips/research_notes/Sarvam%20mini%20PS%20research/data_sources_licences.md) L146).
  - The percentages "≈62% real / 33% synthetic / 5% reserve" (L228) are exactly 62.5 / 32.5 / 5 (S3; write "≈63 / 32 / 5" or "62.5 / 32.5 / 5").
- **Kathbath and Common Voice are assigned to different buckets (S2).**
  - B1 (L137): "FLEURS hi_in 5 h, Common Voice or Kathbath 3 h".
  - C1 (L223–224): R3 is "FLEURS hi_in … (+ optional Kathbath, CC0, 150.2 h)" at 5 h, and R4 is "Common Voice hi (CC0) or SPRING-INX" at 3 h.
  - Fix: pick one assignment, for example R3 = FLEURS 5 h and R4 = Common Voice or Kathbath 3 h, and use it in both places.
- **The entity-TTS share lies outside its own sweep (S2).**
  - C1 commits **50 h = 25%** of the budget to S1, entity-dense synthetic speech (L225).
  - The sweep that is supposed to decide that share tops out at 20%: "sweep 0/5/10/20% entity-TTS hours" (B4, L185; C7, L341).
  - Fix: extend the sweep to 0/10/20/30%, or state that 25% is a prior to be revised by the sweep.
- **AL values: consistent.**
  - 2.625, 3.6875, LAAL 2.625 and 3.3, DAL 3.0, AP 51/72, oracle 41/16 = 2.5625 (L3, L104, L110, L112, L116, L118, L592).
  - README ("AL = 2.625") and NEXT_STEPS ("AL = 2.625", "3.6875") agree.
  - The research note [partA_simultaneous_fundamentals.md](file:///home/user/interiit_minips/research_notes/Sarvam%20mini%20PS%20research/partA_simultaneous_fundamentals.md) L230 and L270 agrees (it writes AP as 17/24, which equals 51/72).
- **Thresholds.**
  - **CER ≤ 8–10%:** L263, L313 and NEXT_STEPS agree.
  - **DNSMOS:** "OVRL ≥ 3.0" (C3, L266), "DNSMOS ≥ 3" (C6, L314) and "DNSMOS > 3" (C5, L291) agree.
  - **Name match:** `partial_ratio` ≥ 85 in D6 (L468) and NEXT_STEPS agree with the note [partD_partE_tooling.md](file:///home/user/interiit_minips/research_notes/Sarvam%20mini%20PS%20research/partD_partE_tooling.md) L248. However, C3's TTS gate uses "normalised edit distance ≤ 0.2 on romanised form" (L264) (S3). Fix: use one name metric in both places, or explain why the TTS gate (Hindi vs Hindi) differs from the translation checker (Hindi vs English).
  - **Aligner-disagreement flags:** 100 ms for Hindi (L269, L319, L429) and 80 ms for English (L250). This looks intentional but is not explained (S3).
- **δ is overloaded and its ranges are not reconciled (S2).**
  - A4: look-ahead "about one word or a 200–500 ms pause, to be tuned" (L75).
  - B3: "δ swept over {0.5, 1, 2} s" (L165). The low end of A4's range (0.2 s) is not in B3's sweep.
  - E-ML: δ is reused for something else entirely, "an optional compute delay δ (e.g. 0.2 s)" (L497).
  - C5 gives "a fixed 3 s δ everywhere" as a failure example (L297).
  - Fix: call the compute delay *c* in E-ML, and sweep δ ∈ {0.25, 0.5, 1, 2} s.
- **k grids differ (S3).** A5 uses k = 1..9 (L114), B3 k = 2…8 (L171), E-ML k ∈ {1, 2, 3, 5, 7, ∞} (L521). This is harmless, but one grid could serve throughout.
- **Source count: consistent.** The appendix has 36 rows (L602–637), matching README "36 sources" and NEXT_STEPS "of the 36". (On completeness, see S1-3.)
- **Parser test counts.**
  - "37 test phrases plus 4 unit tests" (L10), "covers 37 phrases" (L451), README "(37 phrases)" and NEXT_STEPS "37 phrases" all match the code: `len(CASES)` is 37, and there are 4 `test_*` functions ([test_hindi_number_parser.py](file:///home/user/interiit_minips/prototypes/test_hindi_number_parser.py)).
  - **Stale (S2):** D0 still reads "`tests/test_numbers.py` # unit tests incl. the 12 prototype phrases + known hard cases" (L372). Fix: "…incl. the 37 phrases from `prototypes/test_hindi_number_parser.py` + new hard cases".
  - "Seven bugs": the README "first draft had seven bugs", the 7 bullets in D6 (L452–458) and the 7 items in the status box (L10) agree.
- **Cross-references.**
  - These resolve correctly: §D6 (L10, README), §D0 (README), §D1 (L12, NEXT_STEPS), C6 (README), "rule 8" (L77, which refers to the report's own table row 8), "thresholds in Part C" (L179), "A4, B3" (L292), "(A5)" (L497, L523) and §D0–D7 (NEXT_STEPS).
  - **Wrong target (S3):** B3 says "`torchaudio.load` now runs on TorchCodec — which is what breaks other tools (see Part E-SP)" (L175). E-SP actually describes DeepFilterNet breaking because it imports the removed `torchaudio.backend` (L571), not `torchaudio.load`. Fix: "…and the removal of `torchaudio.backend` is what breaks DeepFilterNet (see E-SP)".
  - **Rule numbering clashes with the PS (S2).** A3 says the PS's protocol rules are "numbered 1–6" and that a session fails on "rule 3, 4, 5 or 6" (L54). The very next table numbers its own rows 1–10 with different content: row 3 is the PS's rule 4 (Keeping up), and row 6 is the PS's rule 5 (Format). "Rule 8" later means the cascade ban, which is not a §5.1 rule at all. Fix: label the report's rows R1–R10, and give the PS rule number in a column.
  - **D7 headings use different numbers from the D sections (S3).** The RESULTS.md template uses "D2 Alignment, D3 Translation, D4 Checker" (L478–481), while the report sections are D3 Align, D5 Translate, D6 Checker. Fix: use the report's numbers, or unnumbered headings.
- **File paths: consistent.**
  - `prototypes/hindi_number_parser_prototype.py`, `prototypes/test_hindi_number_parser.py`, `prototypes/data/numbers/{digit,teens_and_ties}.tsv` and `prototypes/data/NEMO_LICENSE` all exist.
  - The README folder tree matches the repository, apart from the new `research_notes/Sarvam mini PS final audit/` folder, which the README does not yet list (S3).
  - D0's planned `checker/data/` path differs from `prototypes/data/numbers/`. That is acceptable, because it describes the future layout.

### Inferences
- None of the S2 drifts change a recommendation. Each one is a sentence an interviewer could quote back as "your report says two different things".

### Gaps
- None.

---

## 3. Does every piece of arithmetic recompute?

### Takeaway
Yes, everything recomputes, with exact fractions where relevant. [test_waitk.py](file:///tmp/claude-0/-home-user-interiit-minips/c41df105-99ef-5d9b-b959-5c09e7324606/scratchpad/audit/test_waitk.py) runs the report's own extracted code ([waitk_extracted.py](file:///tmp/claude-0/-home-user-interiit-minips/c41df105-99ef-5d9b-b959-5c09e7324606/scratchpad/audit/waitk_extracted.py), L500–516) plus a `fractions` re-implementation.

### Cited Findings
- **Wait-3 schedule: correct.**
  - Schedule [3, 4, 5, 6, 7, 8, 9, 9], τ = 7, AL = 2.625 (= 21/8).
  - Σg = 42. Σ(t−1)·9/8 = 189/8 = 23.625. (42 − 23.625)/7 = 2.625.
  - The per-row terms sum to 18.375 (L96–105).
- **Off-by-one variant: correct.** `ps_index=True` gives [4, 5, 6, 7, 8, 9, 9, 9], τ = 6, AL = 3.6875 = 59/16 (L112).
- **k table: matches L116 exactly.** k = 1..9 gives 0.5625, 1.5625, 2.625, 3.6875, 4.75, 5.8125, 6.875, 7.9375, 9.0. For k = 1, g never reaches 9, so τ = |y| = 8 and SimulEval's loop divides by 8; the result is still 9/16.
- **Oracle: correct.** Delays [2, 2, 8, 8, 8, 8, 8, 8] give AL = 41/16 = 2.5625, which is less than 2.625 (L118, L592).
- **LAAL and AL with a 10-word hypothesis: correct.** Delays [3, 4, 5, 6, 7, 8, 9, 9, 9, 9] give AL (ref = 8) = 21/8 and LAAL = 33/10 = 3.3 (L110).
- **DAL and AP: correct.** DAL = 3 exactly: g′ = 3, 4.125, 5.25, …, and every term equals 3. AP = 51/72 = 17/24 (L110).
- **"Avoiding 'got' needs k ≥ 6–7 and AL ≥ 5.8": correct.** g(3) = k + 2 ≥ 8 → k ≥ 6, and AL(6) = 5.8125 (L118). "Delays 'yesterday' by about seven words" is also correct: 9 − 2 = 7.
- **Time-domain example (L50): correct.** With 300 ms per Hindi word and zero-length English, AL = 787.5 ms (0.3 × 2.625). With 350 ms per English word, the starts are 0.9, 1.25, 1.60, 1.95, 2.30, 2.65, 3.00, 3.35 s, AL = 937.5 ms, and EndOffset = 3.70 − 2.70 = 1,000 ms. LAAL is also 937.5 ms.
- **Frames: correct.** 1,234,560 / 1,920 = 643 frames exactly, and 1,234,560 / 24,000 = 51.44 s (L37). 0.08 × 24,000 × 2 = 3,840 bytes per frame (L37). One second of output is 48,000 bytes, matching the "≤ 48,000 bytes" assertion (L567).
- **Silero VAD buffer: correct.** 80 ms at 16 kHz = 1,280 samples = 2.5 chunks of 512 (L571). Eight 10 ms hops per 80 ms frame (L571).
- **Compute (L193): consistent with the README formula.**
  - moshi-finetune README: "Total tokens processed = max_steps × num_gpus × batch_size × duration_seq × 9 (token per step) × 12.5 (step per second)", and "1×H100 ≈12k" tokens/s ([moshi-finetune README](https://github.com/kyutai-labs/moshi-finetune), fetched raw from GitHub during this audit, lines 176–201).
  - 12,000 / (9 × 12.5) = 106.7 audio-s per s. 720,000 s / 106.7 = 6,750 s = 1.875 H100-h, or $7.41 at $3.95/h. This matches the research note [finetune_and_latency.md](file:///home/user/interiit_minips/research_notes/Sarvam%20mini%20PS%20research/finetune_and_latency.md) L87.
  - S3: the report does not show the 9 × 12.5 token formula. Add it in one clause, so the 1.9 H100-h figure can be defended.
- **Smaller ratios: correct.** "LR 2e-6 … about 50–100× below typical LoRA rates": typical rates are 1e-4 to 2e-4, which is 50–100×. "IndicTrans2 200M … about 0.8 GB in fp32": 200M × 4 bytes. "Verbs … AL ≥ 5.8": correct.
- **E-ML call count understated (S3).**
  - "with six clips and k ∈ {1, 2, 3, 5, 7, ∞} it is a few hundred calls" (L521).
  - Regime (2) makes one LLM call per committed English word per k. Six 20–40 s clips at about 2.5–3.5 Hindi words/s give about 50–140 words each, so about 6 × 80 × 5 ≈ 2,000–3,000 calls for the five finite k values. Prefixes cannot be shared across k, because the committed English context differs.
  - Fix: "a few thousand cached calls (a few dollars)".

### Inferences
- The handwritten-appendix numbers are safe to copy as they stand.

### Gaps
- None.

---

## 4. Does the code run, and do the API claims hold?

### Takeaway
- **Wait-k simulator:** passes its specified unit tests.
- **Echo server:** behaves exactly as claimed against a protocol-faithful client.
- **MMS_FA alignment sketch:** runs end to end on the torchaudio 2.11.0 API, with random weights because the real weights host is blocked. It has two small defects: words with an empty romanisation are dropped rather than interpolated, and trailing "…" is kept in the display words.
- **ffmpeg:** all flags are valid, but the D1 `ffprobe` line fails when there is more than one clip.
- **torchaudio.load:** does require TorchCodec, as the critique said, and the report's soundfile advice is correct.
- **Parser:** passes its 37 tests, but fails 27 of 65 adversarial probes, 19 of them in ways that are *not* among the documented limits.

### Cited Findings
- **Wait-k (`schedule()`/`al()`, L500–516).**
  - The code was extracted verbatim and run. With `src_end = [1..9]`, zero durations, k = 3 and 8 words, AL = 2.625. With `ps_index=True`, AL = 3.6875. Both match the unit test stated at L519.
  - `al()` reproduces SimulEval's `ALScorer.compute` line for line: the `delays[0] > source_length` early return, `gamma = target_length/source_length`, `AL += d - t_minus_1/gamma`, and the break at the first `d >= source_length` ([SimulEval latency_scorer.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py), lines 138–164, fetched during this audit).
  - LAAL uses `max(|Y|, |Y*|)` (same file, line 184). `instance.py` line 369 has `start = max(prev_end, delay)`, as the report claims (L50).
- **E-Dev echo server (L530–564), run verbatim** with websockets 17.2 (`Requires-Python: >=3.11`, confirmed from the wheel) on Python 3.13, using the scratch-dir scripts [echo_server_extracted.py](file:///tmp/claude-0/-home-user-interiit-minips/c41df105-99ef-5d9b-b959-5c09e7324606/scratchpad/audit/echo_server_extracted.py) and [echo_client.py](file:///tmp/claude-0/-home-user-interiit-minips/c41df105-99ef-5d9b-b959-5c09e7324606/scratchpad/audit/echo_client.py).
  - The client sends `session_start`, real-time 3,840-byte frames on absolute deadlines, `input_end` with `valid_samples`, then silence until `session_complete`.
  - Results:
    - `GET /health` returns 200, and `/other` returns 404.
    - Session s-0001 (72,777 samples, not a multiple of 1,920): 145,554 bytes echoed, which is exactly `valid_samples` × 2.
    - Session s-0002 (48,000 samples, an exact multiple): 96,000 bytes echoed, also exact.
    - Each session got 1 text frame and 0 frames after it. The largest server frame was 3,840 bytes.
    - The first audio played at 1.002 s, against DELAY = 1.0.
    - `session_complete` arrived 1.002 s after `input_end`.
    - The second session ran after a 2 s gap on the same server.
  - So the claim at L527 ("returned /health 200, echoed exactly valid_samples × 2 bytes and sent a single session_complete") is **confirmed**.
  - Also confirmed from the websockets 17.2 source: `compression="deflate"` is the default (server.py line 494) and `max_queue` defaults to 16 (line 518).
- **MMS_FA `align_mms.py` (L409–426) against torchaudio 2.11.0.**
  - The wheel was downloaded and inspected ([unpacked wheel](file:///tmp/claude-0/-home-user-interiit-minips/c41df105-99ef-5d9b-b959-5c09e7324606/scratchpad/audit/dl_ta/x/torchaudio/)). The API matches the sketch:
    - `pipelines/_wav2vec2/impl.py`: `Wav2Vec2FABundle.get_model(self, with_star=True, *, dl_kwargs=None)`, `get_tokenizer()` → `Tokenizer(self.get_dict())`, `get_aligner()` → `Aligner(blank=0)`, and `MMS_FA = Wav2Vec2FABundle("https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt", …, _sample_rate=16000, _normalize_waveform=True)`.
    - `aligner.py`: `Aligner.__call__(emission: Tensor, tokens: List[List[int]]) -> List[List[TokenSpan]]`, which raises if `emission.ndim != 2`. The sketch passes `em[0]`, so this is fine.
    - `Tokenizer.__call__(transcript: List[str]) -> List[List[int]]`.
    - `functional/_alignment.py` contains both `forced_align` and `merge_tokens`, with no deprecation text.
    - `torchaudio/io`, `sox_effects` and `backend` are absent from the wheel.
  - **Alphabet: confirmed.** The labels are the 26 lowercase letters plus `'`; `get_dict()` adds blank `-` at 0 and star `*` at 28, for 29 outputs (L406).
  - **Frame rate: confirmed.** The conv strides are 5·2·2·2·2·2·2 = 320 samples, which is 20 ms at 16 kHz (L245, L433). A 1 s input gives 49 emission frames. The sketch's `sec = x.size(1)/em.size(1)/sr` gives 20.4 ms for 1 s and about 20.01 ms for 40 s, so the end-of-clip error is at most about one frame (S3; the tutorial has the same convention).
  - **Run:** in a scratch venv with torch 2.11.0 and torchaudio 2.11.0 from PyPI, the sketch ran **unchanged**, with `MMS_FA.get_model` patched to return a randomly initialised network of the same architecture ([run_align_randinit.py](file:///tmp/claude-0/-home-user-interiit-minips/c41df105-99ef-5d9b-b959-5c09e7324606/scratchpad/audit/run_align_randinit.py)). Its output was ordered, non-negative and inside the clip.
  - **Defect 1 (S2):** for the transcript "पाँच लाख… नहीं नहीं, पाँच करोड़। / नंबर है 98765 और राहुल शर्मा।" (12 words), it wrote **11** entries. "98765" was silently dropped, although the code comment says "words with an empty romanisation get interpolated" (L417). No interpolation code exists, and the companion "every transcript word timed" assertion (L429) would then fail. Fix: add the interpolation (split the gap between the neighbouring kept words), or reject digits during transcript validation.
  - **Defect 2 (S3):** the display word is written as "लाख…", because the split regex strips only `[।?,.!]`. The PS's own example contains "…". Fix: strip `[।॥?,.!…;:"“”'()]` from display words too.
  - **Install-size note (S3):** `pip install torch==2.11.0` from PyPI pulls the CUDA 13 build, `2.11.0+cu130`; the venv grew to 4.7 GB. D0 promises CPU-only. Recommend `--index-url https://download.pytorch.org/whl/cpu` on the interview laptop.
  - uroman 1.3.1.1 really gives the romanisations quoted at L406: राहुल → raahul, पाँच → paamc, करोड़ → karodd, बैंक → baimk, अकाउंट → akaaumtt, and "98765" stays "98765", which the regex then reduces to "" ([uroman_check.py](file:///tmp/claude-0/-home-user-interiit-minips/c41df105-99ef-5d9b-b959-5c09e7324606/scratchpad/audit/uroman_check.py)).
  - The "~1.18 GB MMS_FA weights" figure (L377) could not be checked, because the weights URL returns HTTP 403.
- **torchaudio.load.**
  - In the 2.11.0 source, `load()` says "Load audio data from source using TorchCodec's AudioDecoder… As of TorchAudio 2.9, this function relies on TorchCodec", and `_torchcodec.py` raises `ImportError` when torchcodec is missing. TorchCodec uses FFmpeg (`_torchcodec.py` line 188 and nearby).
  - Running it without torchcodec gives `ImportError: TorchCodec is required for load_with_torchcodec`.
  - The report's advice (L377: "Load audio with `soundfile` and pass tensors, because `torchaudio.load` now depends on TorchCodec and system FFmpeg") is correct, and the D3 sketch follows it (`sf.read`).
- **ffmpeg 6.1.1 (installed), tested on a synthetic stereo 48 kHz `.m4a`** ([ff/](file:///tmp/claude-0/-home-user-interiit-minips/c41df105-99ef-5d9b-b959-5c09e7324606/scratchpad/audit/ff/)).
  - The D1 conversions `-ac 1 -ar 24000 -sample_fmt s16 -c:a pcm_s16le` and `-ar 16000 -c:a pcm_s16le` work, producing `pcm_s16le`, 16000 Hz, 1 channel. `-sample_fmt s16` is redundant but harmless.
  - **Bug (S2):** the D1 line `ffprobe -v error -show_entries format=duration -of csv=p=0 clips/24k/*.wav` (L397) **fails as soon as there are two clips**: "Argument 'clips/24k/clip02.wav' provided as input filename, but 'clips/24k/clip01.wav' was already specified", exit code 1. ffprobe accepts only one input. Fix: `for f in clips/24k/*.wav; do printf '%s,' "$f"; ffprobe -v error -show_entries format=duration -of csv=p=0 "$f"; done`.
  - The E-SP telephone chain works: `-af "highpass=f=300,lowpass=f=3400" -ar 8000 -c:a pcm_mulaw`, then back to 24 kHz (L571).
  - Encoders present: pcm_mulaw, pcm_alaw, libgsm, libopus, g722, g726. `libopencore_amrnb` is absent ("Unknown encoder"). This confirms "no AMR-NB encoder in stock Ubuntu ffmpeg" (L280, L571).
  - S3: ffmpeg's `highpass`/`lowpass` default to 2 poles (12 dB/octave). That is a gentle telephone band; cascade them, or say "approximate".
- **Other API claims confirmed.**
  - `soxr.ResampleStream(in_rate, out_rate, num_channels, dtype='float32', quality='HQ', vr=False)` exists in soxr 1.1.0, and per-chunk output lengths do vary (979/1468 samples per 1,920-sample chunk) (L571).
  - torchmetrics 1.9.0: `scale_invariant_signal_distortion_ratio(…, zero_mean: bool = False)` (L571).
  - deepfilternet 0.5.6's `df/io.py` line 9 has `from torchaudio.backend.common import AudioMetaData`. It requires `numpy (>=1.22,<2.0)`. deepfilterlib 0.5.6 has a cp311 wheel but no cp312 wheel (L571).
  - PyPI versions on the audit date: torchaudio 2.11.0, uroman 1.3.1.1, rapidfuzz 3.14.6, IndicTransToolkit 1.1.1, silero-vad 6.2.3, nemo-text-processing 1.2.0, num2words 0.5.14, word2number 1.1 and websockets 17.2 are all current. **torch is at 2.14.1**, so "all current on PyPI on 14 Oct 2026" (L377) is false for `torch==2.11.*` (S3). The pin is right, because torchaudio's last release is 2.11.0, but say "torch pinned to 2.11 to match torchaudio 2.11.0".
- **Parser.**
  - `python prototypes/test_hindi_number_parser.py` prints "OK: 37 phrases and 4 tests passed" on Python 3.13.16. The demo and README commands give the documented outputs, for example "नौ आठ डबल सात" → DIG "9877".
  - The teens/ties table has 220 rows covering 90 distinct values (10–99), as claimed (L181, L441).
  - **Adversarial probe, 65 phrases ([probe_parser.py](file:///tmp/claude-0/-home-user-interiit-minips/c41df105-99ef-5d9b-b959-5c09e7324606/scratchpad/audit/probe_parser.py)).**
    - **Correct:**
      - दो हज़ार पच्चीस → 2025 (as an amount; there is no year type)
      - उन्नीस सौ सैंतालीस → 1947
      - सौ रुपये → 100
      - एक सौ एक → 101
      - डेढ़ हज़ार → 1500
      - ढाई सौ → 250
      - साढ़े सात सौ → 750
      - पौने तीन सौ → 275
      - बारह सौ → 1200
      - चार सौ बीस → 420
      - दो लाख पच्चीस हज़ार पाँच सौ → 225500
      - एक करोड़ पच्चीस लाख → 12500000
      - साढ़े बारह बजे → 12:30
      - दो बजकर पाँच मिनट → 2:05
      - पौने एक → 0.75
      - ओटीपी है ट्रिपल ज़ीरो फोर → "0004"
      - डबल ज़ीरो सात → "007"
      - डबल नाइन → "99"
      - पाँच लाख रुपये। → 500000 (punctuation attached)
      - ५ लाख → 500000
      - ५०० रुपये → 500
      - लाखों / हज़ारों / करोड़ों → nothing
      - पाँच-छह हज़ार → RANGE (5000, 6000)
      - तीन चार सौ → RANGE (300, 400)
      - सत्तर अस्सी हज़ार → RANGE (70000, 80000)
      - पाँच लाख नहीं दिए → 500000
      - पाँच लाख, सॉरी, छह लाख → two amounts
      - dangling डबल / साढ़े → nothing
    - **New failures, not in D6's "Known limits":**
      1. सौ or हज़ार before करोड़ or लाख truncates the amount (S1-2): पाँच सौ करोड़ → 500; सौ करोड़ → 100; एक सौ करोड़ → 100; डेढ़ सौ करोड़ → 150; एक हज़ार करोड़ → 1000. **Cause:** in `read_amount`, a non-सौ multiplier is accepted only if `prev == "num"`, so it is rejected after सौ (where `prev == "mult"`). दो सौ पचास करोड़ passes only because पचास sets `prev = "num"`.
      2. सवा/साढ़े/पौने followed directly by a multiplier: सवा सौ → 100 (wrong; should be 125); सवा लाख and सवा करोड़ → nothing (missed). **Cause:** a multiplier after a fraction is not allowed, and the bare-सौ rule then fires on its own.
      3. पाँच करोड (no nukta, a common ASR/LLM spelling) → AMT 5, wrong by a factor of 10⁷. **Fix:** add करोड to MULT, plus other normal variants (लाख़ is not needed).
      4. Quotes and brackets attached to a word are not stripped by the tokeniser: “पाँच लाख”, (पाँच लाख) and 'पाँच' लाख → nothing.
      5. Homographs give false positives in code-mixed text: वन विभाग ("forest department") → AMT 1; पेमेंट टू मिस्टर शर्मा → AMT 2; वन टू वन मीटिंग → DIG "121". English-in-Devanagari words for "to/too/for/at/one" are common in Hinglish.
      6. पौने एक बजे → TIME "0:45" (should be 12:45 on a 12-hour clock).
      7. Distributive reduplication: दो दो सौ रुपये ("200 each") → AMBIG (2, 2) + AMT 100; सौ सौ / हज़ार हज़ार → two amounts.
      8. दो दशमलव पाँच करोड़ gives two wrong amounts (2 and 50000000) instead of a flag. D6 lists decimals as unhandled (L462), but the failure is silent.
    - **Failures that match the documented limits:** dates (पंद्रह अगस्त → 15), ordinals (पहला), आधा, a phone number split by a comma, and दस बारह पंद्रह → "101215".
    - **Out of scope:** Latin "zero nine eight" → nothing, which is expected because Level-1 transcripts are in Devanagari. A single ज़ीरो → AMT 0 is acceptable.
  - **D6 table wording is loose (S3).**
    - "DIG … any डबल/ट्रिपल/English digit word" (L446): a *single* English digit word gives AMT, because `forced` only matters when there are 2 or more values.
    - "Three or more small numbers are always read as digits" (L463): any 3 or more consecutive number words are read as digits, for example बीस पच्चीस तीस → "202530".
  - **Fix for the parser:**
    - Allow a multiplier after `prev == "mult"` when the previous multiplier was सौ (or was smaller).
    - Allow `FRAC + MULT` (सवा लाख = 1.25 × 10⁵).
    - Add करोड.
    - Add quotes and brackets to the tokeniser's punctuation class.
    - Gate the EN digit words (वन/टू/फोर/एट) to runs of two or more numbers, or to a phone/OTP context.
    - Map पौने एक बजे to 12:45.
    - Add these 19 phrases to the tests.
    - Update the README and NEXT_STEPS wording "fixed" accordingly.

### Inferences
- The code that the report says it ran (wait-k, echo server) is honest. The alignment sketch's "test before relying on it" caveat (L406) is warranted: its two defects would surface on the first real transcript.
- The parser's "fixed" label (L10, README, NEXT_STEPS) overstates what the 37 tests cover. The most damaging gap is the सौ करोड़ family, because B4's sampler deliberately produces those amounts.

### Gaps
- Alignment *quality* (boundary error) cannot be assessed without the real MMS_FA weights.
- The Dockerfile was described in prose, not provided, so it was not built.

---

## 5. Is there an explicit step from telephone (8 kHz) or 16 kHz audio to 24 kHz s16le? Is observation adding given a ratio?

### Takeaway
Both are partly present.
- **Telephone/16 kHz → 24 kHz:** there is an augmentation chain (24 kHz → 8 kHz → codec → back to 24 kHz) and a "resample to 24 kHz s16" box in the C6 diagram. There is no prose step for *real source corpora* recorded at 8 or 16 kHz, nor any mention of the bandwidth mismatch this creates.
- **Observation adding:** present in E-SP, with a ratio of 0.1–0.3 × the noisy signal (the critique proposes 10–20%).

### Cited Findings
- **Present: telephone augmentation chain.** C4: "a telephone chain: 24 kHz → 8 kHz, 300–3400 Hz band-pass, G.711 μ-law/A-law, GSM, AMR-NB 4.75–12.2 kbps or Opus 6–12 kbps, 1–5% packet loss, then back to 24 kHz, matching the PS's 'lower-bandwidth recordings are upsampled'" (L280).
- **Present: E-SP telephone copies.** "`ffmpeg -af "highpass=f=300,lowpass=f=3400" -ar 8000 -c:a pcm_mulaw`, then back to 24 kHz" (L571).
- **Present: diagram box.** C6: "VAD trim · 1-speaker check · resample to 24 kHz s16" on the real-audio branch (L308). D1 converts phone recordings to 24 kHz s16le (L394).
- **Missing.**
  - A prose step stating each real corpus's native sample rate, the resampler (for example soxr VHQ or `ffmpeg -ar 24000 -ac 1 -c:a pcm_s16le`, with no dither and a clip check), and that the original rate is recorded in `data_card.md`.
  - Any warning that real data upsampled from 16 kHz (and any 8 kHz telephone audio) has no energy above 8 kHz (or 4 kHz), while 24 kHz TTS does. Bandwidth then becomes a "real vs synthetic" shortcut cue.
  - Grep for "upsampl", "16 kHz" and "8 kHz" finds nothing beyond L280, L406, L571 and NEXT_STEPS L43.
  - Proposed fix: add to C4 "Resample every real source to 24 kHz mono s16le with soxr (record the native rate); low-pass a share of synthetic clips at 8 kHz (and 4 kHz) during training so bandwidth does not identify the source".
- **Present: observation adding.** "Iwamoto et al. showed … that mixing some noisy input back in ('observation adding') helps ([arXiv 2201.06685]) [S: re-check]" and the mitigation "output = enhanced + 0.1–0.3 × noisy" (L571). Research note [partD_partE_tooling.md](file:///home/user/interiit_minips/research_notes/Sarvam%20mini%20PS%20research/partD_partE_tooling.md) L573 has the same. The appendix row 36 entry is "observation adding helps".
- **Formula not convex (S3).** As written, "enhanced + 0.1–0.3 × noisy" raises the level by up to 30% and can clip in s16. Write it as a convex mix, (1 − α)·enhanced + α·noisy with α ≈ 0.1–0.3, and tag the range as the report's own choice unless it is checked against Iwamoto.

### Inferences
- The critique is half right. The augmentation exists, but the *real-data ingestion* resampling step and the bandwidth-shortcut risk are not written down. That matters for the "real-majority" thesis, because about 125 h of real audio will arrive at sample rates below 24 kHz.

### Gaps
- The native sample rates of IndicVoices, MUCS, FLEURS and Kathbath are not stated in the report, and I did not verify them (Hugging Face is unreachable).

---

## 6. Is E-ML regime (2) clearly labelled as offline analysis?

### Takeaway
No. Regime (3) is labelled; regime (2) is not.

### Cited Findings
- Regime (2): "*Prefix translation*: for each step *i*, translate the Hindi prefix w₁…w_{k+i−1} with the LLM, given the English already committed, and commit only word *i* (STACL-style prefix-to-prefix)…" (L521). It carries no label saying it is offline-only or a cascade if deployed.
- Regime (3): "an oracle analysis policy … This is an analysis tool, not a deployable component: in the real system the same behaviour must be learned from training-pair timing, because an outside Hindi detector steering output would be a cascade" (L521).
- The PS bans "machine translation or LLM output fed to the checkpoint" and "any other speech-translation model used in place of the checkpoint" (PS §3 Part B). Regime (2) also uses gold transcripts, not audio.
- **Fix:** add after regime (2): "Like (3), this is an offline simulation on gold transcripts, run to measure how prefix translation fails. An external LLM translating the input would be a banned cascade in the submitted system."
- Regime (1) ("Oracle shift") should get one clause too: "offline re-timing of a full-sentence translation".

### Inferences
- An interviewer asking "so you'd use an LLM at inference?" is the likely trap. One sentence pre-empts it.

### Gaps
- None.

---

## 7. Does the title "Hold the verb, speak everything else" agree with the body?

### Takeaway
No, and an interviewer would notice.
- **Holding:** the body holds much more than the verb (negations, amounts plus units, names, repairs).
- **The flagship example:** "speak 'Yesterday I… my old bank account' immediately" is impossible under the report's own A5 anticipation analysis and its B3 monotone placement rule, if the target is "Yesterday I got my old bank account closed".
- **B2's causative:** B2's "verb-deferring" causative still commits the auxiliary "got" before the Hindi verb.

### Cited Findings
- **Title (L1):** "Hold the verb, speak everything else".
- **Opening paragraph (L3):** "speak 'Yesterday I… my old bank account' immediately, but hold the English verb until 'बंद करवा दिया' and hold an amount until the repair 'पाँच करोड़' and one beat of silence have been heard … an **earliest-safe-time placement rule** that holds only verbs, negations and entity spans".
- **A5 (L89):** "got … करवा दिया (x₈–x₉) — **not heard: anticipation**". Also L106: "Risk: 'got' (t=3) is spoken after 5 words, before 'बंद करवा दिया' -> anticipation."
- **A5 oracle (L118):** "Yesterday and I after word 2, then everything after 'करवा' (word 8)". This holds "my old bank account" too. The conclusion (L592) calls it "an oracle which waits only at the verb".
- **B2 (L151):** "use cheap verb-deferring constructions (causative 'got X closed', …) … Keep resumptive topicalisation ('my old bank account, I got it closed') rare". Yet the topicalised form is the only target order in which "my old bank account" can precede the held verb.
- **B3 (L165):** "(3) … negators and auxiliaries go to the Hindi verb group … (4) take aᵢ = cummax(max(aᵢᴹᵀ, aᵢᵂᴬ)), which is conservative and monotone". Holding "got" therefore holds every later English word.
- **Proposed fix.**
  - **Title:** "Wait only where the meaning can still change", or "Hold what can flip, speak the rest". Alternatively keep the current title and add a subtitle: "(verbs, negations, amounts, names and repairs wait; time, topic and context go early)".
  - **L3 example:** replace with: "speak 'Yesterday I…' immediately, hold 'got … closed' until 'करवा दिया' is heard, and hold an amount until the repair's unit and one beat of silence. If B2's experiment shows the naturalness cost is acceptable, a topicalised target ('Yesterday, my old bank account — I got it closed') lets the object go early too."
  - **B2:** add one sentence: "The causative defers the *lexical* verb, but its auxiliary 'got' still waits for 'करवा' under B3. Only topicalisation lets the object be spoken before the Hindi verb, which is why the B2 experiment measures that trade-off."
  - **Conclusion (L592):** change "an oracle which waits only at the verb" to "an oracle which waits only from the verb onwards".

### Inferences
- This is the highest-value edit. The title and first paragraph are what the interviewer reads first, and the contradiction is visible from A5's own table.

### Gaps
- None.

---

## 8. Are the evidence tags applied consistently?

### Takeaway
The tags are mostly consistent: [V] appears 50 times, [L] 15 and [S] 42, and the appendix tags match the body. About a dozen strong claims are untagged, and a few [V] tags sit on interpretations rather than on checked facts.

### Cited Findings
- **Counts:** `grep -c` on the report gives 50 lines with "[V]", 15 with "[L" and 42 with "[S".
- **Appendix agreement:** the appendix tags match the body for every row checked. Examples: row 9 is S/V for the Hibiki-Zero paper and repo; row 12 is V/S; row 26 is L for nemo-text-processing, matching "[L, files inspected]"; row 33 is S/V.
- **Untagged strong claims (add a tag, or mark them as the report's own inference):**
  - A2 (L47): "Hibiki-style models advance only when fed frames".
  - A4 (L77): "A naive per-word aligner — awesome-align, or Hibiki-style likelihood gain — would align 'five' to the reparandum 'पाँच'". This is an inference.
  - A4 (L75): "saves perhaps 300 ms and risks 18 points". This is also an overclaim; see S1-6.
  - **B1 exclusion list (L137).** "Gram Vaani (academic-only or CC BY-NC…), Shrutilipi…, XTTS-v2 (non-commercial CPML), F5-TTS base weights (CC-BY-NC), MMS-TTS (CC-BY-NC 4.0)" is untagged; only Fish Speech carries [V].
    - The research note [data_sources_licences.md](file:///home/user/interiit_minips/research_notes/Sarvam%20mini%20PS%20research/data_sources_licences.md) L79–81 has sources for each exclusion.
    - Licence claims are high-stakes, so tag each one.
  - B1 (L139): "Licence-clean open TTS offers perhaps 20–50 voices". This is an estimate.
  - B5 (L193): "about 50–100× below typical LoRA rates, suggesting fragility". This is an inference.
  - B6 (L197): "quantisation is not a latency lever when a ≤ 7B model already fits in 80 ms on an H100". This is unsourced.
  - C1 table S1 (L225): "BPCC-H-Daily (CC BY 4.0, 139K …)" and "PHINC (CC BY 4.0, 13,738 …)". These have no tag.
  - **C3 (L259): "the round-trip ASR itself has about 4–5% CER on clean speech".**
    - This is untagged, and its source in the note is the vendor-reported Saarika v2.5 CER of 4.42% ([hard_content_validation.md](file:///home/user/interiit_minips/research_notes/Sarvam%20mini%20PS%20research/hard_content_validation.md) L144, L155).
    - Saarika is not one of the gate's ASRs, which are "IndicConformer + Saaras or Omnilingual" (L263).
    - Tag it [S], and name the system.
  - C2 table (L241): "dropped words beyond ~18 s reported unofficially". This has no source.
  - C2 table (L253): the Sarvam `mayura:v1` / `sarvam-translate:v1` capabilities. These are untagged.
  - E-SP (L571): "erodes fricatives and aspiration (स, श, फ़)". This is an inference.
  - B2 (L153) and B5 (L193): the MarkTechPost and keenable citations are untagged. That is deliberate, since they are the "misleading" items, but tag them [S] for uniformity.
- **[V] on interpretations:**
  - C1 (L230): "treat IndicVoices-R as the same recordings as IndicVoices: it adds no hours … [V]". The source verifies what IndicVoices-R *is*. Whether it "adds no hours" is the report's reading of PS §3, which counts the "union of the source-file intervals"; IndicVoices-R files are different files. Fix: "[V] for provenance; hour treatment is our interpretation — ask".
  - L110 puts [V] after the computed LAAL = 3.3. The number is [L]; the [V] covers only the formula's source. Fix: "[L, formula V]".
- **Tag-variant spellings:** [S: re-check typography], [S: re-check, secondary], [S: confirm in the Modal dashboard], [L, files inspected], [V, wheel inspected]. These are fine, but the legend (L5) should say that suffixes are free text.
- **C3 thresholds leave a gap (S3).** "Keep if best-of-two CER ≤ 8–10%; review if WER > 25% or CER > 15%" (L263) leaves 10–15% CER undefined. Fix: "10–15%: regenerate; > 15%: drop after 3 tries".

### Inferences
- The tagging scheme is a real strength in an interview, so the untagged licence and threshold claims are worth fixing first.

### Gaps
- I did not re-verify the [S] numbers against primary papers; the hosts are blocked.

---

## 9. Do NEXT_STEPS.md and README.md match the report? Are the critique fixes really in place everywhere?

### Takeaway
README and NEXT_STEPS match the report on structure, points, the 200 h split, AL values, the test count (37), the source count (36), the thresholds and the critique fixes. They differ in three ways:
- They inherit the parser's "fixed" overstatement.
- NEXT_STEPS mentions figures the report never states (Saarika v2.5 WER, "within-20 ms").
- The README tree omits the new audit folder.

All four fixes from commit 6b06d3e (parser rewritten; the "fifty million" claim removed; Hibiki-family wording; the passive-voice caution) are present, with no leftovers.

### Cited Findings
- **Critique fix 1, parser rewritten: present.**
  - Status box (L10), D6 (L441–464), README "Number parser (fixed) … seven bugs", NEXT_STEPS Step 3 "already tested on 37 phrases".
  - Leftover: D0 "the 12 prototype phrases" (L372) (S2, above).
  - Overstatement: "fixed" (L10, README) should become "rewritten; 37 tests pass; known gaps in §D6", once the new failures from section 4 are added to §D6.
- **Critique fix 2, "fifty million" claim removed: present and consistent.**
  - A4 (L73): "Whether the scorer would also accept 'fifty million' is not stated … Keeping 'crore' is safe either way."
  - NEXT_STEPS L120: "whether the scorer accepts 'fifty million' isn't stated."
  - The other occurrences are consistent with this: D6 (L468) has the checker compare by value ("'50 million' … all match 5×10⁷") with a separate unit-preservation flag, and C5 (L294) lists "half a million" as an inconsistency to avoid. No passage still claims the scorer accepts "fifty million".
- **Critique fix 3, Hibiki-family wording: present.**
  - A2 (L37): "say 'most likely Hibiki-family' until the architecture is released, never 'proven'" and "The PS's demand for English word timings does **not** tell Hibiki and Hibiki-Zero apart".
  - NEXT_STEPS L119 says the same.
  - L3 still says the 80 ms framing and English word timings "strongly suggest" a Hibiki/Moshi-style base. That is compatible with A2, because both Hibiki variants are Moshi-family, so it is acceptable.
  - B3's "safe fallback if the base turns out to be Hibiki-Zero-like" (L165) is consistent.
- **Critique fix 4, passive-voice caution: present.**
  - B2 (L151): "Do **not** use the passive as a deferral device: 'Yesterday my old bank account was closed' drops the causative".
  - A grep finds no other passive recommendation (for example "was closed").
  - There is a residual tension with B2's causative "got"; see section 7.
- **README agrees** with the report: the folder tree (seven notes), section table, 36 sources, 37 phrases, evidence-tag definitions, status/caveats (identical wording to L9–11) and next step (§D1).
- **NEXT_STEPS agrees** with the report on these points:
  - Step 1 clip table = D1 table (L383–390).
  - Step 4 page budget = L579–586.
  - Step 4 handwritten values: 3, 4, 5, 6, 7, 8, 9, 9; τ = 7; r = 8/9; AL = 2.625; 3.6875.
  - Step 6 interview points: 36 vs 16 points; the playback rule; the 1 s reading rule; AL ≠ k; name-match ≥ 85; CER ≤ 8–10%.
  - Step 0: the blocked hosts are still blocked. This audit confirmed `dl.fbaipublicfiles.com` returns 403, and arXiv, Hugging Face and download.pytorch.org do not connect; `raw.githubusercontent.com` works.
  - Its own header says it is excluded through `.git/info/exclude`, and that file does contain `NEXT_STEPS.md`.
- **NEXT_STEPS mismatches (S3).**
  - Step 5 asks to re-check "IndicWhisper on Vistaar and Saarika v2.5" WERs. The report cites IndicWhisper as [V], so it needs no re-check, and it never cites Saarika v2.5; it cites Saaras v3's ~19.3% (L240). Fix: "Saaras v3 (~19.3%, 10-language aggregate)".
  - Step 5: "TIMIT within-20/25 ms figures". The report uses only "within 25 ms" (L167, L245, L247). Fix: "within-25 ms".
  - Step 4: "pick 10–15 of the 36". Add "and make sure every source cited in the PDF is in the table" (mini PS requirement; see S1-3).
- **README omission (S3).** README's folder tree lacks `research_notes/Sarvam mini PS final audit/`, which now holds five audit notes plus this one.

### Inferences
- Once the S1/S2 report edits above are made, three companion edits keep the README and NEXT_STEPS in sync:
  - README: the parser wording, and the audit folder in the tree.
  - NEXT_STEPS: Step 5's two re-check items, and Step 4's source-table rule.

### Gaps
- None.
