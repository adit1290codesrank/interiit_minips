# Hold the verb, speak everything else

The Sarvam task rewards a translator that knows when *not* to speak. Of the 70 final KPI points, **36 (intent 18, entity 18) are lost by committing too early** to a verb, a negation, an amount or a name, while latency is worth only 16 and a single protocol failure (gate G1) zeroes all 70 ([Sarvam PS §7–§9.1](file:///home/user/interiit_minips/Sarvam.pdf)). The winning strategy is therefore asymmetric waiting: speak "Yesterday I… my old bank account" immediately, but hold the English verb until "बंद करवा दिया" and hold an amount until the repair "पाँच करोड़" and one beat of silence have been heard. For a Hibiki/Moshi-style base checkpoint, which the 80 ms / 24 kHz framing and the required English word timings strongly suggest, that waiting behaviour is learned from *where the English sits in time in the training pairs*, so the dataset and its timing rule are the main levers ([kyutai-labs/hibiki](https://github.com/kyutai-labs/hibiki)). This report recommends a **real-majority 200-hour dataset** (about 125 h of licence-clean real speech led by IndicVoices, about 65 h of entity-dense synthetic speech, 10 h reserve), **chunk-wise monotonic English targets** that encode only corrected meaning, and an **earliest-safe-time placement rule** that holds only verbs, negations and entity spans. The handwritten appendix answer is a wait-3 schedule of 3, 4, 5, 6, 7, 8, 9, 9 Hindi words with **τ = 7, r = 8/9 and AL = 21/8 = 2.625 words**; the mini PS's own "k + i" hint is off by one against STACL and would give 3.6875. For Parts D and E, every recommended tool runs on a CPU: torchaudio 2.11's MMS_FA aligner (still shipped, despite the tutorial banner announcing its removal), uroman, a plain-Python Hindi number parser built on NeMo's tables, and a ~60-line wait-k simulator that reuses the Part D outputs. The three biggest unknowns no source resolves are the base checkpoint's timing convention, the exact latency formula Sarvam uses, and how many usable single-speaker Hindi hours IndicVoices really holds.

> **How to read the evidence tags.** The research proxy blocked arXiv, Hugging Face and the ACL Anthology, so claims carry one of three tags. **[V]** means checked against primary code, a README, a downloaded wheel or the two PS PDFs. **[L]** means tested locally in the research sandbox. **[S: re-check]** means the claim rests on a search-engine extract or secondary summary of a paper; the URL is the primary page, but the number must be re-checked against the PDF before you quote it in an interview. Paper equations marked [S] were reconstructed from extracts and memory.

> **Status and caveats: read before relying on this report.**
>
> - **Unchecked paper numbers.** The network proxy blocked arXiv, Hugging Face and Sarvam's docs during research. Most paper figures came from search results and are tagged **[S: re-check]**. Re-check any you put in the submission, since you'll be cross-questioned.
> - **Number parser.** The prototype in `prototypes/hindi_number_parser_prototype.py` is a rough draft with four known bugs:
>   - "दो तीन लाख" (two to three lakh) returns 500,000.
>   - "साढ़े चार बजे" (4:30) returns 4.
>   - "एक एक" becomes the digit string 11, though it can mean "one each".
>   - Amounts come back as floats.
>
>   The Part D plan (§D6) lists the fixes. The prototype also needs NeMo's Hindi number tables to run. They weren't installed in the research environment, so it was tested on stand-in tables.
> - **Untested code.** The alignment and other Part D/E scripts in this report are sketches. Torch wasn't installed in the research environment, so they haven't been run. The echo-server sketch (E-Dev) was run against a test client and behaved correctly. The wait-k numbers and simulator were checked by running them.
> - **Next step: Part D.** Record your 6 clips (§D1). Once they're in the repo (`clips/raw/`), the alignment, the entity checker and the wait-k simulator can be built on them.

---

## Part A — The problem: 36 of 70 points ride on not speaking too early

### A1. The system is a no-take-back interpreter scored mostly on meaning

The system listens to live Hindi, freely mixed with English, and speaks English while the caller is still talking, "with meaning, numbers and names intact" ([PS p.1](file:///home/user/interiit_minips/Sarvam.pdf)) [V]. The PS names four difficulties: Hindi is verb-final, so in "मैंने कल अपना पुराना बैंक खाता बंद करवा दिया" the system "cannot tell whether the account was opened, closed or only checked" until "बंद करवा दिया" arrives; speech once played cannot be withdrawn; amounts in lakh and crore, account digits, names, dates, negations and self-corrections must survive; and public Hindi–English data with speech on both sides is scarce ([PS §2](file:///home/user/interiit_minips/Sarvam.pdf)) [V]. Teams build the data before the mid-term, then receive "a streaming Hindi → English speech-to-speech model trained on synthetic data", which they **must** fine-tune using **only their Part A dataset**, with every English target synthesised by **Bulbul v3, voice shubh, en-IN, 24 kHz** ([PS §3 Part B](file:///home/user/interiit_minips/Sarvam.pdf)) [V]. The fine-tuned checkpoint itself must translate; anything outside it that produces Hindi or English text from the audio and "uses it to steer the output" is a banned cascade.

| Stage | Component | Points | Where points are won | Where points are lost |
|---|---|---|---|---|
| Mid-term | Dataset audit | 12 | Clean L1 transcripts, exact Hindi word timings, faithful translations, valid §5.2 format | A dataset that breaks §3 Part A or §5.2 earns nothing ("earns points only if it meets §3 Part A and §5.2") |
| Mid-term | Pipeline reproducibility / report | 3 + 5 | A pipeline that rebuilds records byte-identically from archived inputs | Unarchived LLM/TTS outputs |
| Final | Intent accuracy | 18 | Holding negations, conditions and verbs until heard | A guessed "opened" for "बंद करवा दिया"; a dropped "नहीं" |
| Final | Entity accuracy | 18 | Holding amounts until the unit and any repair are heard | Speaking "five lakh" before "नहीं नहीं, पाँच करोड़"; lakh/crore conversion slips |
| Final | Latency | 16 | Fluent early speech on safe words; fast flush after input_end | Waiting everywhere; English that runs longer than the Hindi and queues up |
| Final | Translation quality | 12 | Targets close to (unknown-style) human references | Over-literal, word-by-word English |
| Final | Audio quality | 6 | Natural Bulbul-like speech without choppy inserted silences | Hard cuts from silence insertion |
| Final | Report + demo | 5 + 5 | Depth and rigour | — |

Two framing facts sharpen this table. First, "KPI components are scored relative to the other teams and the unchanged base checkpoint" ([PS §9.1](file:///home/user/interiit_minips/Sarvam.pdf)) [V], so the floor is *do no harm*: a fine-tune that improves entities but makes the base wait longer or sound choppier can lose net points. Second, the gates are cliffs, not slopes: if the container breaks §5 (G1), if fine-tuning evidence is missing (G2: weights, logs, training-pair manifests with record ID, target text and Bulbul settings, before/after scores), or if records or hours were added after the mid-term (G3), **all 70 KPI points are zero** ([PS §8](file:///home/user/interiit_minips/Sarvam.pdf)) [V]. Protocol robustness therefore comes before any modelling cleverness.

### A2. One session, frame by frame, and why policy lag dominates latency

The harness is the WebSocket client; one connection is one session of 10–60 s ([PS §5.1](file:///home/user/interiit_minips/Sarvam.pdf)) [V]. Audio frames are **80 ms = 1,920 samples × 2 bytes = 3,840 bytes** at 24 kHz s16le, and frame *k* is sent at *t = k × 80 ms*. That frame is exactly one frame of Kyutai's Mimi codec, which "processes 24 kHz audio, down to a 12.5 Hz representation… (latency of 80ms, the frame size)" ([moshi README](https://github.com/kyutai-labs/moshi/blob/main/README.md)) [V] — strong circumstantial evidence, not proof, that the base is Mimi-based. The table walks through a session in which the speaker says the PS sentence; the PS's own example `valid_samples` of 1,234,560 corresponds to 51.44 s of speech, or exactly 643 frames.

| Time | Harness sends | Server must do | What matters |
|---|---|---|---|
| Connect | Opens `/ws` on port 8000 | Accept within 10 s; model already loaded (`/health` 200 within 300 s of start-up) | Load weights at start-up, never on first connection |
| t = 0⁻ | `session_start` (`sample_rate` 24000, `pcm_s16le`, `hi`→`en`) | Reset KV cache and codec state; start **two tasks**: a reader that drains the socket into a queue, and an inference/sender loop | A shared coroutine lets a slow send stall reading |
| t = 0, 80, 160 ms… | Binary frames in real time | int16 → float; encode one codec frame; run one model step per frame; emit nothing (pad/silence) during the initial wait | Real-time factor < 1 on average, or backlog grows without bound |
| Steady state | Frames continue | Send each decoded English chunk at once (≤ 1 s per frame allowed) | Bundling 1 s of audio before sending adds up to 1 s of avoidable lag |
| Harness side | — | Each server frame plays at **max(arrival, previous end)**; gaps are heard as silence | Sending early cannot make speech start before the previous chunk ends |
| Speaker stops | Last frame (zero-padded) | Note the true end of speech from `valid_samples / 24000` | Padding is not speech |
| `input_end` | JSON with `valid_samples`; then silent 80 ms frames keep coming | Keep reading; flush the remaining English as fast as compute allows (Hibiki-style models advance only when fed frames, so feed silence internally) | **Speech after the speaker stops counts toward latency** |
| ≤ 30 s after `input_end` | Silent frames until it sees `session_complete` | Send exactly **one** `session_complete`; nothing may follow | Any debug JSON breaks the format rule |

The latency KPI asks "how far the English speech trails the speaker, and how soon it finishes after the speaker stops", measured on the playback timeline ([PS §7](file:///home/user/interiit_minips/Sarvam.pdf)) [V]. In descending order of size, the drivers are as follows. **Policy lag** — how many Hindi words or seconds the model waits before each English word — is measured in seconds and, for verb-final Hindi, dominated by clause-final verbs; in a Hibiki-like model it is whatever the training alignment taught. **Output-duration compounding** comes next: playback is serialised, so if English takes longer to say than the Hindi it translates, every later word starts at the previous word's end instead of its emission time. Under a hypothetical 300 ms per Hindi word and 350 ms per English word, the wait-3 schedule's time-based AL rises from **787.5 ms to 937.5 ms** and the English finishes **1,000 ms** after the speaker [L, script-verified with SimulEval's formula]. **End offset** is the third: clause-final Hindi verbs mean the final English content ("closed") can only be spoken after the last Hindi word, so a fast flush and concise targets matter. **Compute** matters only when a step exceeds the 80 ms budget, and the **architectural floor** — Moshi's "theoretical latency of 160ms (80ms for the frame size of Mimi + 80ms of acoustic delay)" ([moshi README](https://github.com/kyutai-labs/moshi/blob/main/README.md)) [V] — is negligible next to 2–5 s of linguistic lag. One consequence surprises most people: on this timeline, sending silent PCM and sending nothing are equivalent, except when real audio is queued behind the silence, in which case the silence costs lag. SimulEval's speech-output evaluator implements exactly the same rule, `start = max(prev_end, delay)` ([SimulEval instance.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/instance.py)) [V], so its `*_SpeechAlign` scorers plus EndOffset are the closest open proxy for the official metric.

### A3. Ten rules ranked by what a misreading costs

The PS's protocol rules are numbered 1–6 (Playback, No take-backs, Finishing, Keeping up, Format, Timing), and "a session fails if it breaks rule 3, 4, 5 or 6, or closes early. It gets the worst value on every metric" ([PS §5.1](file:///home/user/interiit_minips/Sarvam.pdf)) [V]. A failed session is a per-session disaster; a container that fails §5 outright is G1 and costs all 70 points.

| # | Rule (PS wording, abridged) | Common misreading | Correct reading and consequence |
|---|---|---|---|
| 1 | "All timing is measured on this playback timeline" | Latency is measured when I send | Start = max(arrival, previous end); a burst queues, it does not play faster |
| 2 | "Speech after the speaker stops counts toward latency"; `session_complete` within 30 s | After `input_end` I can take my time | 30 s is a hard failure cap, not a budget; the tail is scored |
| 3 | "If a frame cannot be delivered within 1 second… because your server is not reading, the session fails" | The 1 s rule is about producing output on time | It is about **reading**; output gaps are allowed and heard as silence |
| 4 | Silent frames continue after `input_end` until `session_complete` | Stop reading after `input_end` | Keep draining, or the reading rule fails the session |
| 5 | `valid_samples` = real samples; the last frame is zero-padded | The last frame is all speech | Use it to locate end of speech and trigger the flush |
| 6 | "The only server-to-client text frame is one session_complete, and nothing may follow it" | A debug or error JSON is harmless | Any extra text frame fails the session |
| 7 | "No take-backs" + FAQ: "Speaking the abandoned value as fact is wrong" | A later correction repairs an early mistake | The reference holds the corrected meaning; the abandoned value is an error |
| 8 | Cascade ban: no outside component that makes Hindi/English text from the audio and "uses it to steer the output" | A small Hindi number detector that delays output is just pre-processing | It produces Hindi text from audio and steers timing: a cascade. Gate on the checkpoint's own text stream instead |
| 9 | Only Part A data for training; English targets Bulbul v3 shubh en-IN 24 kHz, pace/temperature declared; English word timings self-aligned, ordered, non-negative, inside the audio; no records or hours added after the mid-term | Any TTS voice or extra data is fine for fine-tuning | G2/G3 audit these; violation zeroes the final round |
| 10 | Scorer integrity: no "filler speech, padding, or reading out alternatives the speaker did not offer" | Saying "five lakh or five crore" hedges safely | Reading out alternatives can lead to disqualification |

Three smaller rules also catch people. Equivalent number forms are accepted ("25,000 and 25k both count"), but **amounts in the dataset's English keep the speaker's units** (lakh, crore); English words inside the Hindi "must come out correctly in the English speech", so "bank" and product names are not free ([PS §3, §11](file:///home/user/interiit_minips/Sarvam.pdf)) [V]. Real hours count as the union of source-file intervals ("overlapping or repeated cuts of the same recording count once"), while every distinct synthetic file counts, including the same text in another voice, and augmentation adds no hours ([PS §3 Part A](file:///home/user/interiit_minips/Sarvam.pdf)) [V]. Causality is absolute: "reading future audio, fingerprinting test recordings or accessing the network leads to disqualification" ([PS §10](file:///home/user/interiit_minips/Sarvam.pdf)) [V].

### A4. "पाँच लाख… नहीं नहीं, पाँच करोड़।" — speak the amount after the repair's unit plus one beat

The utterance has the classic repair anatomy: a **reparandum** "पाँच लाख", an interruption point "…", an **editing term (interregnum)** "नहीं नहीं", and a **repair** "पाँच करोड़" (terminology from Shriberg's disfluency work, [S: re-check]). The FAQ settles what must be said: "The references encode the corrected meaning. Speaking the abandoned value as fact is wrong" ([PS §11](file:///home/user/interiit_minips/Sarvam.pdf)) [V]. So the only correct English content is "five crore" (or "fifty million", since equivalent forms are accepted).

**When is it safe?** Not at "पाँच": in Western digits 5 lakh = 500,000 and 5 crore = 50,000,000 share no spoken word, and even "five" is unsafe because the number itself can be repaired ("पाँच… नहीं, छह लाख"); speaking it early saves perhaps 300 ms and risks 18 points. Not at "लाख" either: an amount followed by "नहीं" is ambiguous, because "पाँच लाख नहीं दिए" is a *negated predicate* ("did not give five lakh"), itself an intent-critical item, whereas "नहीं नहीं, पाँच करोड़" is a *repair*. The disambiguating cues — a repeated "नहीं", a pause, and a following same-type entity — come only after the first "नहीं". The earliest safe point is therefore **the end of the repair's unit word "करोड़" plus a short look-ahead** (about one word or a 200–500 ms pause, to be tuned) that rules out a second correction; the sentence-final "।" in the transcript corresponds to that pause in audio. Everything around the entity — "I need…", "for the loan…" — can still be spoken without waiting; only the entity span is held.

**How to teach a model to wait only when it must.** For a model whose waiting behaviour comes from training-target timing, five ingredients do it. (1) *Targets* say only the corrected value. (2) *Entity-atomic timing*: treat each entity (amount + unit, name, digit group, date) as one span, and place the first English word of the span no earlier than the end of the **last** Hindi word of the **final** (repair) span plus a margin δₑ. A naive per-word aligner — awesome-align, or Hibiki-style likelihood gain — would align "five" to the reparandum "पाँच", exactly teaching the early "five". (3) *Polarity anchoring*: place the English verb, its auxiliary and any "not" after the Hindi verb group and its negator. (4) *Balanced counter-examples*: inject synthetic repairs (abandoned → "नहीं नहीं" / "सॉरी" / "मतलब" → corrected) alongside confusable non-repairs (amount + negated verb) and plain amounts timed promptly, so the model learns that the trigger to wait is an *unfinished entity or an editing cue*, not entities in general. (5) *A decode-time guard that uses only the checkpoint's own state*: bias the text stream's PAD token up while the model's own inner-monologue text is inside an unfinished number. An external Hindi number detector would be a cascade (rule 8). No source studies streaming translation of self-repairs or the latency cost of waiting for them, which is itself a question worth raising in the interview; the closest evidence is fluent-target training from disfluent speech in Salesky et al. ([arXiv 1906.00556](https://arxiv.org/abs/1906.00556)) [S: re-check].

### A5. Handwritten appendix: wait-3 gives AL = 21/8 = 2.625, not 3

**Definitions.** STACL's prefix-to-prefix model generates target word *t* after reading *g(t)* source words, and wait-k sets **g(t) = min{k + t − 1, |x|}** for 1-indexed *t* ([STACL, arXiv 1810.08398](https://arxiv.org/abs/1810.08398)) [S: re-check typography]. Average Lagging is **AL = (1/τ) Σ_{t=1}^{τ} [ g(t) − (t − 1)/r ]**, with **r = |y|/|x|** and **τ = argmin_t { g(t) = |x| }**, the first target word written after the whole source has been read. SimulEval's `ALScorer` implements exactly this loop (`gamma = target_length / source_length`; `AL += d - t_minus_1 / gamma`; break at the first `d >= source_length`; divide by τ), using the reference length by default ([SimulEval latency_scorer.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)) [V].

**Tokens.** Source x₁…x₉ = मैंने, कल, अपना, पुराना, बैंक, खाता, बंद, करवा, दिया (|x| = 9). Target y₁…y₈ = Yesterday, I, got, my, old, bank, account, closed (|y| = 8). So r = 8/9 and the oracle term (t − 1)/r = (t − 1) × 9/8.

| t | English yₜ | g(t) = min(t+2, 9) | Last Hindi word heard | Hindi source of yₜ | (t−1)·9/8 | g(t) − (t−1)·9/8 |
|---|---|---|---|---|---|---|
| 1 | Yesterday | 3 | अपना | कल (x₂) — heard | 0 | 3.000 |
| 2 | I | 4 | पुराना | मैंने (x₁) — heard | 1.125 | 2.875 |
| 3 | got | 5 | बैंक | करवा दिया (x₈–x₉) — **not heard: anticipation** | 2.250 | 2.750 |
| 4 | my | 6 | खाता | अपना (x₃) — heard | 3.375 | 2.625 |
| 5 | old | 7 | बंद | पुराना (x₄) — heard | 4.500 | 2.500 |
| 6 | bank | 8 | करवा | बैंक (x₅) — heard | 5.625 | 2.375 |
| 7 | account | 9 | दिया (all read) → **τ = 7** | खाता (x₆) | 6.750 | 2.250 |
| 8 | closed | 9 | (after τ, not counted) | बंद (x₇) — heard | — | — |

```text
COPY BY HAND
Schedule (wait-3):  Yesterday | I | got | my | old | bank | account | closed
Hindi words heard:      3     | 4 |  5  | 6  |  7  |  8   |    9    |   9
|x| = 9, |y| = 8, r = |y|/|x| = 8/9, so 1/r = 9/8
tau = first t with g(t) = 9  ->  tau = 7
Sum g(t), t=1..7          = 3+4+5+6+7+8+9            = 42
Sum (t-1)*9/8, t=1..7     = (9/8)(0+1+2+3+4+5+6) = (9/8)(21) = 189/8 = 23.625
AL = (42 - 23.625) / 7    = 18.375 / 7 = 147/56 = 21/8 = 2.625 Hindi words
Check: 3 + 2.875 + 2.75 + 2.625 + 2.5 + 2.375 + 2.25 = 18.375   (ok)
Risk: "got" (t=3) is spoken after 5 words, before "बंद करवा दिया" -> anticipation.
"closed" (t=8) is spoken after all 9 words -> safe.
```

**Why AL is below k.** Because the English is one word shorter than the Hindi (r = 8/9), the ideal translator may spend 9/8 Hindi words per English word, and wait-3's one-in/one-out rhythm gains 1/8 of a word on it every step, so the per-word lag falls from 3 to 2.25 by τ. **AL = k holds only when |x| = |y|**; the common shorthand "wait-k has AL = k" is a misleading claim when lengths differ [L]. Other metrics on the same schedule [L, exact fractions]: LAAL = 2.625 (hypothesis and reference both 8 words); DAL = 3.0 (it forces 9/8-word spacing and gives no length credit); AP = 51/72 ≈ 0.708. Over-generation shows why LAAL exists: a 10-word hypothesis with wait-3 delays still scores AL = 2.625 against an 8-word reference but **LAAL = 3.3** ([FBK LAAL](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/LAAL.md); [SimulEval LAALScorer](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)) [V].

**The off-by-one in the mini PS.** Part E says "English word i is spoken once k + i Hindi words have been heard" ([Mini PS](file:///home/user/interiit_minips/Inter%20IIT%20Tech%20Meet%2015.0_%20Sarvam.ai%20Mini%20PS.pdf)) [V]. With 1-indexed *i* that is wait-(k+1): the schedule becomes 4, 5, 6, 7, 8, 9, 9, 9 with τ = 6 and AL = 59/16 = **3.6875** [L]. It matches STACL only if *i* starts at 0. In the appendix, write STACL's form, state the indexing, and note the alternative.

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 (offline) |
|---|---|---|---|---|---|---|---|---|---|
| AL (this sentence) | 0.5625 | 1.5625 | **2.625** | 3.6875 | 4.75 | 5.8125 | 6.875 | 7.9375 | 9.0 |

Two interview-grade observations follow from this table [L]. Avoiding the anticipation of "got" under a *fixed* policy needs g(3) ≥ 8–9, i.e. k ≥ 6–7 and AL ≥ 5.8, paid on every word. An oracle that waits *only when needed* — Yesterday and I after word 2, then everything after "करवा" (word 8) — has **AL = 41/16 = 2.5625, lower than wait-3 with zero anticipation**, because AL rewards a fast tail; in speech, though, those six tail words must still be *spoken* after the source ends, which is exactly what the KPI's "how soon it finishes" clause penalises. Finally, AL cannot tell "Yesterday I got my old bank account closed" from "I got my old bank account closed yesterday": g(t) depends only on *t*, so both score 2.625, although the second speaks "got" even earlier and delays "yesterday" by about seven words. Alignment-aware metrics such as ATD distinguish them ([SimulEval ATDScorer](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)) [V].

---

## Part B — Four proposals: data, targets, timing and hard content decide the score

The four full proposals below were chosen because they reinforce each other and feed Parts C–E directly: *what audio* (B1), *what English* (B2), *when the English is spoken* (B3) and *whether the hard tokens are right* (B4). Fine-tuning blind (B5) and decoding-time latency (B6) are covered compactly, because before the mid-term both depend on an architecture nobody outside Sarvam has seen.

| Question | Recommendation in one line | Status |
|---|---|---|
| B1 Data sources | ~125 h licence-clean real (IndicVoices-led) + ~65 h entity-dense synthetic + 10 h reserve; augmentation for telephone | Full |
| B2 Training targets | Chunk-wise monotonic, grammatical English as `en_text`; fluent version in `en_alt`; corrected meaning only; 1 Hindi sentence → ≥1 English sentence | Full |
| B3 Timing the English | Earliest-safe-time placement: later of MT-likelihood and word alignment, with verb/negation/entity/repair overrides, + margin δ | Full |
| B4 Hard content | Slot-filled LLM scripts from programmatic samplers, gold entity records generated with the spoken form, two-ASR round trip with zero-tolerance entity match | Full |
| B5 Fine-tuning blind | Neutral pair store + Moshi-format exporter + protocol-faithful eval harness + rehearsal on Hibiki-Zero | Compact |
| B6 Latency | Exact EOS at `valid_samples` + fast flush; gated PAD bias; low text temperature in entity contexts | Compact |

### B1. Data sources: a real majority, because the base has already seen plenty of synthetic speech

**Recommendation.** Spend about **125 h on real speech** (IndicVoices Hindi 105 h, MUCS 2021 Hindi-English 12 h, FLEURS hi_in 5 h, Common Voice or Kathbath 3 h), about **65 h on synthetic speech** (50 h entity-dense code-mixed scripts, 15 h disfluent monologues), and hold **10 h in reserve** for regenerating failed TTS files. Cover telephone and noise conditions with training-time augmentation, which costs zero hours. Exclude Gram Vaani (academic-only or CC BY-NC, depending on the page), Shrutilipi (licence unconfirmed, not verbatim), XTTS-v2 (non-commercial CPML), F5-TTS base weights (CC-BY-NC), MMS-TTS (CC-BY-NC 4.0) and Fish Speech, whose October 2026 licence forbids using outputs "to create, train, distill, or improve… any speech, audio, or voice generation model" ([fish-speech LICENSE](https://github.com/fishaudio/fish-speech/blob/main/LICENSE)) [V].

**Evidence.** IndicVoices is the only source combining a redistributable licence (CC BY 4.0 per [AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/indicvoices.html), [S: re-check on the gated HF card]), spontaneous style (**76% extempore, 15% conversational**, 22,563 speakers across languages) and native **Level-1 verbatim transcripts**, the PS's own convention ([IndicVoices README](https://github.com/AI4Bharat/IndicVoices)) [V]. Kathbath gives **150.2 h of read Hindi under CC0** from 121 speakers ([IndicSUPERB README](https://github.com/AI4Bharat/IndicSUPERB)) [V]; MUCS gives about **89.9 h of real intra-sentential Hindi-English** under CC BY-SA 4.0, but in a narrow software-tutorial domain ([OpenSLR 104](https://openslr.org/104/)) [S: re-check]; FLEURS adds read speech with human English translations under CC BY 4.0 ([google/fleurs](https://huggingface.co/datasets/google/fleurs/commits/4683b04af03d2d9549064c7d72060a9a94bb6046/data/hi_in/test.tsv)) [S: re-check]. On mixing, the literature agrees on direction but not on a ratio: a 50/50 natural/synthetic mix "edged out natural-only" on LibriSpeech while synthetic-only was far worse ([arXiv 1811.00707](https://arxiv.org/pdf/1811.00707)) [S: re-check]; "naïve mixing… often underperforms due to mismatched distributions" ([arXiv 2503.08954](https://arxiv.org/pdf/2503.08954)) [S: re-check]; for code-switching ASR "the crucial ingredient is speaker diversity" ([arXiv 2601.00935](https://arxiv.org/html/2601.00935v1)) [S: re-check]; and a single high-quality TTS voice trained worse ASR than multi-speaker TTS ([Rosenberg et al., arXiv 1909.11699](https://arxiv.org/pdf/1909.11699)) [S: re-check]. Licence-clean open TTS offers perhaps 20–50 voices (Indic Parler-TTS Apache 2.0, Veena Apache 2.0, IndicF5 MIT with permission-only cloning, Svara-TTS if its licence is confirmed) against thousands of IndicVoices speakers. Since the base checkpoint is "trained on synthetic data" ([PS §3](file:///home/user/interiit_minips/Sarvam.pdf)) [V], the marginal value of our data is highest exactly where it has seen least: real, noisy, conversational Hindi. Bulbul's Hindi voices are a legitimate extra synthetic source: §10 explicitly lets outputs generated with competition credits "be used to build your data, to train and evaluate models for this competition, and to be shared with Sarvam", though not published or redistributed ([PS §10](file:///home/user/interiit_minips/Sarvam.pdf)) [V].

**Main risk.** The 105 h IndicVoices figure is unconfirmed. Hindi-only transcribed hours appear nowhere primary; the only number found is a third-party aggregate of **300 h / 163,428 rows** ([fhai50032/asr-hindi](https://huggingface.co/datasets/fhai50032/asr-hindi)) [S: re-check], and conversational files may hold two voices on one channel, which would breach "one speaker per clip". The fallback is to move a shortfall to Kathbath while keeping synthetic at or below about 45%.

**Cheapest confirming experiment.** One CPU day: download the IndicVoices Hindi manifest, run VAD and a speaker-count heuristic (or diarisation) on the conversational subset, and count single-speaker extempore hours after trimming to speech spans; at the same time spot-check 50 L1 transcripts for English-in-Devanagari and numbers-as-words. Before the checkpoint exists, the ratio question can be probed with an ASR proxy: fine-tune a small Hindi ASR on 10 h mixes at 0/33/50% synthetic and score WER and entity exact-match on a real, speaker-disjoint dev set.

**Question no source answers.** What real-to-synthetic ratio is optimal when fine-tuning an S2ST model that was *already trained on synthetic data*? Every ratio study found concerns ASR from scratch. A residual, narrower question also remains about Sarvam API outputs: §10 permits them in data shared with Sarvam, but whether hosting them in the private Hugging Face repo and archiving them for byte-identical reproducibility counts as "redistribution" is not spelled out, so ask once in writing.

**Claim that turned out wrong.** The research brief listed AI4Bharat's Rasa as a Hindi source, but Rasa contains **only Assamese, Bengali and Tamil** ([Rasa README](https://github.com/AI4Bharat/Rasa)) [V]. Runners-up: IndicVoices is quoted as 7,348 h (paper v1), 12,000 h (README) or 23.7K h (HF card), all *collected* totals whose transcribed share is far smaller (1,639 h, 3,200 h, 11.2K h) ([README](https://github.com/AI4Bharat/IndicVoices)) [V]; and AIKosh lists Indic Parler-TTS as MIT where the model card says Apache 2.0 ([card](https://huggingface.co/ai4bharat/indic-parler-tts-pretrained)) [S: re-check].

### B2. Training targets: chunk-wise monotonic English that never guesses

**Recommendation.** Store two English versions per Hindi sentence, which the PS allows ("A sentence may carry several English translations", [PS §3](file:///home/user/interiit_minips/Sarvam.pdf)) [V]. The **training target** (`en_text`) is *chunk-wise monotonic but grammatical*: keep the Hindi chunk order (time adverbials, topics, objects) wherever English tolerates fronting, use cheap verb-deferring constructions (causative "got X closed", fronted adverbials, a topicalised object with a resumptive pronoun only when the object is long), and never produce word-by-word English. The **fluent** version goes in `en_alt` as a hedge and as the reference for measuring what monotonicity costs. Both encode only the **corrected meaning**, keep **lakh/crore in words** in one canonical form, and make every negation and condition explicit. Map **one Hindi sentence to one or more English sentences**, never pulling content across a Hindi sentence boundary; split run-ons rather than merging.

**Evidence.** Simultaneous models trained on full-sentence references with long reorderings "either make bold hallucinations (for fixed-latency policies) or introduce long delays (for adaptive ones)"; pseudo-references made by wait-k decoding gave up to **+2.7 BLEU** on Zh→En and Ja→En ([Chen et al., arXiv 2010.11247](https://arxiv.org/abs/2010.11247)) [S: re-check]. LLM-generated chunk-wise monotonic translations (LLM-SI-Corpus) "reduce latency while achieving better quality" for En→Ja speech-to-text ([arXiv 2404.12299](https://arxiv.org/abs/2404.12299)) [S: re-check], and Simul-MuST-C found that the benefit of monotonicity "depends on the grammatical distance" between the languages ([ACL 2024.emnlp-main.1238](https://aclanthology.org/2024.emnlp-main.1238/)) [S: re-check] — Hindi→English (SOV→SVO, postpositions→prepositions) is a distant pair. The PS's own gloss "Yesterday I got my old bank account closed" already defers the lexical verb with a causative ([PS §2](file:///home/user/interiit_minips/Sarvam.pdf)) [V]. Fluent targets from disfluent speech are an established end-to-end recipe ([Salesky et al., arXiv 1906.00556](https://arxiv.org/abs/1906.00556)) [S: re-check]. The other side of the ledger: offline-style references **under-score** monotonic output, as NAIST found at IWSLT 2024 ([ACL 2024.iwslt-1.30](https://aclanthology.org/2024.iwslt-1.30.pdf)) and Simul-COMET argues ([ACL 2026.findings-acl.2110](https://aclanthology.org/2026.findings-acl.2110/)); one study reports a **13.83-BLEU** gap for the same output scored against interpretation vs translation references [S: re-check, single study].

**Main risk.** The quality KPI (12 points) compares against "human reference translations" of unknown style, and naturalness (6) is listener-rated; aggressive topicalisation could cost more there than it saves in latency (16) and intent (18).

**Cheapest confirming experiment.** Take 200 Part A sentences. Have an LLM produce both versions under a CWMT-style guideline, run the Part D entity and negation checker as a filter, then measure the *cost* (BLEU/chrF/COMET of the monotonic version against the fluent one) and the *benefit* (mean earliest-safe lag and premature-commit rate under the B3 rule, computed with the Part E simulator). A decision rule: if the monotonic version loses more than about 10 BLEU against fluent while saving under about 0.5 s of mean lag, move toward fluency. This costs a few dollars of LLM calls and no GPU.

**Question no source answers.** Are Sarvam's human references offline-style or interpretation-style, and is quality scored with BLEU, COMET or an LLM judge? That decides how much monotonicity is penalised, and the PS gives only prose.

**Claim that turned out misleading.** Hibiki-Zero's title promises translation "Without Aligned Data", and headlines say it "eliminates the need for word-level alignments entirely" ([MarkTechPost](https://www.marktechpost.com/2026/02/13/kyutai-releases-hibiki-zero-a3b-parameter-simultaneous-speech-to-speech-translation-model-using-grpo-reinforcement-learning-without-any-word-level-aligned-data/)). The paper itself "only requires sentence-level translation alignment easily obtained from punctuation", and builds those coarse alignments **with silence insertion** before an RL stage cuts latency ([arXiv 2602.11072](https://arxiv.org/abs/2602.11072)) [S: re-check]. Sentence mapping is still a prerequisite — which is why the 1:≥1 sentence correspondence above must be stored explicitly.

### B3. Timing the English: the placement rule *is* the policy

**Recommendation.** Build a *placement engine* that, for each English word eᵢ, computes an earliest safe Hindi time and places one contiguous Bulbul rendering so that no word starts before it. The steps: (1) compute aᵢᴹᵀ, the Hindi position whose arrival most raises the MT log-likelihood of eᵢ (Hibiki's contextual alignment, using MADLAD-400, IndicTrans2 rescoring or an LLM as a training-time teacher, which the FAQ allows); (2) compute aᵢᵂᴬ, the last aligned Hindi index from awesome-align or SimAlign for content words; (3) apply overrides — negators and auxiliaries go to the Hindi verb group, entity spans are atomic and anchored to their *last* Hindi word, repairs anchor to the *last* matching span, and multi-word amounts wait for their final multiplier ("पाँच लाख पचास हज़ार" waits for "हज़ार"); (4) take aᵢ = cummax(max(aᵢᴹᵀ, aᵢᵂᴬ)), which is conservative and monotone; (5) set the safe time sᵢ = end(Hindi word aᵢ) + δ, with δ swept over {0.5, 1, 2} s; (6) synthesise the sentence once with Bulbul at a declared pace, force-align it, and insert silence only at Bulbul's own prosodic breaks so that offset + onsetᵢ + inserted silence ≥ sᵢ, raising pace or re-synthesising per chunk when cumulative lag exceeds a cap (e.g. 4 s); (7) shift the English word timestamps by the same offsets and assert they are ordered, non-negative and inside the audio. Also export a sentence-level schedule (each English sentence starts after its Hindi sentence ends), which is never premature within a sentence and is the safe fallback if the base turns out to be Hibiki-Zero-like.

**Evidence.** Hibiki's README states the rule this engine generalises: "a word should only appear in the target once it's predictable from the source", applied "either by inserting silences or by synthesizing targets with a voice controlled, alignment-aware TTS" ([hibiki README](https://github.com/kyutai-labs/hibiki)) [V]. Extracts of the paper add that the target lags "by at least 2 seconds compared to the contextual alignment" and that spikes "higher than 25% of the average delay over a window of 5 words" are removed ([arXiv 2502.03382](https://arxiv.org/pdf/2502.03382)) [S: re-check — another extract says the lag was varied from 0.5 to 3 s]. TransLLaMa inserts WAIT tokens so that "target content words never appear earlier than their closest equivalents in the source" ([arXiv 2402.04636](https://arxiv.org/abs/2402.04636)) [S: re-check]. Unsupervised English–Hindi word alignment is noisy, with an **alignment error rate of about 0.39–0.44** ([OTTAWA, arXiv 2406.01919](https://arxiv.org/abs/2406.01919)) [S: re-check], which is why the rule takes the later of two aligners. If the base is Moshi-like, the public fine-tuning format is a stereo WAV (left = model output, right = user input) plus `{"alignments": [[word, [start, end], "SPEAKER_MAIN"], …]}` per file, turned into a 12.5 Hz text stream by an interleaver ([moshi-finetune README](https://github.com/kyutai-labs/moshi-finetune); [interleaver.py](https://github.com/kyutai-labs/moshi-finetune/blob/main/finetune/data/interleaver.py)) [V] — our English word times play exactly that role. On English, MFA puts about **72.8% of TIMIT word boundaries within 25 ms**, against 52.7% for WhisperX and 43.5% for MMS ([Rousso et al., arXiv 2406.19363](https://arxiv.org/pdf/2406.19363)) [S: re-check]; all are far finer than the 80 ms text-stream frame, so aligner *robustness* on numbers, "lakh/crore" and Indian names matters more than precision. Write numbers as words before aligning, because WhisperX gives no timing to tokens like "2014." ([WhisperX README](https://github.com/m-bain/whisperX)) [V].

**Main risk.** The base may have been trained with a different lag convention, so fine-tuning on a new one could first *raise* errors; silence insertion produces the chunk-by-chunk pauses listeners dislike ([NaturalFlow, arXiv 2606.13121](https://arxiv.org/abs/2606.13121)) [S: re-check]; and alignment noise makes schedules either premature or over-conservative.

**Cheapest confirming experiment.** No GPU, about one day: hand-mark, for 50 sentences plus the six Part D clips, the Hindi word after which each English content word is first "safe". Compute schedules for wait-k (k = 2…8), sentence-level, Hibiki MT-only and the combined rule at each δ, and report (a) the **premature-commit rate** separately for verbs, negations and entities, (b) LAAL and end offset on the Hindi timeline, and (c) total inserted silence and the count of pauses over 300 ms. The rule with the lowest (a) at a given (b) wins; once the checkpoint arrives, confirm with a two-arm LoRA run that differs only in the timing rule.

**Question no source answers.** What timing convention (word-level contextual alignment with a lag floor, or sentence-level plus RL) did Sarvam use for the base, and does a different per-word lag rule help or destabilise it with ≤ 200 h? Nothing can answer this before the mid-term reveal.

**Claim that turned out misleading.** The torchaudio multilingual forced-alignment tutorial — the one the mini PS recommends — carries the banner "The APIs described in this tutorial are deprecated in 2.8 and will be removed in 2.9" ([tutorial source](https://github.com/pytorch/audio/blob/main/examples/tutorials/forced_alignment_for_multilingual_data_tutorial.py)) [V], and one research note concluded from it that torchaudio must be pinned to ≤ 2.8. The removal was **reversed**: the 2.10 release notes say `forced_align` is "preserved and will remain in torchaudio" ([torchaudio releases](https://github.com/pytorch/audio/releases)) [V], and the **torchaudio 2.11.0 wheel still ships `functional.forced_align`, `merge_tokens` and `pipelines.MMS_FA`** ([torchaudio 2.11.0 on PyPI](https://pypi.org/project/torchaudio/2.11.0/)) [V, wheel inspected]. What *was* removed is `torchaudio.io`, `sox_effects` and `torchaudio.backend`, and `torchaudio.load` now runs on TorchCodec — which is what breaks other tools (see Part E-SP).

### B4. Hard content: generate entities with their gold values, then distrust the TTS

**Recommendation.** Write entity-heavy content as **slot-filled LLM call scripts** (banking, UPI, telecom recharge, e-commerce returns, insurance, loans, travel) in which programmatic samplers, not the LLM, choose the entities: names across regions, cities and PIN codes, organisations and products in Devanagari; amounts drawn log-uniformly from ₹10 to ₹100 crore and spoken with साढ़े/सवा/पौने/डेढ़/ढाई; 10-digit mobile numbers starting 6–9 spoken digit-by-digit, in pairs, with "डबल"/"ट्रिपल" or English digit words; account, IFSC and OTP strings; dates and times; negations (नहीं, मत, ना, कभी नहीं) and conditionals (अगर…तो, वरना). Generate the Devanagari Level-1 spoken form and a structured **gold record** (type, value, span, and for repairs the *abandoned* value as a forbidden entity) together, never by post-hoc normalisation. Inject repairs from a small Hindi editing-term inventory ("नहीं नहीं", "सॉरी", "मतलब", "एक मिनट", "अरे"), add negation minimal pairs, and send TTS the same number-as-words string that becomes the transcript. Gate every TTS clip with a two-ASR round trip and zero-tolerance entity matching (thresholds in Part C), regenerating up to three times before dropping. Cap name reuse (each name ≤ 5 uses) and hold out entity values and templates between train and dev.

**Evidence.** TTS-generated numeric utterances cut WER on the longest numeric sequences by **up to 8×** at Google ([Peyser et al., arXiv 1907.01372](https://arxiv.org/pdf/1907.01372)) [S: re-check], and TTS audio helped OOV entities at Amazon ([arXiv 2011.11564](https://arxiv.org/pdf/2011.11564)) [S: re-check] — but transfer is weakest exactly here: an Indic entity "flywheel" reached a Telugu Entity-Hit-Rate of 0.473 yet only **0.337 on Hindi against a 0.65 target**, below commercial systems, with real-speech validation on just 20 Telugu utterances ([arXiv 2605.03073](https://arxiv.org/abs/2605.03073)) [S: re-check; single-author preprint]. Labels must match what the TTS actually said; that study had to rewrite digits as spelled words. LLM-prompted disfluency (DRIVE) beats LARD-style rule insertion on naturalness ([arXiv 2507.19867](https://arxiv.org/html/2507.19867v1)) [S: re-check], and LARD inserts only one disfluency per utterance ([LARD, ACL 2022.lrec-1.249](https://aclanthology.org/2022.lrec-1.249/)) [S: re-check]. Negation is a disproportionate MT error source, with quality drops above 60% in some directions ([arXiv 2010.05432](https://arxiv.org/pdf/2010.05432)) [S: re-check]. For the checker's word table, NVIDIA NeMo text-processing 1.2.0 ships Hindi ITN with हज़ार…शंख magnitudes, साढ़े/सवा/पौने/डेढ़/ढाई handling, and **220 spellings for the values 10–99** ([nemo-text-processing 1.2.0](https://pypi.org/project/nemo-text-processing/1.2.0/)) [L, files inspected]. Hindi round-trip ASR is itself imperfect — IndicWhisper scores **10.3 WER on Kathbath and 26.8 on Gramvaani telephone audio** ([Vistaar README](https://github.com/AI4Bharat/vistaar)) [V] — so a plain WER gate over-rejects good clips.

**Main risk.** Entity templates overfit: the same 50 names, round amounts and carrier phrases teach the model to copy priors instead of listening, and the Hindi transfer result above warns that the synthetic entity slice may help less than hoped.

**Cheapest confirming experiment.** Before any model training: render 200 sampled amounts and 100 phone numbers through each candidate TTS voice, run the two-ASR round trip and the entity parser, and have a human listen to 50 clips. This measures each voice's entity error rate and the gate's precision and recall in a day. After the checkpoint arrives, sweep 0/5/10/20% entity-TTS hours and score entity accuracy on 30–50 *real recorded* entity clips.

**Question no source answers.** At what round-trip pass threshold, and what fraction of entity-dense TTS hours in a 200 h mix, does a streaming Hindi→English S2ST model's real-speech entity accuracy stop improving or start degrading? The only nearby evidence measured ASR, in Telugu, on 20 utterances.

**Claim that turned out wrong.** AI assistants routinely suggest `num2words(…, lang='hi')` for Hindi verbalisation; **num2words 0.5.14 raises `NotImplementedError` for `hi`** (it has bn, kn, te and en_IN, which renders 2,500,000 as "twenty-five lakh") ([num2words on PyPI](https://pypi.org/project/num2words/)) [L]. Its parsing cousin is worse because it fails silently: `word2number` 1.1 contains a `'lakh': 100000, 'crore': 10000000` table that is never used, so **`word_to_num("five lakh")` returns 5** and "nine eight seven six" returns 85 ([word2number on PyPI](https://pypi.org/project/word2number/)) [L]. `indic-num2words` uses non-standard spellings (उन्तालिस, निन्यान्बे) and falls back to digit pairs at or above 100 crore ([indic-num2words](https://pypi.org/project/indic-num2words/)) [L].

### B5. Fine-tuning blind (compact): build what transfers to any architecture

**Recommendation.** Build a **neutral pair store** (Parquet/JSONL per sentence: Hindi wav, `hi_words`, Bulbul English wav per pace/temperature, English word timings, the Hindi↔English alignment, entity spans with positions on both sides, speaker, split) with thin exporters, one of which writes Moshi-format stereo WAV plus `alignments` JSON. Add the B3 placement engine with margin, entity wait and pace as knobs, so several lag regimes can be generated after the reveal; an **eval harness that replays §5.1 in real time** and reconstructs the playback timeline; a reference server skeleton (Part E-Dev); and LoRA scaffolding with adapter export and Modal checkpointing. Rehearse end to end by fine-tuning **Hibiki-Zero 3B** on 10–20 h of our pairs: it is public, built on the same `moshi` library, and "8 GB VRAM should work, 12 GB is safe" ([hibiki-zero README](https://github.com/kyutai-labs/hibiki-zero)) [V]; its CC BY-NC-SA weights are fine for practice and are never submitted. **Evidence.** moshi-finetune's example config uses LoRA rank 128, scaling 2, **LR 2e-6** (about 50–100× below typical LoRA rates, suggesting fragility), weight decay 0.1, 100 s windows, batch 16, 2,000 steps, codebook-0 loss ×100 and text-padding weight 0.5, and warns that "lowering `duration_sec` may make the model go silent sooner" ([moshi_7B.yaml](https://github.com/kyutai-labs/moshi-finetune/blob/main/example/moshi_7B.yaml); [README](https://github.com/kyutai-labs/moshi-finetune)) [V]. At the README's ≈12k tokens/s on one H100 for Moshi 7B, one epoch of 200 h is ≈1.9 H100-hours, about **$7.4 at $3.95/h** ([Spheron on Modal pricing](https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/)) [S: confirm in the Modal dashboard] — the $1,000 budget is limited by evaluation time and iteration count, not GPU-hours. **Main risk.** Fine-tuning drifts the base's lag or voice; mitigate by measuring the base's own lag distribution on our dev set first and generating pairs that match or slightly beat it, freezing the codec, starting with LoRA on the temporal transformer only, early-stopping on intent/entity/lag rather than loss, and an optional KL term toward the frozen base (an allowed teacher). **Cheapest experiment.** On reveal day, run the unchanged base through our harness on the real dev set before training anything. **Unanswered question.** Does the base expose lag or quality conditioning, an input-EOS token, or an acoustic delay? **Misleading claim.** The moshi README says "we do not support quantization for the PyTorch version", yet lists `moshiko-pytorch-q8` (experimental) and ships a bitsandbytes int8 `QLinear` ([moshi README](https://github.com/kyutai-labs/moshi/blob/main/README.md); [quantize.py](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/utils/quantize.py)) [V]; and an aggregator headline's "$0.92 an hour" for Modal H100s has no Modal-specific rate behind it ([keenable](https://select.keenable.ai/r/nvidia-h100-cloud-cost-real-price-per-gpu-hour/MYEY81nBL3-DIQrqXC_g-g)).

### B6. Latency at decode time (compact): flush fast, gate the PAD bias, cool the entities

**Recommendation.** Cheapest first: (1) on `input_end`, locate the last real frame from `valid_samples`, inject an input-EOS if the base accepts one (Hibiki feeds a frame of code 2048, then silence, and stops on text EOS — [run_inference.py](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/run_inference.py) [V]), and then step the model on internally generated silence as fast as the GPU allows, streaming ≤ 1 s frames; (2) apply word-preserving post-processing only when audio is queued behind earlier audio (trim internal pauses over ~250 ms, mild 1.05–1.2× WSOLA speed-up) — silence trimming is explicitly allowed, and time-stretch arguably is because it "does not change its words", but confirm with the organisers; (3) use a slightly faster declared Bulbul pace in training targets; (4) sweep a PAD/EPAD logit bias β, gated off while the model's *own* text stream is inside an unfinished number or before the clause verb; (5) lower the text temperature below Moshi's default 0.7 (or top-k 1–5) in entity contexts, keeping audio temperature ~0.8 ([lm.py](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/lm.py)) [V]; (6) optionally a *self-consistency wait*: sample B parallel text continuations and commit a word only when ≥ k agree. Beam search is incompatible with no take-backs, CFG is an audio-quality knob that "can lead to worse translations" ([hibiki README](https://github.com/kyutai-labs/hibiki)) [V], and quantisation is not a latency lever when a ≤ 7B model already fits in 80 ms on an H100. **Main risk.** No study measures how PAD bias trades lag against entity accuracy in a Moshi-class model. **Cheapest experiment.** A β sweep on the dev set, plotting lag against entity, intent and BLEU. **Unanswered question.** How exactly Sarvam times English output words against Hindi input words for "trails the speaker", and how that is weighted against "how soon it finishes". **Misleading claim.** An AI search summary said Hibiki's LAAL is computed "without requiring word-level alignments", then in the next sentence that it comes "from word-level emission timestamps produced by Whisper" — LAAL needs per-word delays by construction ([SimulEval latency_scorer.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)) [V].

### B7. Original idea: "counterfactual-continuation" safe times

**The idea.** Hibiki-style contextual alignment asks *when does the true English word become likely?* given the true future-free prefix. Under no take-backs the question that matters is different: *could anything a speaker might still say overturn this word?* So, for each Hindi prefix x≤ⱼ, sample M plausible continuations of the Hindi with an LLM — deliberately including a calibrated share of repairs ("नहीं नहीं, …"), negations and late verbs — translate each completed hypothetical sentence, and declare English word eᵢ **safe at j** once it appears, aligned, in at least p (say 90%) of the continuations' translations. The fraction itself is a per-word *overturn risk*, which sets the margin δ only where risk is high: "Yesterday" becomes safe at "कल", "got…closed" stays unsafe until "बंद करवा दिया", and "five" after "पाँच लाख" stays unsafe for as long as continuations with repairs keep changing it.

**Why it should work here.** It optimises the quantity the KPIs punish — committed words later contradicted — rather than a likelihood proxy, and it treats verbs, negations and repairs with one mechanism instead of hand-written overrides. External models are explicitly allowed as teachers during training ([PS §11](file:///home/user/interiit_minips/Sarvam.pdf)) [V], and the output is just better timings for the same Bulbul audio, so it costs no hours and leaves the submitted system cascade-free.

**First-week test.** On the six Part D clips plus 100 Part A sentences with hand-marked safe points (the B3 experiment), compute safe times by wait-k, Hibiki-style likelihood gain (IndicTrans2 or an LLM as scorer) and counterfactual continuation (M = 20), then run all three through the Part E simulator. Call it a success if it cuts premature commits on verbs, negations and entities by at least 30% at matched LAAL (a threshold I am proposing, not one from a source).

**When it would fail.** It fails if the LLM's continuations are unrepresentative of real speech — under-generating repairs and code-mixing, or over-generating them, which would make the model wait everywhere; if lexical variation across continuations' translations ("closed"/"shut") is mistaken for disagreement, which needs lemma or embedding matching; if the cost explodes (20 continuations × 15 prefixes × tens of thousands of sentences is millions of generations, so run it on a calibration subset and fit a cheap per-word risk model for the rest); and if the base turns out to be trained at sentence level, in which case fine-grained timings matter less than the sentence mapping of B2.

---

## Part C — The dataset: 125 real hours, 65 synthetic hours, and gates that check entities, not just format

The dataset "decides most of the score" because the base model's waiting, its entity handling and its robustness to real Hindi will all be learned from it, and because only it may be used for training ([Mini PS Part C](file:///home/user/interiit_minips/Inter%20IIT%20Tech%20Meet%2015.0_%20Sarvam.ai%20Mini%20PS.pdf)) [V]. Each record must carry 24 kHz mono 16-bit Hindi audio from one speaker, a verbatim IndicVoices Level-1 transcript (disfluencies kept, English words in Devanagari without bracketed Latin, numbers as spoken words, sentence-final punctuation), start and end times for every word, and sentence-level English translations, in the §5.2 JSONL format with no empty source field ([PS §3, §5.2](file:///home/user/interiit_minips/Sarvam.pdf)) [V].

### C1. Sources and generation: IndicVoices for realism, samplers plus open TTS for entities

The licence screen in B1 leaves a short list. The table gives the 200-hour split, with licences, usable hours and why each bucket exists.

| Bucket | Hours | Source and licence | Why |
|---|---|---|---|
| R1 real extempore + single-speaker conversational | **105** | IndicVoices Hindi, CC BY 4.0 (gated; verify card) | Real, spontaneous, many speakers, native L1 transcripts; disfluencies and repairs occur naturally ([README](https://github.com/AI4Bharat/IndicVoices)) [V] |
| R2 real code-mixed | **12** | MUCS 2021 Hi-En (SLR104), CC BY-SA 4.0, 89.9 h available | Real intra-sentential English; small because the domain is technical ([OpenSLR 104](https://openslr.org/104/)) [S: re-check]; ShareAlike binds this subset |
| R3 real read with human English | **5** | FLEURS hi_in, CC BY 4.0 (+ optional Kathbath, CC0, 150.2 h) | Free human references; speaker diversity ([IndicSUPERB](https://github.com/AI4Bharat/IndicSUPERB)) [V] |
| R4 real, optional | **3** | Common Voice hi (CC0) or SPRING-INX if its licence is confirmed | Extra speakers |
| S1 synthetic entity-dense, code-mixed | **50** | LLM scripts seeded by BPCC-H-Daily (CC BY 4.0, 139K human-translated daily sentences) and PHINC (CC BY 4.0, 13,738 human Hinglish→English pairs), voiced by Indic Parler-TTS, Veena, IndicF5 (permissioned voices), Svara, Bulbul hi-IN | Lakh/crore, digits, dates, negations, conditions, repairs are rare in found speech ([IndicTrans2 README](https://github.com/AI4Bharat/IndicTrans2); [PHINC](https://arxiv.org/pdf/2004.09447)) |
| S2 synthetic disfluent monologues | **15** | LLM scripts with fillers, repetitions, false starts | Mirrors the verbatim style; teaches the model to ignore fillers |
| Reserve | **10** | — | Regenerate failed TTS without breaching 200 h |
| **Total** | **200** | | ≈62% real / 33% synthetic / 5% reserve |

Four hour-accounting tactics follow from the counting rules. Trim each real clip to its speech span before writing `source.start`/`end`, because silence inside the interval counts. Cut as many overlapping streaming windows from one IndicVoices file as useful, because overlapping cuts count once. For synthetic data, give each voice a *different* text, because re-voicing the same text doubles hours for little new content. And treat IndicVoices-R as the same recordings as IndicVoices: it adds no hours, and its enhancement may raise "is this still the real recording?" questions ([IndicVoices-R](https://github.com/AI4Bharat/IndicVoices-R)) [V].

**How others generated code-mixed or entity-heavy speech.** Microsoft's GCM toolkit generates code-mixed text from parallel pairs under Equivalence-Constraint and Matrix-Language theory ([CodeMixed-Text-Generator](https://github.com/microsoft/CodeMixed-Text-Generator)) [V], and X-RiSAWOZ used it for English–Hindi with word-alignment substitution as a fallback ([arXiv 2306.17674](https://arxiv.org/pdf/2306.17674)) [S: re-check]. LLM prompting is the practical alternative, and a 2025 study filtered generated code-mix by M-index and I-index ≥ 0.1 ([ACL 2025.emnlp-main.109](https://aclanthology.org/2025.emnlp-main.109.pdf)) [S: re-check] — but these indices measure *how much* mixing there is, not how natural it is, and automatic naturalness prediction for Hinglish is weak ([HinglishEval](https://aclanthology.org/2022.inlg-genchal.3.pdf)) [S: re-check], so a human must spot-check. LLM-written two-speaker conversations voiced by conversational TTS beat classical synthesis for Whisper fine-tuning but did not match real in-domain data ([Cornell et al., arXiv 2408.09215](https://arxiv.org/abs/2408.09215)) [S: re-check]. Hibiki made its synthetic pairs "by translating and resynthesizing the transcript of single-language audio" ([hibiki README](https://github.com/kyutai-labs/hibiki)) [V] — real source speech, synthetic target speech — which is the exact shape of our R buckets. **Recommendation:** LLM scripts with sampler-filled slots and gold records (B4), CMI ≥ 0.1 on a share of the code-mixed slice, Devanagari transliteration of Roman-script seeds (PHINC and Hinglish-TOP are Roman), and Bulbul Hindi voices as one of several TTS sources, which §10 permits.

### C2. Transcripts, timings and translations: every tool needs a human at the edges

No off-the-shelf model writes the PS's Level-1 format directly, no aligner has a published Hindi word-boundary accuracy, and no MT benchmark uses disfluent, Devanagari-code-mixed transcripts. The table gives what each tool reportedly achieves, with the protocol stated, because numbers from different protocols are not comparable.

| Task | Tool | Reported accuracy (protocol) | Use | Caveat |
|---|---|---|---|---|
| Verbatim ASR | Sarvam `saaras:v3`, `mode="verbatim"` | ~19.3% WER on a 10-language IndicVoices aggregate, vendor-reported, **not Hindi-only** ([Sarvam STT](https://www.sarvam.ai/speech-to-text)) [S: re-check] | First draft; documented to keep fillers and spoken numbers | `codemix` mode returns English in Latin; verbatim's script for English words is undocumented — test it |
| Verbatim ASR | IndicConformer-600M (MIT, CTC+RNNT) | Community repack: 13.5 WER Kathbath, 15.2 FLEURS-hi [S: re-check] | Second draft; its CTC head can align | Needs AI4Bharat's NeMo fork ([README](https://github.com/AI4Bharat/IndicConformerASR)) [V]; dropped words beyond ~18 s reported unofficially |
| Verbatim ASR | IndicWhisper (Vistaar) | Hindi WER 10.3 Kathbath, 12.0 Kathbath-Hard, 12.0 MUCS, 26.8 Gramvaani, avg 13.6 ([Vistaar](https://github.com/AI4Bharat/vistaar)) [V] | Benchmark reference | The README swaps Hindi CommonVoice/FLEURS values between two tables |
| Verbatim ASR | Whisper large-v3 | 27.5 WER FLEURS-hi ([Trelis card](https://huggingface.co/Trelis/whisper-hinglish-preview)) [S: re-check]; "implicit disfluency removal" ([arXiv 2309.07927](https://arxiv.org/pdf/2309.07927)) [S: re-check] | Avoid for verbatim | Deletes fillers |
| Verbatim ASR | Omnilingual ASR 7B LLM-ASR | `hin_Deva` CER 3.6 (CER, not WER) ([results CSV](https://github.com/facebookresearch/omnilingual-asr/blob/main/per_language_results_table_7B_llm_asr.csv)) [V] | Third opinion | < 40 s audio |
| Hindi alignment | torchaudio 2.11 `MMS_FA` (uroman input) | No Hindi figure; on English TIMIT, MMS puts 43.5% of boundaries within 25 ms ([arXiv 2406.19363](https://arxiv.org/pdf/2406.19363)) [S: re-check] | **Primary for Part D** (CPU, ~40 lines) | Weights CC-BY-NC 4.0; digits are deleted unless written as words; 20 ms frames [V] |
| Hindi alignment | `ctc-forced-aligner` (from GitHub) | Same MMS family | Cross-check | `pip install ctc-forced-aligner` installs a *different* ONNX project ([PyPI 1.0.2](https://pypi.org/project/ctc-forced-aligner/1.0.2/)) [V] |
| Hindi alignment | WhisperX `hi` (`theainerd/Wav2Vec2-large-xlsr-hindi`) | WhisperX 52.7% within 25 ms on English TIMIT [S: re-check] | Optional | No timing for out-of-vocabulary tokens ([alignment.py](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py)) [V] |
| Hindi alignment | NeMo Forced Aligner + IndicConformer CTC | None found | Native Devanagari, no romanisation | Inference, untested ([NFA](https://github.com/NVIDIA/NeMo/tree/main/tools/nemo_forced_aligner)) |
| Hindi alignment | MFA / Qwen3-ForcedAligner | — | **Not usable for Hindi** | MFA has a Hindi dictionary but no Hindi acoustic model ([mfa-models](https://github.com/MontrealCorpusTools/mfa-models/blob/main/docs/source/acoustic/index.rst)); Qwen3 aligner covers 11 languages, no Hindi ([Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)) [V] |
| English (Bulbul) alignment | MFA english_us_arpa + MMS_FA cross-check | ~19 ms mean boundary error on TIMIT ([arXiv 2610.17782](https://arxiv.org/abs/2610.17782)) [S: re-check] | Primary for training-pair timings | Add pronunciations for "lakh", "crore", Indian names; flag > 80 ms aligner disagreement |
| Translation | IndicTrans2 (MIT; 1B and distilled 200M) | NTREX Hindi→English 41.3 BLEU [S: re-check] | Open baseline and back-translation | Not benchmarked on disfluent code-mixed input ([README](https://github.com/AI4Bharat/IndicTrans2)) [V] |
| Translation | LLM with a CWMT-style prompt | Large-unit number errors up to ~20% for Llama-3.1-8B in zh↔en ([arXiv 2501.04927](https://arxiv.org/pdf/2501.04927)) [S: re-check] | **Primary** for monotonic + fluent pairs | Must be gated by the entity and negation checker |
| Translation | Sarvam `mayura:v1` (code-mixed mode) / `sarvam-translate:v1` (`numerals_format`) | No public Hindi figures | Optional second opinion | Allowed in data per §10 |

Human labour is still needed in five places, and a human-in-the-loop budget should be planned for them. Fillers, repetitions and false starts must be restored, because Whisper-family models delete them and no "verbatim" mode is verified for Hindi. The spoken form of each number and date must be chosen ("ढाई लाख" versus "दो लाख पचास हज़ार", whichever was said). English loanwords need consistent Devanagari spelling: an LLM with a lookup cache beats IndicXlit here, whose English→Hindi top-1 accuracy is only 55.6–60.6% because it was trained on romanised *Indic* words ([IndicXlit](https://github.com/AI4Bharat/IndicXlit)) [V]. Sentence segmentation is judgement. And a stratified sample of word boundaries must be hand-checked, because the 30-boundary check in Part D will be the only Hindi alignment evidence anyone has. Level 1 means *phonetic fidelity*: keep a colloquial "मतबल" rather than standardising it to "मतलब", which is a Level-2 operation ([IndicVoices](https://arxiv.org/html/2403.01926v1)) [S: re-check].

### C3. Validation: hard gates on entities and polarity, soft scores for routing review

Published builders filter with explicit thresholds. IndicVoices-R kept speech with SNR ≥ 25 dB, C50 ≥ 30 dB, 0.2–30 s duration, pitch mean ≤ 350 Hz, ≤ 30 characters/s and CER ≤ 5% between transcript levels ([IndicVoices-R, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/7dfcaf4512bbf2a807a783b90afb6c09-Paper-Datasets_and_Benchmarks_Track.pdf)) [S: re-check]. Emilia kept DNSMOS P.835 OVRL > 3.0, required language-ID confidence ≥ 80% and removed per-source duration outliers outside 1.5×IQR, keeping only 29.43% of a 600 h subset ([arXiv 2407.05361](https://arxiv.org/pdf/2407.05361)) [S: re-check]. Seed-TTS-style evaluation pairs Whisper-large-v3 WER with WavLM speaker-verification cosine ([arXiv 2406.02430](https://arxiv.org/html/2406.02430v1)) [S: re-check]. For translations, COMETKiwi is used as a reference-free filter without a standard cut-off ([arXiv 2508.14951](https://arxiv.org/pdf/2508.14951)) [S: re-check], and the ACES challenge set dedicates a whole category to number and named-entity errors that metrics miss ([arXiv 2401.16313](https://arxiv.org/pdf/2401.16313)) [S: re-check]. The adopted checks adapt these to Hindi, where the round-trip ASR itself has about 4–5% CER on clean speech:

| Check | Applies to | Threshold (adopted) | Action |
|---|---|---|---|
| Two-ASR round trip (IndicConformer + Saaras or Omnilingual), NFC + nukta/anusvara folding | TTS Hindi | Keep if best-of-two **CER ≤ 8–10%**; review if WER > 25% or CER > 15% (≈2× the ASRs' own CER; IndicVoices-R's 5% was for human transcripts) | Regenerate ≤ 3×, then drop |
| Entity exact match via the Hindi parser | TTS Hindi | **Zero tolerance** for amounts, digit strings, dates; names by normalised edit distance ≤ 0.2 on romanised form; abandoned value never in a corrected position | Regenerate, or relabel to what was said and propagate to English |
| Skip/repeat | TTS Hindi | chars/s within the per-voice 1.5×IQR fence and ≤ 30; > 2 aligner insertions or an n-gram repeated > 2× not in the script; bottom 1–2% per-word alignment log-prob | Regenerate |
| Signal quality | Clean TTS render, before augmentation | DNSMOS OVRL ≥ 3.0; > 0.1% clipped samples flagged; leading/trailing silence ≤ 1 s; no unscripted internal silence > 2 s | Regenerate |
| Speaker consistency | Cloned voices | WavLM-SV cosine above the 5th percentile of real same-speaker pairs (no published Hindi threshold) | Drop |
| Language ID | Code-mixed clips | **Do not** apply Emilia's LID ≥ 80%; instead check that scripted English words come back as Devanagari English | Review |
| Hindi word timings | All records | Ordered, non-negative, inside clip, every transcript word timed; two-aligner disagreement > 100 ms → review | Fix or re-align |
| English entity multiset | Translations | Canonical values equal to the Hindi gold record; lakh/crore kept in surface form | Hard fail → retranslate |
| Negation polarity and conditions | Translations | Hindi negator count (नहीं/मत/ना/न/कभी नहीं/बिना, minus tag-question ना and repair "नहीं नहीं") ≤ English negator count; अगर/वरना/जब तक ↔ if/otherwise/until present | Mismatch → LLM yes/no polarity check → review |
| Self-repair | Translations | `corrected_present` true and `abandoned_leaked` false for the training target | Hard fail |
| QE and back-translation | Translations | Bottom 5–10% of COMETKiwi per batch, and IndicTrans2 back-translation re-checked by the entity checker | Route to review |
| Human audit | Everything | Listen to 1–2% of accepted clips, stratified by voice and entity type; report the gate's precision and recall | Recalibrate thresholds |

### C4. Real-world conditions: augment the channel, diversify the voices, measure on real Hindi

The synthetic–real gap is large and well documented for ASR. Synthetic-only training reportedly gave about 50% dev-clean WER against about 5% for natural speech in an early LibriSpeech study, attributed largely to limited speaker variety ([Rossenbach et al. 2019 via awesomepapers](https://awesomepapers.io/speech-audio/papers/rossenbach2019generating)) [S: re-check, secondary]; the gap depends strongly on the speaker embedding ([arXiv 2407.17997](https://arxiv.org/pdf/2407.17997)) [S: re-check]; and channel effects are hard for TTS to capture ([Synth2Aug, arXiv 2011.11818](https://arxiv.org/abs/2011.11818)) [S: re-check]. For Hindi specifically, the channel gap is visible in Vistaar: every system's WER on Gramvaani telephone-quality audio is **2–3× its clean Kathbath WER** (IndicWhisper 26.8 vs 10.3; Google STT 59.9 vs 14.3) ([Vistaar](https://github.com/AI4Bharat/vistaar)) [V]. No Hindi- or S2ST-specific real-vs-synthetic ablation exists.

The methods that close the gap are cheap here because augmentation adds no hours. Speed perturbation at 0.9/1.0/1.1 gave an average **4.3% relative WER gain** across four tasks ([Ko et al. 2015](https://www.danielpovey.com/files/2015_interspeech_augmentation.pdf)) [S: re-check]. Add noise at 0–20 dB SNR (self-recorded fan, traffic and TV babble have no licence issues; MUSAN/DEMAND/ESC-50 terms were not verified), real and simulated room impulse responses, and a telephone chain: 24 kHz → 8 kHz, 300–3400 Hz band-pass, G.711 μ-law/A-law, GSM, AMR-NB 4.75–12.2 kbps or Opus 6–12 kbps, 1–5% packet loss, then back to 24 kHz, matching the PS's "lower-bandwidth recordings are upsampled" ([PS §4](file:///home/user/interiit_minips/Sarvam.pdf)) [V]. `phonesim` runs real codecs including AMR through ffmpeg and defaults to 24 kHz ([phonesim](https://github.com/deepmark/phonesim)) [V]; a stock Ubuntu ffmpeg 6.1 has μ-law, A-law, GSM and Opus encoders but **no AMR-NB encoder** [L]. Encoding training audio through AMR-NB and G.711 gave clear gains on 8 kHz test data for Mandarin ASR ([arXiv 2010.12715](https://arxiv.org/pdf/2010.12715)) [S: re-check]. On the content side, inject disfluencies into TTS scripts and maximise voice count, since diversity is the documented main cause of the gap.

**Measuring the gap.** Keep a **speaker-disjoint real Hindi dev set** that includes telephone and noisy clips (Part D-style recordings, plus held-out IndicVoices speakers). Report train-synthetic/test-real against train-real/test-real performance — ASR WER and entity exact-match before the checkpoint, ASR-BLEU, entity and intent accuracy after it — and, as a cheap secondary diagnostic only, a Fréchet speech distance on WavLM embeddings between the synthetic and real pools, a metric with no agreed embedding and no validation as a predictor of training gains ([arXiv 2601.21386](https://arxiv.org/abs/2601.21386v1)) [S: re-check].

### C5. Failure modes: eight ways to pass the audit and still train a worse model

Similar datasets have failed through TTS repetition and skips, labels that do not match the spoken verbalisation, low speaker diversity, synthetic data that does not transfer (worst in Hindi), metrics blind to entities and negation, and aggressive filtering that discards most of the data (Emilia kept under a third). Speech-translation systems got only **37–40% of person names** right on NEuRoparl-ST, and entity accuracy correlated only loosely with overall translation quality ([Gaido et al., EMNLP 2021](https://aclanthology.org/2021.emnlp-main.128/)) [S: re-check]. The mid-term audit samples transcripts, Hindi timings, translations and format, so the dangerous failures are the ones it cannot see:

| Failure | Why the audit passes | Why the model gets worse | Guard |
|---|---|---|---|
| Entity templates overfit (50 names, round amounts, fixed carrier phrases) | Every record is well-formed and correct | The model copies priors instead of listening | Long-tailed samplers, reuse caps, held-out entity values and templates |
| Too clean (read-style TTS, DNSMOS > 3, no channel simulation) | Clean audio audits beautifully | Hidden test is real, noisy, telephone Hindi | Filter the clean render, then augment at training time |
| Abandoned values timed early | Translation text is correct | Naive alignment of "five" to the reparandum teaches speaking it first | Repair-aware entity-atomic timing (A4, B3) |
| Literal targets keep "five lakh, no no, five crore" | Faithful and arguably correct | Teaches the streaming model to voice the abandoned value | Training target = corrected meaning; literal only in `en_alt` |
| Inconsistent English number forms ("5 lakh", "500,000", "half a million") | Each is a valid translation | Unstable outputs and diffuse learning | One canonical form in training targets, keeping lakh/crore |
| Speaker leakage (cloned dev speakers in train) | No format violation | Inflated dev scores hide real-speech weakness | Speaker-disjoint splits; never clone dev voices |
| Round-trip ASR shares the TTS's biases | Round trip says "pass" | Mispronounced numbers enter training | Two ASRs from different families + 1–2% human audit |
| Timings legal but systematically late (e.g. a fixed 3 s δ everywhere) | Ordered, non-negative, inside audio | Fine-tuning teaches the base to wait longer than it did, losing latency points relative to it | Measure the base's lag first; δ only where risk is high |

### C6. The recommended pipeline, as one diagram

```text
                 ┌──────────────── SOURCES (licence-screened, data_card.md) ─────────────────┐
                 │ REAL: IndicVoices hi (CC BY) · MUCS Hi-En (CC BY-SA) · FLEURS hi · CV/Kathbath │
                 │ TEXT: BPCC-H-Daily · PHINC (→ Devanagari) · entity samplers · repair grammar   │
                 └───────────────┬──────────────────────────────────────┬───────────────────────┘
                                 │ real audio                           │ LLM scripts + GOLD records
                                 ▼                                      │ (entities, forbidden values,
             VAD trim · 1-speaker check · resample to 24 kHz s16        │  speech-act labels)
                                 │                                      ▼
                                 │                     open TTS (Parler, Veena, IndicF5, Svara) + Bulbul hi
                                 │                     same spoken-form string → audio AND transcript
                                 │                                      │
                                 │                     TTS GATE: 2-ASR CER ≤ 8–10% · entity exact match ·
                                 │                     skip/repeat · DNSMOS ≥ 3 · ≤3 retries → drop
                                 ▼                                      ▼
         L1 TRANSCRIPT: ASR draft (Saaras verbatim + IndicConformer) → human fix (fillers, numbers as words,
                        English in Devanagari, danda)            [synthetic: transcript = script]
                                 │
         HINDI WORD TIMES: MMS_FA (uroman) ⟷ cross-aligner; >100 ms disagreement → human; legality asserts
                                 │
         TRANSLATION: LLM → {monotonic en_text, fluent en_alt}; IndicTrans2 back-translation
         TRANSLATION GATE: entity multiset · negation/conditions · repair flags · COMETKiwi bottom 5–10% → review
                                 │
         HOURS LEDGER (union of real intervals; every synthetic file) ≤ 200 h → §5.2 JSONL → private HF repo
         + archive of every transcript/translation/TTS output for byte-identical rebuild        ◄── MID-TERM AUDIT
                                 │
   ───────────── after mid-term (no new records or hours) ─────────────────────────────────────────────
                                 ▼
         Bulbul v3 shubh en-IN 24 kHz (declared pace/temp) → English forced alignment (MFA + MMS check)
                                 │
         PLACEMENT ENGINE: safe time = cummax(max(MT-gain, word-align)) + overrides + δ → silence at breaks
                                 │
         EXPORT: Moshi-style stereo + alignments JSON (or base's format) + G2 manifest (record ID, text, Bulbul)
                                 │
         TRAIN (LoRA, augmentation: noise · RIR · speed · telephone codecs)  →  EVAL on real speaker-disjoint
         dev via §5.1 replay harness: entity · intent · LAAL/EndOffset on playback timeline · ASR-BLEU
```

### C7. Three open questions to test first

The first question gates the whole budget: **how many usable, single-speaker, extempore Hindi hours does IndicVoices actually hold, and do its Level-1 transcripts already meet the PS's extra rules** (English in Devanagari with no bracketed Latin, numbers as words)? If the answer is under 105 h, or the transcripts need heavy rewriting, the split and the human-labour plan both change; the one-day manifest count in B1 answers it. The second is **how accurate open aligners are on code-mixed Hindi and on Bulbul English** — especially around fillers, Devanagari-written English words, "लाख/करोड़", digit strings and Indian names — because the audit checks Hindi timings and the placement engine depends on English ones; scale the Part D 30-boundary protocol to about 200 stratified boundaries on both sides. The third is **whether entity-dense synthetic speech transfers to real Hindi entity accuracy, and at what share** of the 200 h; probe it with the B4 TTS calibration now and an ASR-proxy sweep (0/5/10/20% entity-TTS) scored on real recorded entity clips, and confirm with the fine-tuned model after the mid-term.

---

## Part D — Build plan: six clips, one aligner, one checker, every number reproducible

Part D asks for six 20–40 s clips of natural code-mixed Hindi containing lakh/crore amounts, a phone number, names, a negation and self-corrections; Level-1 transcripts and word timings from an open aligner, with 30 boundaries hand-checked; literal and fluent translations; and a checker that verifies amounts, numbers, names and negations, with its accuracy and failure cases ([Mini PS Part D](file:///home/user/interiit_minips/Inter%20IIT%20Tech%20Meet%2015.0_%20Sarvam.ai%20Mini%20PS.pdf)) [V]. Because "in the interview you will run your code on a new clip and explain every part of it", the plan favours small scripts you can read aloud over heavyweight toolkits, CPU over GPU, and cached outputs so nothing depends on the network.

### D0. Repository layout and environment

```text
mini_ps/
├── README.md                    # how to run everything, in order
├── RESULTS.md                   # every number + the command that produced it
├── Makefile                     # make env | make clip C=clip07 | make results
├── requirements.txt             # main env (pins below)
├── requirements-dfn.txt         # separate py3.11 env, only if doing the SP track
├── clips/{raw,24k,16k}/         # phone recordings and converted copies
├── transcripts/L1/clipNN.txt    # deliverable: one sentence per line, ends with । or ?
├── transcripts/align_view/clipNN.json   # display token -> alignment token map
├── align/align_mms.py           # ~40 lines: MMS_FA + uroman, writes words.json + Audacity labels
├── align/eval_boundaries.py     # compares hand labels vs aligner, prints the error table
├── align/out/  align/hand_labels/
├── translate/translate_llm.py   # literal + fluent JSON, cached in translate/cache/
├── translate/translate_it2.py   # IndicTrans2 200M baseline (CPU)
├── translations.jsonl
├── checker/data/{digit,teens_and_ties}.tsv   # vendored from NeMo 1.2.0 (keep its licence notice)
├── checker/{hindi_numbers,en_numbers,entities,negation}.py
├── checker/gold/clipNN.json     # hand-labelled entities, negations, repairs
├── checker/eval_checker.py      # confusion matrix per entity type
├── manifest/{build_manifest,validate}.py    # §5.2 JSONL for the 6 clips (shows Part A understanding)
├── tests/test_numbers.py        # unit tests incl. the 12 prototype phrases + known hard cases
├── partE_ml/{wait_k_sim,prefix_translate,plot}.py
└── run_clip.py                  # one command for the interview: new clip -> all outputs
```

Use Python 3.11 or 3.12 with `torch==2.11.*`, `torchaudio==2.11.0`, `soundfile`, `uroman==1.3.1.1`, `rapidfuzz==3.14.6`, `numpy`, `scipy`, `matplotlib`, `transformers` and `IndicTransToolkit==1.1.1`, all current on PyPI on 14 Oct 2026 ([torchaudio 2.11.0](https://pypi.org/project/torchaudio/2.11.0/); [uroman](https://pypi.org/project/uroman/); [rapidfuzz](https://pypi.org/project/rapidfuzz/); [IndicTransToolkit](https://pypi.org/project/IndicTransToolkit/)) [V]. Load audio with `soundfile` and pass tensors, because `torchaudio.load` now depends on TorchCodec and system FFmpeg. Make the first line of `make env` a smoke test — `python -c "import torchaudio; from torchaudio.pipelines import MMS_FA; print(torchaudio.__version__)"` — and pre-download the ~1.18 GB MMS_FA weights and the IndicTrans2 model so the interview demo runs offline.

### D1. Record and convert six clips designed for coverage

Speak from a bullet outline rather than a script, so the speech stays natural and produces real fillers and repairs; record on a phone in a quiet room, one speaker, with your own consent noted in the README. Design the six clips so every checker category appears at least twice:

| Clip | Scenario | Planted content |
|---|---|---|
| 01 | Closing an old bank account, loan amount | "पाँच लाख… नहीं नहीं, पाँच करोड़"; a negated verb ("बंद नहीं हुआ") |
| 02 | Giving a callback number | 10-digit number with "डबल", one English digit word, and a digit repair ("नौ आठ सात… सॉरी, नौ आठ छह") |
| 03 | Booking for a relative | Two person names, a city, an organisation and a product in Devanagari; a date ("पंद्रह अगस्त") |
| 04 | EMI dispute | A condition ("अगर कल तक… तो…"), a prohibition ("मत भेजना"), a tag-question "ना?" that is *not* a negation |
| 05 | Property budget | Fractions: "साढ़े तीन लाख", "सवा", "डेढ़ करोड़"; a time "साढ़े चार बजे" |
| 06 | Heavy code-mix, phone-call audio | An English clause inside Hindi; record via a real phone call for the SP track |

```bash
for f in clips/raw/*.m4a; do b=$(basename "$f" .m4a)
  ffmpeg -y -i "$f" -ac 1 -ar 24000 -sample_fmt s16 -c:a pcm_s16le clips/24k/$b.wav   # PS format
  ffmpeg -y -i clips/24k/$b.wav -ar 16000 -c:a pcm_s16le clips/16k/$b.wav             # MMS_FA / Silero
done
ffprobe -v error -show_entries format=duration -of csv=p=0 clips/24k/*.wav            # 20–40 s each
```

### D2. Transcribe in IndicVoices Level-1 style

Get a draft from Saaras v3 `verbatim` or IndicConformer if convenient, then correct by ear; a fully manual transcript of 3–4 minutes of audio is also feasible. Apply the rules as a checklist and quote them in the README: exactly what was said, with fillers (अं, हम्म, मतलब), repetitions and false starts kept; phonetic spelling ("मतबल" stays "मतबल"); English words in Devanagari as spoken (अकाउंट, ईएमआई), with no bracketed Latin; numbers as the spoken words ("पाँच करोड़", "नौ आठ डबल सात"); every sentence ending in । or ?; noise tags optional and, if used, excluded from alignment ([PS §3](file:///home/user/interiit_minips/Sarvam.pdf)) [V]. Save one sentence per line in `transcripts/L1/clipNN.txt`, NFC-normalised (nukta letters such as ढ़ exist in two Unicode forms, and NFC unifies them [L]).

### D3. Align with MMS_FA, keeping an alignment view that maps 1:1 to the display words

MMS_FA's alphabet is lowercase Latin plus apostrophe and a `*` star token, at 16 kHz, so Devanagari must be romanised with uroman (`lcode='hin'`), and anything that romanises to an empty string — punctuation, digits — must be dropped and its time interpolated ([torchaudio 2.11.0 wheel](https://pypi.org/project/torchaudio/2.11.0/); [tutorial](https://github.com/pytorch/audio/blob/main/examples/tutorials/forced_alignment_for_multilingual_data_tutorial.py)) [V]. Local uroman output shows what the model actually sees: राहुल → raahul, पाँच → paamc, करोड़ → karodd, बैंक → baimk, अकाउंट → akaaumtt, and **"98765" stays as digits, which the tutorial's normaliser deletes** — one more reason the transcript writes numbers as words [L]. The core of `align_mms.py` is short enough to explain line by line (sketch; test before relying on it):

```python
import json, re, sys, soundfile as sf, torch, uroman
from torchaudio.pipelines import MMS_FA as B          # torchaudio 2.11: still shipped

wav, sr = sf.read(sys.argv[1], dtype="float32"); assert sr == B.sample_rate == 16000
words = [w for line in open(sys.argv[2], encoding="utf-8")
         for w in re.sub(r"[।?,.!]", " ", line).split()]
U = uroman.Uroman()
rom = [re.sub(r"[^a-z']", "", U.romanize_string(w, lcode="hin").lower().replace("’", "'")) for w in words]
keep = [i for i, r in enumerate(rom) if r]             # words with an empty romanisation get interpolated
model, tok, aligner = B.get_model(), B.get_tokenizer(), B.get_aligner()
x = torch.from_numpy(wav).unsqueeze(0)
with torch.inference_mode():
    em, _ = model(x)
spans = aligner(em[0], tok([rom[i] for i in keep]))   # one list of TokenSpans per word
sec = x.size(1) / em.size(1) / sr                      # seconds per emission frame (~20 ms)
out = [{"w": words[i], "start": round(s[0].start * sec, 3), "end": round(s[-1].end * sec, 3)}
       for i, s in zip(keep, spans)]
json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
```

A companion step writes the same intervals as an Audacity label file (`start\tend\tword`, UTF-8) and asserts the legality rules the audit will apply: ordered, non-negative, inside the clip, every transcript word timed. Run `ctc-forced-aligner` (installed from GitHub, not PyPI) as a cross-check and log words where the two disagree by more than 100 ms.

### D4. Hand-check 30 boundaries blind, and report error the way the literature does

Pick 30 boundaries, five per clip, over-sampling the hard cases: English–Hindi switch points, the boundary inside an amount (पाँच | करोड़), digit groups, the "नहीं नहीं" repair, and words next to pauses. Mark them in Audacity on a label track **before** importing the aligner's labels, to avoid anchoring, and export the tab-separated `start end label` file ([Audacity manual](https://manual.audacityteam.org/man/importing_and_exporting_labels.html)) [S: re-check], naming each point label `wordIndex:s` or `wordIndex:e`. `eval_boundaries.py` computes the signed error e = t_aligner − t_hand and reports mean and median |e|, the mean signed error (bias), the share within 20/50/100 ms, the count of gross errors above 200 ms, starts and ends separately (CTC models are often biased differently on each), and a bootstrap 95% interval, stated as wide because n = 30. Put the result in context with two floors: MMS emissions are 20 ms frames, so ±10 ms is the resolution limit [V], and human labellers themselves disagree by roughly 16 ms on average in Buckeye ([arXiv 2403.19509](https://arxiv.org/pdf/2403.19509)) [S: re-check]. No published Hindi figure exists, so your number is the only evidence anyone has — say so.

### D5. Translate twice, with two systems

Prompt an LLM for a JSON object `{"literal": …, "fluent": …}` per sentence. Literal keeps Hindi order where English allows, keeps every disfluency and the repair ("five lakh… no no, five crore"), and keeps amounts in lakh/crore; fluent is idiomatic and states only the corrected value. Cache every response under `translate/cache/` keyed by a hash of the sentence and prompt, so reruns are byte-identical. Run IndicTrans2's distilled 200M model as an open, CPU-friendly baseline (about 0.8 GB in fp32), sentence by sentence: `IndicProcessor(inference=True)` → `preprocess_batch(src_lang="hin_Deva", tgt_lang="eng_Latn")` → tokenizer → `generate(num_beams=5, max_length=256)` → `postprocess_batch` ([IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit)) [V]. Whether IndicTrans2 renders "पाँच लाख" as "five lakh" or "500,000", and how it handles Devanagari-written English words, is undocumented, which makes it a reportable finding. Store `clip, sent_id, hi_L1, en_literal, en_fluent, system` per line in `translations.jsonl`.

### D6. Build the checker from the prototype, fix its known failures, and score it against human verdicts

Start from `/home/user/interiit_minips/prototypes/hindi_number_parser_prototype.py`, a ~70-line parser that loads NeMo 1.2.0's Hindi `numbers/digit.tsv` and `numbers/teens_and_ties.tsv` (passed as a directory argument), NFC-normalises, maps Devanagari digits to ASCII, adds English-in-Devanagari digit words (वन…नाइन) and "डबल", treats सौ/हज़ार/लाख/करोड़/अरब as multipliers and साढ़े (+0.5), सवा (+0.25), पौने (−0.25), डेढ़ (1.5) and ढाई (2.5) as fractions, and reads three or more bare digit words as a digit string. Vendor the two TSVs into `checker/data/` so the checker is plain Python with no pynini dependency:

```bash
pip download nemo_text_processing==1.2.0 --no-deps -d /tmp/nemo && cd /tmp/nemo && unzip -oq *.whl
find /tmp/nemo -path '*hi/data/numbers/*' -name '*.tsv' | grep -E 'digit|teens_and_ties'   # copy these two
```

The prototype parses its 12 test phrases correctly — पच्चीस हज़ार → 25000; ढाई लाख → 250000; साढ़े तीन करोड़ → 35000000; पौने दो लाख → 175000; सवा पाँच सौ → 525; एक करोड़ बीस लाख पचास हज़ार → 12050000; a spoken 10-digit number → `9876543210`; नाइन एट डबल सात फाइव → `98775`; and "पाँच लाख नहीं नहीं पाँच करोड़" → two amounts ([hard-content research note, NeMo tables](https://pypi.org/project/nemo-text-processing/1.2.0/)) [L]. Re-running it in this session against stub tables exposed four failures to fix and to list in RESULTS.md as known limits [L]: **"दो तीन लाख" (two-to-three lakh) returns 500,000** because short digit runs are summed, so detect adjacent single digits before a multiplier and emit a RANGE; **"साढ़े चार बजे" returns 4** because the fraction is dropped without a multiplier, so treat a number followed by बजे as TIME (साढ़े X = X:30, सवा X = X:15, पौने X = (X−1):45); **"एक एक" returns the digit string 11**, which can also mean "one each", so flag it as ambiguous outside a number context; and fractional results come back as floats (250000.0), so cast to int. Repair logic goes on top: when two amounts of the same type are separated by an editing term within a few words, the last is `corrected` and the first `abandoned`.

The rest of the checker is four small modules. `en_numbers.py` parses English digits with Indian or Western grouping ("5,00,000", "500,000"), number words, "k", "lakh/lac", "crore", "million", "billion" and decimals ("2.5 crore") to an integer — do **not** use `word2number`, which returns 5 for "five lakh" [L]. Amounts compare by value, so "five crore", "50 million" and "₹5 crore" all match 5×10⁷, with a separate unit-preservation flag because the dataset rule keeps lakh/crore. Digit strings compare exactly, with digit edit distance as a secondary score. Names are romanised with uroman and normalised (lowercase, w→v, sh→s, ph→f, collapse doubled letters), then matched with `rapidfuzz.fuzz.partial_ratio` at ≥ 85, a threshold set from a five-name local test (it lifted राहुल/शर्मा/अग्रवाल scores from 91/92/67 to 100/100/86) and to be re-tuned and reported ([rapidfuzz](https://pypi.org/project/rapidfuzz/)) [L]. Negation counts Hindi cues (नहीं, मत, न, ना, कभी नहीं, बिल्कुल नहीं, ना कि) minus tag-question "ना?" and repair "नहीं नहीं", and requires at least as many English cues (not, n't, no, never, neither, nor, without, instead of) in the aligned sentence. Self-repair scores two flags, `corrected_present` and `abandoned_leaked`, where the abandoned value may appear in the literal version only before the repair marker.

Hand-label `checker/gold/clipNN.json` with each entity's type, Hindi span and canonical value, each negation and each repair, then hand-label a *human verdict* for every (entity, English output) pair across the four outputs (literal/fluent × LLM/IndicTrans2). The checker is a binary classifier of "entity preserved", so report its accuracy, precision and recall against the human verdicts per entity type as a confusion matrix, and separately report each translation system's entity accuracy *as measured by the checker*. The failure list to look for and report: value-equivalence switched off ("fifty lakh" vs "5 million"), fractions mistranslated ("quarter to three lakh"), phone numbers chunked differently, a name that is also a common noun translated literally ("Kamal" → "lotus"), double negation and tag-question ना, negation scope ("I didn't say five lakh" vs "I said not five lakh"), and English that hedges or drops the abandoned value.

### D7. RESULTS.md structure and the interview runbook

```markdown
# RESULTS
Commit: <hash> · Python 3.12 · torch 2.11.x · torchaudio 2.11.0 · CPU: <model> · date
## D1 Clips            | clip | duration s | entities | negations | repairs | code-mix tokens | command
## D2 Alignment        | metric (starts / ends / all) | mean |e| ms | median | bias | ≤20 | ≤50 | ≤100 | >200 | 95% CI | command
## D3 Translation      | system × version | entity acc (checker) | negation kept | repair correct | command
## D4 Checker          | entity type | n | accuracy | precision | recall | confusion matrix | command
## D4 Failure cases    | clip:sent | phenomenon | checker said | human said | why
## E  Track results    | k | AL_s | LAAL_s | StartOffset | EndOffset | entity acc | intent proxy | command
## Known limitations   (ranges, times, "एक एक", stub-table caveats, n = 30 CI width)
## Reproduce all       make results   # regenerates every table above from cached inputs
```

Every row's "command" cell is the exact shell line, and `make results` regenerates every table from the archived transcripts, labels and cached translations. For the interview, `python run_clip.py new.wav --transcript new.txt` converts the clip, aligns it, writes Audacity labels, translates it (live if online, else IndicTrans2 locally), runs the checker and draws the wait-k timeline. Typing a 30 s Level-1 transcript live takes a few minutes; an `--asr-draft` flag can prefill it from an offline IndicConformer if one is installed. Rehearse once on a clip recorded by a friend, since the new clip will not be in your voice.

---

## Part E — Track plans: ML reuses Part D, Dev and SP are fallbacks

**Recommendation: take the ML track.** Its inputs are exactly Part D's outputs (Hindi word times, translations, checker), so the effort compounds; it produces the plot that answers the question the whole PS is about — where a streaming model should wait — and it rehearses Part B's timing proposal on real clips. One wording slip to note: the mini PS's ML hints say "use your Part C word timings" and "your Part C entity checker", but both are built in Part D ([Mini PS Part E](file:///home/user/interiit_minips/Inter%20IIT%20Tech%20Meet%2015.0_%20Sarvam.ai%20Mini%20PS.pdf)) [V].

### E-ML. Simulate wait-k on the playback timeline and plot entity accuracy against lag

**Schedule.** For each English word *i* (1-indexed), the ready time is the end of Hindi word min(k + i − 1, J) — STACL's convention — plus an optional compute delay δ (e.g. 0.2 s); once the source is exhausted, every remaining word is ready at the end of speech X. Playback is serialised exactly as in §5.1: start₁ = ready₁ and startᵢ = max(readyᵢ, startᵢ₋₁ + durᵢ₋₁). Give each word a fixed 0.35 s, or measure durations from Bulbul audio of the English sentence aligned with MMS_FA. Add a `--ps-index` flag that uses k + i instead, and report that it equals wait-(k+1), which is the off-by-one in A5.

```python
def schedule(src_end, n_tgt, k, dur, delta=0.0, ps_index=False):
    """src_end: end time (s) of each Hindi word; returns playback start of each English word."""
    J, starts, free = len(src_end), [], 0.0
    for i in range(1, n_tgt + 1):
        need = k + i if ps_index else k + i - 1
        ready = src_end[min(need, J) - 1] + delta
        s = max(ready, free); starts.append(s); free = s + dur[i - 1]
    return starts, free                                  # free = end of the last English word

def al(delays, X, ref_len, laal=False):                 # SimulEval ALScorer / LAALScorer logic
    if delays[0] > X: return delays[0]
    gamma = (max(len(delays), ref_len) if laal else ref_len) / X
    tot = 0.0
    for t, d in enumerate(delays):
        tot += d - t / gamma
        if d >= X: return tot / (t + 1)
    return tot / len(delays)
```

A unit test pins the implementation to A5: with `src_end = [1, 2, …, 9]`, zero durations, k = 3 and an 8-word target, `al(...)` must return 2.625, and with `ps_index=True` it must return 3.6875. Report AL_s and LAAL_s on playback start times, StartOffset = start₁, and **EndOffset = end of the last English word − X**, since the PS scores "how soon it finishes".

**Three regimes.** (1) *Oracle shift*: re-time the full-sentence fluent translation for each k; entity accuracy is constant and only lag moves, isolating the latency side. (2) *Prefix translation*: for each step *i*, translate the Hindi prefix w₁…w_{k+i−1} with the LLM, given the English already committed, and commit only word *i* (STACL-style prefix-to-prefix); this is the regime where small k speaks "five lakh" before the repair, guesses the verb, or drops a negation. Cache every call to JSON so the interview demo runs offline; with six clips and k ∈ {1, 2, 3, 5, 7, ∞} it is a few hundred calls. (3) *Adaptive hold*, an oracle analysis policy: wait-3 everywhere, except hold the first word of each entity span until its last Hindi word (the repair's, for corrected entities), and hold the verb, auxiliary and negator until the Hindi verb group, using the gold annotations. This is an analysis tool, not a deployable component: in the real system the same behaviour must be learned from training-pair timing, because an outside Hindi detector steering output would be a cascade.

**Plots and the answer to "where should it wait?"** Plot entity accuracy (y) against AL_s (x), one point per k and one line per regime, with per-type curves (amount, phone, name, negation); a second panel plots EndOffset against k; a timeline figure shows Hindi words on one row and English on another for clip 01 at k = 2 versus k = 5 versus adaptive hold. The hypothesis to test is that errors concentrate where the deciding token *follows* the content — the unit after "पाँच", the repair after "लाख", the verb and negator at clause end — so the adaptive policy should reach near-oracle entity accuracy at a lag close to wait-3, whereas fixed k needs about 6–7 on verb-final clauses (A5). If the data show that, the conclusion writes itself: wait at numbers until the multiplier and any repair arrive, at clause ends for polarity, and nowhere else.

### E-Dev (compact). An echo server that never stops reading

Use `websockets` 17.2 (Python ≥ 3.11) with its asyncio implementation, a `process_request` hook for `/health`, separate reader and sender tasks joined by an `asyncio.Queue`, absolute-deadline scheduling from `loop.time()`, and `compression=None`, since permessage-deflate is on by default and wastes CPU on incompressible PCM ([websockets 17.2](https://pypi.org/project/websockets/17.2/); [health-check FAQ example](https://github.com/python-websockets/websockets/blob/main/example/faq/health_check_server.py)) [V]. This sketch holds back one frame so it can trim the zero-padding named by `valid_samples`, reads and ignores the post-`input_end` silence, sends exactly one `session_complete`, and never closes first. Run against a local test client with websockets 17.2, it returned `/health` 200, echoed exactly `valid_samples` × 2 bytes and sent a single `session_complete` [L]:

```python
import asyncio, json
from http import HTTPStatus
from websockets.asyncio.server import serve
DELAY = 1.0                                              # echo delay, seconds (> one 80 ms frame)

def http(conn, req):
    if req.path == "/health": return conn.respond(HTTPStatus.OK, "OK\n")
    if req.path != "/ws":     return conn.respond(HTTPStatus.NOT_FOUND, "not found\n")

async def handler(ws):
    loop, q = asyncio.get_running_loop(), asyncio.Queue()
    sid = json.loads(await ws.recv())["session_id"]      # session_start
    async def reader():
        held, got, ended = None, 0, False
        async for msg in ws:                             # keeps draining until the client closes
            if isinstance(msg, bytes):
                if ended: continue                       # post-input_end silence: read, never echo
                if held: q.put_nowait(held)
                held, got = (loop.time() + DELAY, msg), got + len(msg)
            elif (m := json.loads(msg)).get("type") == "input_end" and not ended:
                ended, valid = True, m["valid_samples"] * 2
                if held and (keep := len(held[1]) - (got - valid)) > 0:
                    q.put_nowait((held[0], held[1][:keep]))   # drop the zero-padding
                q.put_nowait(None)
    async def sender():
        while (item := await q.get()) is not None:
            await asyncio.sleep(max(0.0, item[0] - loop.time()))   # absolute deadline, no drift
            await ws.send(item[1])                                   # ≤ 1 s per frame
        await ws.send(json.dumps({"type": "session_complete", "session_id": sid}))
    rd = asyncio.create_task(reader()); await sender(); await rd

async def main():
    server = await serve(handler, "0.0.0.0", 8000, process_request=http, compression=None)
    await server.serve_forever()
asyncio.run(main())
```

The test client reads a 24 kHz WAV, zero-pads it into 3,840-byte frames, sends `session_start`, frame *k* at t₀ + 0.08k on absolute deadlines, then `input_end` with `valid_samples`, then silent frames until `session_complete`. A concurrent receive task stamps each arrival and rebuilds the playback timeline (start = max(arrival, previous end)), then asserts the protocol: exactly one text frame, nothing after it, every server frame ≤ 48,000 bytes, completion within 30 s, and measured echo delay ≈ DELAY. The Dockerfile is `FROM python:3.12-slim`, `pip install websockets==17.2`, `EXPOSE 8000`, and a HEALTHCHECK using Python's `urllib` because the slim image has no curl. What makes it fall behind is mostly self-inflicted: blocking CPU work (a model step, `time.sleep`) on the event-loop thread, which stops reading and fails the session under the 1 s rule; reading and sending in one coroutine, so send backpressure stalls reads; cumulative `sleep(0.08)` drift; unbounded queues that hide growing lag; permessage-deflate; GC pauses; loading weights on first connection (the 10 s accept limit); and torch thread oversubscription. The preventions are respectively `asyncio.to_thread` or a separate inference process with a step under 80 ms, separate tasks, absolute deadlines, queue-depth and lateness logging, `compression=None`, pre-allocated buffers and `gc.freeze()` after warm-up, loading at start-up before `/health` returns 200, and `torch.set_num_threads`. Asyncio already sets `TCP_NODELAY` on TCP transports, so Nagle's algorithm is not the culprit ([CPython base_events.py](https://github.com/python/cpython/blob/3.13/Lib/asyncio/base_events.py)) [V]. Add a `--stress` flag that injects a blocking 0.2 s sleep per frame, to *demonstrate* the failure in the interview.

### E-SP (compact). A causal front-end, and the case where denoising hurts

Process audio in a loop over 80 ms frames (1,920 samples at 24 kHz). Resample causally with `soxr.ResampleStream(24000, 16000, 1, dtype='float32')`, whose per-chunk output length legitimately varies ([python-soxr](https://github.com/dofuuz/python-soxr)) [V]. Denoise with a classical STFT Wiener filter (20 ms sqrt-Hann window, 10 ms hop, eight hops per frame with a 10 ms carry-over; noise PSD from the first 0.3 s or VAD-silent frames with λ ≈ 0.98; decision-directed a-priori SNR with α = 0.98; gain floor ≈ −20 dB against musical noise). Trim silence with Silero VAD, which takes exactly 512-sample chunks at 16 kHz, so keep a ring buffer (80 ms = 1,280 samples = 2.5 chunks), use `VADIterator` with hangover, and only ever withhold *leading* silence, because causal trimming cannot remove silence not yet heard ([silero-vad 6.2.3](https://pypi.org/project/silero-vad/)) [V]. Compare against DeepFilterNet3 (48 kHz, 20 ms windows, 10 ms hop, 2-frame look-ahead, "overall latency of 40 ms", [arXiv 2305.08227](https://arxiv.org/pdf/2305.08227)) [S: re-check] — but run it in a separate Python 3.11 environment with torch/torchaudio ≤ 2.8 and numpy < 2, because `deepfilternet` 0.5.6 (Aug 2023) has Rust wheels only up to cp311 and imports `torchaudio.backend`, which the 2.11 wheel no longer contains ([deepfilternet 0.5.6](https://pypi.org/project/deepfilternet/0.5.6/)) [V]; RNNoise via `pyrnnoise` is a lighter second neural baseline. Make telephone copies with `ffmpeg -af "highpass=f=300,lowpass=f=3400" -ar 8000 -c:a pcm_mulaw`, then back to 24 kHz, plus GSM and low-bitrate Opus (no AMR-NB encoder in stock Ubuntu ffmpeg [L]), and noisy copies at 20/10/5/0 dB SNR with self-recorded noise. Score with SI-SDR (`torchmetrics`, set `zero_mean=True` as Le Roux et al. recommend, since it defaults to False) **after removing each method's integer lag by cross-correlation within ±100 ms**, because a 10–40 ms causal delay otherwise collapses SI-SDR; for telephone copies use the band-limited clean signal as the reference ([torchmetrics sdr.py](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/audio/sdr.py); [Le Roux et al., arXiv 1811.02508](https://arxiv.org/abs/1811.02508)) [V]. Denoising hurts most predictably on **clean or high-SNR input** (≥ 20–30 dB), where it erodes fricatives and aspiration (स, श, फ़) and lowers SI-SDR below the unprocessed input, and on **μ-law/GSM telephone audio**, where codec noise is speech-correlated; Iwamoto et al. showed that enhancement *artifacts* hurt ASR more than residual noise and that mixing some noisy input back in ("observation adding") helps ([arXiv 2201.06685](https://arxiv.org/abs/2201.06685)) [S: re-check]. Report ΔSI-SDR per condition, and the mitigations: an SNR-gated bypass (denoise only below ~15 dB), output = enhanced + 0.1–0.3 × noisy, and gain floors.

---

## Compressing this into the 4-page submission

The submission is "one PDF, at most 4 pages" (or an LLM-generated artifact), plus a repository with RESULTS.md ([Mini PS §2](file:///home/user/interiit_minips/Inter%20IIT%20Tech%20Meet%2015.0_%20Sarvam.ai%20Mini%20PS.pdf)) [V]. The mini PS does not say whether the handwritten appendix and the source table count toward the four pages, so ask; if no answer comes, put them after page 4 and keep the body self-sufficient. Allocate space by points (A 10, B 40, C 25, D 10, E 15) and by what only you can show:

| Part | Points | Space | Keep | Cut |
|---|---|---|---|---|
| A | 10 | ~0.5 page | The points table compressed to one sentence (36/70 on intent+entity; gates zero 70); the playback rule and the four latency drivers; the three most misread rules (playback timeline, 1 s *reading* rule, post-speaker speech counts); the self-correction answer in four sentences | The full session table and rules table |
| B | 40 | ~1.6 pages | B1–B4 at ~0.33 page each, each ending with *experiment / unanswered question / wrong claim* in one line apiece; the original idea in ~0.25 page; one-line pointers to B5 and B6 | Long evidence lists; keep 2–3 strongest citations per proposal |
| C | 25 | ~1.1 pages | The 200 h split table; one condensed validation-threshold table; the "passes audit, trains worse" list as one sentence each; **the pipeline diagram** (redraw it compactly); the three open questions | The tool-accuracy table (move to the repo README) |
| D | 10 | ~0.35 page | One results table: boundary error (mean/median/≤20/≤50 ms, starts vs ends), checker accuracy/precision/recall per type, top three failure cases; a link to RESULTS.md | Method prose (it lives in the repo) |
| E | 15 | ~0.45 page | The entity-accuracy-vs-AL plot, the EndOffset panel, and three sentences on where to wait | Code |
| Appendix | — | after page 4 | Handwritten wait-3 schedule and AL; the ≥ 10-row source table | — |

---

## Conclusion

The deepest lesson of this problem is that, for a Hibiki-style model, *the dataset is the policy*. "When is it safe to speak?" is not answered at decode time by a clever threshold; it is answered months earlier by where each English word sits relative to the Hindi in the training pairs, and the placement rule is the single design decision that touches intent, entity and latency points at once. That reframes the competition's apparent trade-off: AL arithmetic shows that an oracle which waits only at the verb is *both* lower-lag and anticipation-free compared with wait-3 (2.5625 vs 2.625), so the real enemy is not waiting but *uniform* waiting — and the real cost of selective waiting shows up in EndOffset, not AL, which is why the flush after `input_end` matters as much as the policy.

The second lesson is methodological and will serve the interview well. Several confident claims collapsed on checking — a deprecation banner that was reversed, a Hindi option that raises an exception, a lakh table that is never used, a "no alignment needed" headline that needs alignment, and even an off-by-one in the brief itself — while most headline paper numbers here could only be read through search extracts. The defensible stance is to state each number with its provenance, test the cheap things locally, and spend the first week on the three measurements no source can supply: real usable IndicVoices hours, aligner error on code-mixed Hindi and Bulbul English, and whether synthetic entities transfer to real speech.

---

## Appendix — Sources and what each told us

| # | Source | Tag | What it told us |
|---|---|---|---|
| 1 | [Sarvam PS](file:///home/user/interiit_minips/Sarvam.pdf) | V | Rules, KPIs (12/18/18/16/6), gates G1–G3, §5.1 protocol, §5.2 format, §10 permission for Sarvam API outputs in data shared with Sarvam |
| 2 | [Mini PS](file:///home/user/interiit_minips/Inter%20IIT%20Tech%20Meet%2015.0_%20Sarvam.ai%20Mini%20PS.pdf) | V | Task parts and points; the "k + i" wait-k hint that is off by one against STACL |
| 3 | [STACL, arXiv 1810.08398](https://arxiv.org/abs/1810.08398) | S | Prefix-to-prefix training, g(t) = min{k + t − 1, \|x\|}, AL definition (re-check typography) |
| 4 | [SimulEval latency_scorer.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py) | V | Exact AL/LAAL/DAL/AP/ATD/offset code; reference length by default; AL = 2.625 reproduced |
| 5 | [SimulEval instance.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/instance.py) | V | Same max(prev_end, delay) playback rule as Sarvam; computation-aware vs unaware timing |
| 6 | [FBK LAAL](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/LAAL.md) | V | LAAL uses max(\|hyp\|, \|ref\|) so over-generation cannot lower lag |
| 7 | [kyutai-labs/hibiki](https://github.com/kyutai-labs/hibiki) | V | Contextual alignment rule, silence insertion or alignment-aware TTS, 12.5 Hz, CFG caveat |
| 8 | [Hibiki, arXiv 2502.03382](https://arxiv.org/abs/2502.03382) | S | ≥ 2 s lag floor, spike smoothing, ~0.7 s extra lag vs baselines; headline scores conflict across summaries |
| 9 | [Hibiki-Zero, arXiv 2602.11072](https://arxiv.org/abs/2602.11072) + [repo](https://github.com/kyutai-labs/hibiki-zero) | S/V | Sentence-level alignment + GRPO; still needs silence-inserted alignment; 3B, 8–12 GB VRAM |
| 10 | [moshi README](https://github.com/kyutai-labs/moshi/blob/main/README.md), [loaders.py](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/loaders.py), [lm.py](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/lm.py) | V | Mimi 80 ms at 24 kHz, 160 ms floor, sampling defaults, PAD/EPAD tokens, stale quantisation note |
| 11 | [moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) | V | Stereo + alignments JSON format, LoRA r128 at LR 2e-6, throughput for budgeting |
| 12 | [IndicVoices README](https://github.com/AI4Bharat/IndicVoices) / [paper](https://arxiv.org/html/2403.01926v1) | V/S | 76% extempore, L1 verbatim vs L2 standardised, 12,000 h collected vs 3,200 h transcribed |
| 13 | [IndicVoices-R](https://proceedings.neurips.cc/paper_files/paper/2024/file/7dfcaf4512bbf2a807a783b90afb6c09-Paper-Datasets_and_Benchmarks_Track.pdf) | S | Filtering thresholds (SNR, C50, rate, CER ≤ 5%) and ~74.6 h Hindi enhanced subset |
| 14 | [IndicSUPERB / Kathbath](https://github.com/AI4Bharat/IndicSUPERB) | V | 150.2 h read Hindi, CC0, 121 speakers |
| 15 | [OpenSLR 104 (MUCS)](https://openslr.org/104/) | S | ~89.9 h real Hindi-English code-switching, CC BY-SA 4.0 |
| 16 | [Rasa README](https://github.com/AI4Bharat/Rasa) | V | No Hindi at all — refutes the brief |
| 17 | [fish-speech LICENSE](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) | V | Bans using outputs to train speech-generation models |
| 18 | [IndicTrans2 README](https://github.com/AI4Bharat/IndicTrans2) + [IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit) | V | MIT models incl. 200M distilled; BPCC licences; CPU usage recipe |
| 19 | [Vistaar](https://github.com/AI4Bharat/vistaar) | V | Hindi WERs (Kathbath 10.3, Gramvaani 26.8): direct evidence of a telephone gap; swapped table values |
| 20 | [Sarvam STT page](https://www.sarvam.ai/speech-to-text) / [docs](https://docs.sarvam.ai/api-reference-docs/models/saaras) | S | Saaras v3 verbatim/codemix modes; ~19% WER is a 10-language aggregate, not Hindi |
| 21 | [torchaudio releases](https://github.com/pytorch/audio/releases) + [2.11.0 wheel](https://pypi.org/project/torchaudio/2.11.0/) | V | forced_align and MMS_FA preserved; io/sox_effects/backend removed; TorchCodec loading |
| 22 | [torchaudio multilingual FA tutorial](https://github.com/pytorch/audio/blob/main/examples/tutorials/forced_alignment_for_multilingual_data_tutorial.py) | V | uroman recipe, `<star>` token, digits deleted; misleading "removed in 2.9" banner |
| 23 | [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) | V | Cross-check aligner; the PyPI name is a different project |
| 24 | [Rousso et al., arXiv 2406.19363](https://arxiv.org/pdf/2406.19363) | S | English boundary accuracy: MFA ≫ WhisperX > MMS; no Hindi numbers anywhere |
| 25 | [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) / [mfa-models](https://github.com/MontrealCorpusTools/mfa-models/blob/main/docs/source/acoustic/index.rst) | V | Neither Qwen3-ForcedAligner nor MFA has a Hindi acoustic model |
| 26 | [nemo-text-processing 1.2.0](https://pypi.org/project/nemo-text-processing/1.2.0/) | L | Hindi number tables (220 spellings for 10–99, fractions, magnitudes) behind the checker |
| 27 | [num2words](https://pypi.org/project/num2words/) / [word2number](https://pypi.org/project/word2number/) | L | No Hindi; "five lakh" → 5 — do not use for the checker |
| 28 | [Chen et al., arXiv 2010.11247](https://arxiv.org/abs/2010.11247) / [LLM-SI-Corpus, arXiv 2404.12299](https://arxiv.org/abs/2404.12299) | S | Monotonic pseudo-references improve the latency–quality trade-off |
| 29 | [NAIST IWSLT 2024](https://aclanthology.org/2024.iwslt-1.30.pdf) | S | Offline references under-score simultaneous output |
| 30 | [TTS–STT Flywheel, arXiv 2605.03073](https://arxiv.org/abs/2605.03073) | S | Synthetic entity data transfers poorly to Hindi (EHR 0.337 vs 0.65 target) |
| 31 | [Peyser et al., arXiv 1907.01372](https://arxiv.org/pdf/1907.01372) / [Rosenberg et al., arXiv 1909.11699](https://arxiv.org/pdf/1909.11699) | S | TTS numerics up to 8× lower WER; speaker diversity beats single-voice quality |
| 32 | [Emilia, arXiv 2407.05361](https://arxiv.org/pdf/2407.05361) | S | DNSMOS > 3.0, LID ≥ 80%, IQR duration filter; kept only ~29% |
| 33 | [Ko et al. 2015](https://www.danielpovey.com/files/2015_interspeech_augmentation.pdf) / [phonesim](https://github.com/deepmark/phonesim) | S/V | Speed perturbation +4.3% relative; real-codec telephone simulation at 24 kHz |
| 34 | [websockets 17.2](https://pypi.org/project/websockets/17.2/) + [FAQ example](https://github.com/python-websockets/websockets/blob/main/example/faq/health_check_server.py) | V | Asyncio server API, health-check hook, default compression and queue limits |
| 35 | [silero-vad](https://pypi.org/project/silero-vad/) / [deepfilternet 0.5.6](https://pypi.org/project/deepfilternet/0.5.6/) | V | 512-sample VAD chunks; DeepFilterNet breaks on torchaudio ≥ 2.9 and Python ≥ 3.12 |
| 36 | [Iwamoto et al., arXiv 2201.06685](https://arxiv.org/abs/2201.06685) | S | Enhancement artifacts hurt ASR more than residual noise; observation adding helps |
