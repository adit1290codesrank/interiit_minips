# Part A fundamentals: simultaneous (streaming) speech-to-speech translation for the Sarvam mini PS

**Verification legend** (needed because arxiv.org, aclanthology.org, openreview.net, kyutai.org, semanticscholar and huggingface.co were blocked by this session's egress proxy; only raw.githubusercontent.com and web search worked):
- **[V-code]**: checked line by line against the primary source code on GitHub (SimulEval, FBK-fairseq, kyutai moshi) or the official README.
- **[V-PS]**: checked against the two local PDFs (`Sarvam.pdf`, mini PS PDF), read with `pdftotext -layout`.
- **[S]**: only a search-engine snippet or summary of the primary paper backs this. Treat as likely but not verified.
- **[M]**: from my memory of the paper. The canonical URL is given, but I could not re-open the paper this session. **Re-check before quoting in the interview.**
- Arithmetic for the worked example was checked with exact fractions in Python, using a re-implementation of SimulEval's `ALScorer.compute` loop.

---

## Q1. What is the system trying to do, and where are points won and lost?

### Takeaway
The system is a real-time Hindi (code-mixed) → English speech-to-speech translator that cannot take speech back. 36 of the 70 final KPI points are intent (18) and entity (18) accuracy, so the main way to lose points is to commit too early to a verb, negation, amount or name. A failed gate zeroes all 70 KPI points, so protocol robustness comes before everything else.

### Cited Findings
- Task: "listens to live Hindi speech and speaks the English translation while the speaker is still talking, with meaning, numbers and names intact." Source is Hindi freely mixed with English, target is English speech. [V-PS] — [Sarvam PS, p.1](file:///home/user/interiit_minips/Sarvam.pdf)
- The four stated difficulties [V-PS] — [Sarvam PS §2](file:///home/user/interiit_minips/Sarvam.pdf):
  - **Word order** (SOV, verb last). In "मैंने कल अपना पुराना बैंक खाता बंद करवा दिया" the verb phrase "बंद करवा दिया" comes last: "until it is heard, the system cannot tell whether the account was opened, closed or only checked."
  - **No take-backs.**
  - **Details**: lakh/crore, account digits, names, dates, negations, self-corrections.
  - **Little data.**
- Final KPIs, all "measured on your system's English speech": Translation quality 12, Intent accuracy 18 ("what is asked, refused, confirmed or corrected, negations, conditions, and who does what to whom"), Entity accuracy 18 ("names, places, organisations, amounts, numbers, dates and products come out exactly right"), Latency 16 ("How far the English speech trails the speaker, and how soon it finishes after the speaker stops"), Audio quality/naturalness 6 (listener-rated). [V-PS] — [Sarvam PS §7, §9.1](file:///home/user/interiit_minips/Sarvam.pdf)
- "KPI components are scored relative to the other teams and the unchanged base checkpoint." [V-PS] — [Sarvam PS §9.1](file:///home/user/interiit_minips/Sarvam.pdf)
- Gates: "If any gate fails, the final-round KPI points (70) are zero." G1 is that the container meets §5. G2 is fine-tuning evidence (weights, logs, training-pair manifests, before/after scores). G3 is that no records or hours were added to the Part A dataset after the mid-term. [V-PS] — [Sarvam PS §8](file:///home/user/interiit_minips/Sarvam.pdf)
- FAQ on corrections: "The references encode the corrected meaning. Speaking the abandoned value as fact is wrong." On numbers: "Equivalent written forms of the same number, such as 25,000 and 25k, both count as correct." On English words inside the Hindi: "They must come out correctly in the English speech." [V-PS] — [Sarvam PS §11 FAQ](file:///home/user/interiit_minips/Sarvam.pdf)
- The base checkpoint is "a streaming Hindi → English speech-to-speech model trained on synthetic data". Fine-tuning is mandatory, and "the fine-tuned checkpoint must do the translation". A draft model "may only propose tokens that the checkpoint verifies, as in speculative decoding". [V-PS] — [Sarvam PS §3 Part B, FAQ](file:///home/user/interiit_minips/Sarvam.pdf)

### Inferences
- **Where points are won.** Waiting on the content words that carry intent and entities: verbs, negation "नहीं", amounts, names. Speaking fluently and early on everything else ("Yesterday I …", "my old bank account").
- **Where points are lost.**
  1. Anticipating the verb or polarity wrongly. Because there are no take-backs, a wrong "opened" can never be fixed.
  2. Speaking the reparandum of a self-correction.
  3. Mangling lakh/crore conversion (5 lakh = 500,000; 5 crore = 50,000,000).
  4. Letting latency compound because English speech runs longer than the Hindi.
  5. Any protocol failure, which zeroes the session or, via G1, all 70 points.
- Scoring is relative to the base checkpoint, so the minimum goal is "do no harm": fine-tuning must not regress latency or audio quality while it improves entities and intent.

### Gaps
- The exact formula the organisers use for the latency KPI is not disclosed (AL? LAAL? ATD? EndOffset? on which timeline unit?). The PS only says it covers "how far … trails" and "how soon it finishes". No source answers this.
- How intent and entity accuracy are computed (LLM judge? ASR transcript + regex?) is not specified.

---

## Q2. Walkthrough of one session, from first frame to session_complete, and what dominates latency

### Takeaway
Frame k arrives at t = k·80 ms. The server must read continuously, run one model step per 80 ms frame faster than real time, and stream English PCM, which the harness plays at max(arrival, previous end). After `input_end` it must flush the remaining English quickly and send exactly one `session_complete` within 30 s. Latency is driven by the policy lag (how many source words or seconds the model waits), compute per frame, and the serialization of English audio that is longer than the Hindi. Everything is measured on the playback timeline, and speech after the speaker stops counts.

### Cited Findings
- **Protocol** [V-PS] — [Sarvam PS §5.1](file:///home/user/interiit_minips/Sarvam.pdf):
  - `session_start` (JSON: `sample_rate` 24000, `encoding` pcm_s16le, `source_language` "hi", `target_language` "en").
  - Audio is "raw PCM with no WAV header, 80 ms per frame (3,840 bytes), sent in real time. The first frame goes out at t = 0, right after session_start; frame k goes out at t = k × 80 ms. The last frame is zero-padded."
  - `input_end` carries `valid_samples`, "the number of real Hindi samples sent; everything after them is zero-padding".
  - "After input_end, silent 80 ms frames continue in real time until session_complete arrives."
- **Server → client** [V-PS] — [Sarvam PS §5.1](file:///home/user/interiit_minips/Sarvam.pdf):
  - English PCM (mono, 24 kHz, s16le), "at most 1 second per frame".
  - One `session_complete`: "The only server-to-client text frame is one session_complete, and nothing may follow it."
- **Playback** [V-PS] — [Sarvam PS §5.1](file:///home/user/interiit_minips/Sarvam.pdf):
  - "Each frame starts when it arrives or when the previous one finishes, whichever is later. Gaps are heard as silence. All timing is measured on this playback timeline."
  - "After input_end, finish translating, then send session_complete within 30 seconds of input_end. Speech after the speaker stops counts toward latency."
  - "If a frame cannot be delivered within 1 second of its scheduled time because your server is not reading, the session fails."
  - Connection must be accepted within 10 s. Sessions run one at a time with a 2 s gap. Sessions are 10 to 60 s long.
- **Frame arithmetic**: 24,000 samples/s × 0.080 s = 1,920 samples × 2 bytes (s16) = 3,840 bytes. This matches the PS. [V-PS + arithmetic]
- **Mimi codec**: "processes 24 kHz audio, down to a 12.5 Hz representation with a bandwidth of 1.1 kbps, in a fully streaming manner (latency of 80ms, the frame size)". Moshi's "theoretical latency of 160ms (80ms for the frame size of Mimi + 80ms of acoustic delay), with a practical overall latency as low as 200ms on an L4 GPU." [V-code] — [kyutai-labs/moshi README](https://github.com/kyutai-labs/moshi/blob/main/README.md)
- The Moshi loader hard-codes `SAMPLE_RATE = 24000` and `FRAME_RATE = 12.5`. The default LM config has `n_q: 16`, `dep_q: 8`, `text_card: 32000`, and `delays: [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]`. That is a text stream plus 8 output codebooks plus 8 input codebooks, with the semantic codebook at delay 0 and acoustic codebooks delayed by 1 frame. [V-code] — [moshi/models/loaders.py](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/loaders.py)
- **SimulEval uses the same playback rule as Sarvam.** In `SpeechOutputInstance.summarize`, `start = max(prev_end, delay)`. If `start > prev_end` it inserts silence and records a discontinuity, then sets `prev_end = start + duration`. [V-code] — [SimulEval instance.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/instance.py)
- **Computation-aware timing**: SimulEval's "elapsed" time for speech input is `len_sample_to_ms(step) + (current_time - start_time) * 1000`, i.e. audio consumed plus wall-clock compute. Non-computation-aware "delays" count only the audio consumed. [V-code] — [SimulEval instance.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/instance.py); [latency_scorer.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)
- Whisper-Streaming's `--comp_unaware` mode is described as follows: "the timer that counts the emission times 'stops' when the model is computing … The latency is caused only by the model being unable to confirm the output … We implement this feature for finding the lower bound for latency." [V-code] — [ufal/whisper_streaming README](https://github.com/ufal/whisper_streaming/blob/main/README.md)

**Session walkthrough (step by step).**
1. **Connect.** The harness opens the WebSocket and the server accepts within 10 s. The model is already loaded; `/health` must be up within 300 s of start-up.
2. **Start.** `session_start` arrives. The server resets model state (KV cache, codec state), allocates per-session buffers and starts two concurrent tasks: a reader that drains the socket into a queue, and an inference/sender loop.
3. **First frames (t = 0, 80, 160 … ms).** Each 3,840-byte frame is converted to int16, then float, then possibly resampled or normalised, and encoded causally (for a Mimi-style codec, exactly one codec frame per 80 ms). One decoder step runs per frame. During the initial lag (the "wait") the model emits silence or pad tokens. The server should not send silent PCM it does not need to: gaps are heard as silence anyway, and only arrival time matters.
4. **Steady state.** For every input frame the model emits zero or one 80 ms output frame. The server sends output PCM as soon as it is decoded; bundling up to 1 s per frame is allowed, but bundling adds latency. Playback start for each output chunk = max(arrival, previous chunk end).
5. **Speaker stops.** The last real sample is at `valid_samples / 24000` s. The zero-padded tail and the later silent frames must not be treated as speech, but the model must keep stepping on them to flush. In Hibiki/Moshi-style models the output stream only advances when input frames are fed, so silence frames act as the flush clock.
6. **input_end.** The model keeps generating on the silent frames (or on internally fed silence, faster than real time if compute allows) until its text and audio streams signal end. The server sends the remaining audio, then exactly one `session_complete` (within 30 s), and nothing after it.

### Inferences
**What dominates latency, in rough order:**
1. **Policy lag.** How many seconds of Hindi the model waits before each English word. With SOV→SVO this is dominated by verb-final constructions. For Hibiki-like models it is learned from the alignment of the training targets, so it can be changed through fine-tuning data.
2. **Output-duration compounding.** Playback is serialized. If English speech for a span is longer than the time until the next span is ready, every later word starts at the previous end, not at its emission time. Lag then accumulates and shows up in the end offset. Worked hypothetical in Q5: 350 ms English words against 300 ms Hindi words raise time-AL from 787.5 ms to 937.5 ms. Bulbul pace and concise targets are therefore latency levers.
3. **End offset.** After the speaker stops, the time to finish speaking counts. Verbs at the end of Hindi clauses mean the final English content ("closed") can only come after the final Hindi word.
4. **Compute per frame.** The real-time factor must stay below 1 on average. Otherwise the backlog grows without bound and, if inference blocks the socket reader, the 1 s "not reading" rule fails the session. Codec encode/decode, the depth transformer (per-codebook steps) and the network all sit on the 80 ms budget.
5. **Architectural minimum.** One frame (80 ms) plus acoustic delay (80 ms in Moshi) is a floor of about 160 ms. This is negligible next to linguistic lag of 2 to 5 s.
6. **Buffering choices.** Sending in 1 s bundles instead of 80 ms frames adds up to about 1 s of avoidable lag.

- **Decoupling.** The reader must run in its own task so that a slow model step never stops socket reads. Even so, a persistently slow model makes latency grow; it does not fail the session, but the KPI suffers.

### Gaps
- Whether the base checkpoint uses a Mimi-like 12.5 Hz codec is not confirmed. The match between the PS's 80 ms / 24 kHz framing and Mimi's 80 ms / 24 kHz frame is strong circumstantial evidence only.
- How the harness timestamps "arrival" (socket receive time on the client) is not specified.

---

## Q3. Which rules are most important or most easily misunderstood?

### Takeaway
The highest-stakes rules are the ones that zero scores: protocol (G1), fine-tuning evidence (G2) and dataset immutability (G3). The most easily misunderstood ones are about timing: playback-timeline measurement, post-speaker speech counting toward latency, the 1 s reading rule being about reading rather than output, silence frames continuing after input_end, and `valid_samples`.

### Cited Findings
- All rules quoted in Q2. [V-PS] — [Sarvam PS §5.1](file:///home/user/interiit_minips/Sarvam.pdf)
- "No take-backs. Audio, once sent, is final." [V-PS] — [Sarvam PS §5.1](file:///home/user/interiit_minips/Sarvam.pdf)
- "Only your Part A dataset may be used for training." "All English target audio in supervised training pairs must be synthesised with Bulbul: bulbul:v3, voice shubh, language_code="en-IN", 24 kHz. Pace and temperature are free, but declare them." "Training pairs need word-level timings for the English audio. Bulbul does not return them, so align the English audio yourself … they must be ordered, non-negative and inside the English target audio." [V-PS] — [Sarvam PS §3 Part B](file:///home/user/interiit_minips/Sarvam.pdf)
- External models are allowed during training "to align the English audio, as reward models or as teachers". [V-PS] — [Sarvam PS FAQ](file:///home/user/interiit_minips/Sarvam.pdf)

### Inferences
**Easily misunderstood rules:**
1. **"Latency is measured when I send."** Wrong. It is measured on the playback timeline: start = max(arrival, previous end). Sending early cannot make speech start before the previous chunk ends. A burst of audio queues up; it does not play faster.
2. **"After input_end I can take my time."** Speech after the speaker stops counts toward latency, and the 30 s limit is a hard cap, not a target.
3. **"The 1 s rule is about producing output on time."** It is about the server not reading frames. Output gaps are allowed and are "heard as silence".
4. **"Stop reading after input_end."** Silent frames keep coming until `session_complete`, and they must still be read.
5. **"The last frame is all speech."** It is zero-padded. Use `valid_samples` for the true end of speech, e.g. for an end-of-speech flush.
6. **"Repairs can be fixed later."** No take-backs, and the reference encodes the corrected value, so saying the abandoned value as fact is an error even if it is followed by a correction.
7. **"Only one text frame" is easy to break**, e.g. by sending debug JSON or an error message.
8. **Training-data rules.** English target audio must be Bulbul shubh, and other TTS voices are not allowed in supervised pairs. The training data must come from the Part A dataset. English word timings must come from aligning the Bulbul audio. These are what G2 and G3 audit.
9. **Number equivalence.** 25,000 and 25k both count, so "five crore" and "50 million" should both be acceptable. But "five lakh" spoken for a corrected "five crore" is wrong.
10. **English words inside the Hindi** ("bank", "account", product names) must be reproduced correctly. They are not "free".

### Gaps
- Whether audible English disfluencies ("five lakh, sorry, five crore") count as "speaking the abandoned value as fact" is not stated. The FAQ wording suggests an explicitly corrected mention may be less bad, but this is unverified.

---

## Q4. Wait-k and prefix-to-prefix (STACL, Ma et al., 2019): exact definitions

### Takeaway
STACL trains the decoder to predict each target word from only a source prefix. Under wait-k, target word t is generated after reading g(t) = min{k + t − 1, |x|} source words: wait k words, then alternate one READ and one WRITE. The model learns to anticipate (implicit anticipation) because training forces it to predict verbs it has not yet seen.

### Cited Findings
- **Formulas (STACL)** [M] — [STACL, arXiv:1810.08398](https://arxiv.org/abs/1810.08398):
  - Prefix-to-prefix factorisation: p_g(y | x) = ∏_{t=1}^{|y|} p(y_t | x_{≤g(t)}, y_{<t}), where g(t) is the number of source words read when writing y_t (a monotonic non-decreasing function).
  - Wait-k policy: **g_wait-k(t) = min{k + t − 1, |x|}**.
  - Training uses the same prefix constraint, i.e. the source attention is masked to x_{≤g(t)}. This differs from test-time wait-k on a full-sentence model.
- A search snippet confirms the wait-k description: wait-k "commences by processing k source tokens and subsequently alternating between WRITE and READ action". The same summary gives AL = (1/τ) Σ_{i=1}^{τ} g_i − (i−1)/(|y|/|x|), with the "wait-0 … AL of 0" and "wait-1 … one word behind" figure explanation. [S] — [STACL arXiv PDF via search](https://arxiv.org/pdf/1810.08398); [ar5iv mirror](https://ar5iv.labs.arxiv.org/html/1810.08398)
- **Implicit anticipation**: STACL's motivating Chinese→English example has the verb at the end, and the wait-k model learns to emit the English verb before seeing it, from training statistics of the prefix. [M] — [STACL, arXiv:1810.08398](https://arxiv.org/abs/1810.08398)
- **Catch-up**: when |y| > |x|, plain wait-k falls further behind over time. STACL proposes a "catchup" variant that occasionally writes two words per read. [M] — [STACL](https://arxiv.org/abs/1810.08398)
- The mini PS gives its own operational version: "Under wait-k, English word i is spoken once k + i Hindi words have been heard". [V-PS] — [mini PS Part E](file:///home/user/interiit_minips/Inter%20IIT%20Tech%20Meet%2015.0_%20Sarvam.ai%20Mini%20PS.pdf). This equals STACL's k + t − 1 only if i is 0-indexed. See Q10.

### Inferences
- In speech, the "word" unit of wait-k becomes either ASR-detected words (cascade or StreamSpeech-style CTC counts) or a fixed time chunk ("wait k seconds"). Hibiki-like models have no explicit k. Their lag is whatever their training alignment taught them; Hibiki's ablation tried a constant lag as a baseline (see Q9, [S]).
- For Hindi→English, wait-k with small k forces anticipation of the verb, its polarity (negation is often verb-adjacent: "नहीं किया") and the causative ("करवा" → "got … closed").

### Gaps
- I could not re-open the STACL PDF to quote equation numbers. Re-check the exact typesetting (curly braces, τ subscript) in arXiv:1810.08398 §2 and §4 before writing it by hand.

---

## Q5. Latency metrics: AL, LAAL, DAL, AP, StartOffset/EndOffset, ATD, computation-aware lag, and S2ST latency measurement

### Takeaway
AL averages, over target words up to the cut-off τ (the first word written after the whole source is read), how far each word lags behind an ideal translator that reads at rate |x|/|y|. SimulEval implements it in source tokens for text and in milliseconds for speech. By default it uses the reference length, and LAAL replaces this with max(|hyp|, |ref|) so that over-generation cannot lower lag. For speech output, SimulEval computes these metrics on the playback timeline from MFA word alignments and adds StartOffset, EndOffset and ATD.

### Cited Findings
**AL as defined in STACL** [M]; the code [V-code] matches.
- AL_g(x, y) = (1/τ_g(|x|)) Σ_{t=1}^{τ_g(|x|)} [ g(t) − (t − 1)/r ], with r = |y|/|x| and τ_g(|x|) = argmin_t { g(t) = |x| }, the first target index at which the full source has been read. — [STACL](https://arxiv.org/abs/1810.08398)
- For wait-k with |x| = |y|, AL = k.

**AL in SimulEval** [V-code] — [SimulEval latency_scorer.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)
- Docstring: `AL = 1/τ Σ_i^τ D_i − (i−1)|X|/|Y|`, `τ = argmin_i(D_i = |X|)`, and "When reference was given, |Y| would be the reference length".
- The code:
  - starts with `if delays[0] > source_length: return delays[0]`;
  - sets `gamma = target_length / source_length`;
  - loops `AL += d - t_minus_1 / gamma; tau = t_minus_1+1; if d >= source_length: break`;
  - returns `AL /= tau`.
- `use_ref_len` defaults to True, so `tgt_len = ins.reference_length` (in words, chars or SPM units, depending on `latency_unit`).

**Speech units in SimulEval** [V-code] — [instance.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/instance.py)
- For speech input, `source_length` is `len_sample_to_ms(len(samples))`, the source duration in ms. Delays are the ms of source consumed when each output unit was emitted.
- The time-based AL is therefore AL_ms = (1/τ) Σ_{i=1}^{τ} [ d_i − (i−1)·|X|_ms/|Y| ], with |Y| the reference word count.

**Speech output in SimulEval** [V-code] — [latency_scorer.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)
- The `*_SpeechAlign_{BOW,COW,EOW}` scorers exist for AL, LAAL, AP, DAL, ATD, StartOffset and EndOffset.
- They run the Montreal Forced Aligner on the output wav (which already has playback-rule silences inserted). Each output word's delay is `target_offset + 1000*interval.minTime` (BOW), `maxTime` (EOW) or the midpoint (COW), where `target_offset = delays[0]`.
- This is the closest open-source analogue of "how far the English speech trails the speaker" on a playback timeline.

**LAAL (Papi et al., 2022)** [V-code]
- SimulEval's `LAALScorer`: `gamma = max(len(delays), target_length) / source_length`, docstring `LAAL = 1/τ Σ D_i − (i−1)|X|/max(|Y|,|Y*|)`. LAAL is "robust to the length difference between the hypothesis and reference". — [SimulEval latency_scorer.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)
- It was first used in CUNI-KIT IWSLT 2022 (arXiv:2204.06028) and named in "Over-Generation Cannot Be Rewarded: Length-Adaptive Average Lagging for Simultaneous Speech Translation" (AutoSimTrans 2022). — [same file](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py); [FBK LAAL.md](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/LAAL.md); [arXiv:2206.05807](https://arxiv.org/abs/2206.05807)
- **Why it matters.** With the reference length, a hypothesis longer than the reference is compared against an ideal policy that is too slow, so its lag looks smaller than it is. With max(|Y|, |Y*|), over-generation can no longer lower the score. Worked number in Q6: a 10-word hypothesis with wait-3 delays gives AL (ref = 8) = 2.625 but LAAL = 3.3.
- FBK's script computes the oracle delays as `arange(max_tgt_len) * src_lens / tgt_lens`, masks lagging after the first delay ≥ src_len ("tau is the cut-off step"), and reports a computation-aware LAAL (CA) as well. [V-code] — [FBK LAAL_metric.py](https://github.com/hlt-mt/FBK-fairseq/blob/master/examples/speech_to_text/scripts/LAAL_metric.py)

**DAL (Arivazhagan et al., 2019, MILk)** [V-code] — [SimulEval DALScorer](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)
- g′_1 = g_1, g′_i = max(g_i, g′_{i−1} + 1/γ), DAL = (1/|y|) Σ_i [g′_i − (i−1)/γ], with γ = |y|/|x|.
- There is no τ cut-off: every target word must lag at least 1/γ behind the previous one, so a burst of tail tokens written after the source ends still costs lag.
- In SimulEval's DAL, `target_length = len(delays)`, the hypothesis length, not the reference.
- Paper: [MILk, arXiv:1906.05218](https://arxiv.org/abs/1906.05218)

**AP (Cho and Esipova, 2016)** [V-code] — [SimulEval APScorer](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)
- AP = (1/(|X||Y|)) Σ_i D_i (arXiv:1606.02012).
- STACL criticised AP as sensitive to length. [S] — [STACL search snippet](https://arxiv.org/pdf/1810.08398)

**Offset metrics** [V-code] — [SimulEval](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)
- StartOffset = `delays[0]`.
- EndOffset = (start + duration of the last output interval) − source_length, i.e. the end of output speech after the end of the source, on the playback timeline.
- RTF = end of output / source length.
- The Discontinuity{Sum,Ave,Num} scorers measure the inserted playback silences.

**ATD (Kano et al., arXiv:2211.13173)** [V-code] — [SimulEval ATDScorer](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)
- ATD measures, per target token, the end time of the target token minus the end time of its paired source token. Pairing is by chunk accounting. Speech tokens are virtually 300 ms ("SRC_TOKEN_LEN = 300 # 300ms per word").
- Target end time = max(delay, previous target end) + token length (+ compute). Unlike AL, ATD therefore accounts for output duration and serialization.

**Hibiki's metrics** [S] — [Hibiki, arXiv:2502.03382 / OpenReview PDF via search](https://openreview.net/pdf?id=fgjN8B6xVX)
- ASR-BLEU (Whisper transcription of the output speech, then BLEU).
- LAAL in seconds, computed from word-level emission timestamps of the output.
- "End Offset": time between the end of the last source word and the end of the last output word.
- Speaker similarity and naturalness.

**Seamless** [S] — [Seamless, arXiv:2312.05187 via search](https://arxiv.org/pdf/2312.05187)
- Adopts EMMA as its simultaneous policy, with each attention head running its own policy.
- From memory [M]: SeamlessStreaming reports AL/LAAL for text output and "ending offset" for speech output, evaluated with SimulEval.

**Other S2ST evaluation**: NaturalFlow (arXiv:2606.13121, 2026) uses End Offset and ASR-BLEU (Whisper-medium + SacreBLEU) following the IWSLT protocol. [S] — [NaturalFlow](https://arxiv.org/pdf/2606.13121)

**Computation-aware (CA) vs non-computation-aware (NCA)**: CA uses `elapsed` (audio consumed + wall-clock), NCA uses `delays` (audio consumed only). CA is not supported for text input, and the speech-alignment scorers raise an error for CA. [V-code] — [SimulEval latency_scorer.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)

### Inferences
- Sarvam's harness is effectively a computation-aware, playback-timeline evaluation, because real wall-clock frames and the max(arrival, previous end) rule are used. The closest proxies to use internally are SimulEval's `LAAL_SpeechAlign_BOW` (or ATD) plus `EndOffset`, computed on the reconstructed playback wav.
- AL's τ cut-off ignores everything written after the source ends. A model that waits and then dumps the whole tail at the end can look good on AL but bad on EndOffset and DAL. The PS's latency KPI explicitly includes "how soon it finishes", so EndOffset-like behaviour is scored.

### Gaps
- Exact Hibiki and Seamless metric definitions, e.g. whether Hibiki's LAAL uses BOW or EOW timestamps, could not be checked in the papers this session.

---

## Q6. Worked example: wait-3 schedule and Average Lagging for "मैंने कल अपना पुराना बैंक खाता बंद करवा दिया" → "Yesterday I got my old bank account closed"

### Takeaway
|x| = 9 and |y| = 8, so g(t) = min(t + 2, 9) = 3, 4, 5, 6, 7, 8, 9, 9. τ = 7 and r = 8/9, giving **AL = 147/56 = 21/8 = 2.625 source words**. LAAL is also 2.625 (hypothesis = reference length), DAL = 3, AP = 17/24 ≈ 0.708. "Yesterday", "I", "got" and "my" are all spoken before "बंद" (x7) is heard, so "got" (the causative, really carried by "करवा दिया", x8–x9) is an anticipation.

### Cited Findings
- The formulas are as in Q4 and Q5: g_wait-k(t) = min{k + t − 1, |x|}; AL as in STACL / SimulEval `ALScorer`. [V-code] — [SimulEval latency_scorer.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py); [M] — [STACL](https://arxiv.org/abs/1810.08398)
- The sentence and its English come from the PS. [V-PS] — [Sarvam PS §2](file:///home/user/interiit_minips/Sarvam.pdf)

**Tokens.** Source: x1 मैंने, x2 कल, x3 अपना, x4 पुराना, x5 बैंक, x6 खाता, x7 बंद, x8 करवा, x9 दिया (|x| = 9). Target: y1 Yesterday, y2 I, y3 got, y4 my, y5 old, y6 bank, y7 account, y8 closed (|y| = 8).

**Wait-3 schedule.** g(t) = min(3 + t − 1, 9). Oracle term = (t−1)/r = (t−1)·9/8.

| t | English y_t | g(t) | Hindi heard so far (last word) | aligned Hindi source of y_t | (t−1)·9/8 | g(t) − (t−1)·9/8 |
|---|---|---|---|---|---|---|
| 1 | Yesterday | 3 | … अपना | कल (x2), already heard | 0 | 3.000 |
| 2 | I | 4 | … पुराना | मैंने (x1), heard | 1.125 | 2.875 |
| 3 | got | 5 | … बैंक | करवा दिया (x8–9), **not heard: anticipation** | 2.250 | 2.750 |
| 4 | my | 6 | … खाता | अपना (x3), heard | 3.375 | 2.625 |
| 5 | old | 7 | … बंद | पुराना (x4), heard | 4.500 | 2.500 |
| 6 | bank | 8 | … करवा | बैंक (x5), heard | 5.625 | 2.375 |
| 7 | account | 9 | … दिया (all read) → **τ = 7** | खाता (x6) | 6.750 | 2.250 |
| 8 | closed | 9 | (all read, after τ, not counted) | बंद (x7) | — | — |

**AL step by step.**
- Σ_{t=1}^{7} g(t) = 3+4+5+6+7+8+9 = 42.
- Σ (t−1)·9/8 = (9/8)·(0+1+…+6) = (9/8)·21 = 23.625.
- AL = (42 − 23.625)/7 = 18.375/7 = **2.625**.
- Cross-check: the per-row terms 3 + 2.875 + 2.75 + 2.625 + 2.5 + 2.375 + 2.25 sum to 18.375. ✓.
- Verified with a script re-implementing SimulEval's `ALScorer.compute` with exact fractions: AL = 21/8, τ = 7.

**Why AL < k = 3.** Because |y| < |x| (r = 8/9), the ideal translator is allowed 9/8 source words per target word. Wait-3's fixed one-in/one-out rhythm therefore gains 1/8 of a word on the ideal each step: lag goes 3 → 2.25 by τ. AL = k holds only when |x| = |y|.

**Other metrics on the same schedule** (all computed by script):
- **LAAL**: max(|hyp|, |ref|) = 8, so LAAL = 2.625, the same as AL.
- **DAL**: because DAL forces a spacing of at least 1/γ = 9/8 source words between consecutive target words, g′_t = max(g_t, g′_{t−1} + 9/8) = 3 + (t−1)·9/8. That gives 3, 4.125, 5.25, 6.375, 7.5, 8.625, 9.75, 10.875, so every term g′_t − (t−1)·9/8 = 3 and DAL = 3.0. DAL does not "credit" wait-k for |y| < |x| the way AL does.
- **AP** = 51/(9·8) = 17/24 ≈ 0.708.
- **Over-generation illustration**: a 10-word hypothesis with wait-3 delays gives AL (reference length 8) = 2.625, but LAAL = 3.3, because the ideal rate becomes 10/9.

**AL for other k** (same sentence, |y| = 8; script):

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 (offline) |
|---|---|---|---|---|---|---|---|---|---|
| AL | 0.5625 | 1.5625 | 2.625 | 3.6875 | 4.75 | 5.8125 | 6.875 | 7.9375 | 9.0 |

To avoid anticipating the verb under fixed wait-k, "got" (t = 3) would need g(3) ≥ 8 to 9, i.e. k ≥ 6 or 7 (AL ≥ 5.8). Fixed policies pay that lag on every word, not just the verb.

**Reordered English "I got my old bank account closed yesterday".**
- The wait-3 g(t) depends only on t and |x|, and |y| is still 8. Token-level **AL is identical: 2.625**. AL measures *when* words are emitted, not *which* source word they translate.
- What changes is the risk profile and the real per-word lag:
  - "got" is now y2, emitted after x4, which is even earlier.
  - "closed" is y7, emitted at g = 9, which is safe.
  - "yesterday" is y8, emitted after the whole source, about 7 words after "कल" was heard, which is a large real delay for that word.
- Alignment-aware metrics (ATD, or SimulEval speech AL with word alignments) and the semantic-risk analysis distinguish the two orders; plain AL does not.

**"Wait only when needed" oracle (illustration).**
- Emit each English word only once its aligned Hindi is heard, keeping English order monotone: Yesterday@2, I@2, got@8 (करवा), then my/old/bank/account/closed@8. This gives AL = 41/16 = 2.5625, *lower* than wait-3, with zero anticipation. The reason is that the long early wait is offset by the fast tail and the τ cut-off.
- If "got … closed" needs "दिया" (x9) to fix the tense, then everything from "got" on waits until 9. τ becomes 3 and AL = 77/24 ≈ 3.21.
- This shows AL rewards bursty tails. In speech, the tail of 6 words still has to be *spoken* after the source ends, so EndOffset grows. That is exactly the trade-off the Sarvam latency KPI's "how soon it finishes" clause captures.

**Time-based (speech) version, hypothetical illustration** (the uniform word durations are assumptions, not data):
- Assume each Hindi word lasts 300 ms (|X| = 2,700 ms) and English words start the moment wait-3 allows. Then d = 300·g(t) ms and AL_ms = 300 × 2.625 = **787.5 ms**.
- If each English word also lasts 350 ms and playback is serialized (start = max(emit, previous end), the Sarvam/SimulEval rule), the start times become 900, 1250, 1600, 1950, 2300, 2650, 3000, 3350 ms.
- Time-AL (BOW) then rises to **937.5 ms** and EndOffset = 3,700 − 2,700 = **1,000 ms**. Script-verified.

### Inferences
- The handwritten appendix answer:
  - **Schedule**: Yesterday | I | got | my | old | bank | account | closed after 3, 4, 5, 6, 7, 8, 9, 9 Hindi words.
  - **τ = 7, r = 8/9, AL = 2.625 words.**
  - **Risk**: "got" is spoken before "बंद करवा दिया" is heard. "Yesterday I … my old bank account" are safe because they are aligned to already-heard words. "closed" is safe because it is emitted after बंद.
- A better English target for streaming is one whose verb comes late, so less anticipation is needed. For example "Yesterday, my old bank account, I got it closed" or "Yesterday my old bank account got closed" (passive). This is a Part B "training targets" lever. Fluency and the "literal vs fluent" trade-off need judging.

### Gaps
- Whether the judges expect |y| = hypothesis length (STACL's original) or reference length (SimulEval default) does not matter here, since both are 8. Mention both.

---

## Q7. Adaptive read/write policies that "wait only when they must"

### Takeaway
Adaptive policies decide READ or WRITE from a model signal that indicates sufficient context. Examples:
- agreement between successive hypotheses (wait-if-diff, LocalAgreement);
- learned monotonic attention stopping probabilities (MILk, MMA, EMMA);
- attention mass on the latest frames (EDAtt, AlignAtt);
- CTC-recognised source progress (StreamSpeech);
- for Hibiki, a lag baked into training-data alignment from an MT model's likelihood gain.

### Cited Findings
- **wait-if-diff / wait-if-worse** (Cho and Esipova, 2016, arXiv:1606.02012) [M]. Run a full-sentence NMT on the current prefix. *Wait-if-worse* reads more if the probability of the candidate next word decreases when one more source word is added. *Wait-if-diff* reads more if the predicted next word changes when one more source word is added. The MMA paper cites MILk as better than "fixed policy approaches, such as wait-k … or wait-if-* (Cho & Esipova, 2016)". [S] — [MMA, OpenReview](https://openreview.net/pdf?id=Hyg96gBKPS)
- **MILk** (Arivazhagan et al., ACL 2019): "a hard, monotonic attention head to schedule the reading of the source sentence, and a soft attention head that extends from the monotonic head back to the beginning of the source". Its adaptive schedule beats wait-k "for many latency values". It is trained with a latency-augmented loss using DAL. [S] — [MILk, ACL P19-1126](https://aclanthology.org/P19-1126/); [arXiv:1906.05218](https://arxiv.org/abs/1906.05218)
- **MMA** (Ma et al., ICLR 2020): monotonic attention extended to every multihead head. MMA-H (hard, limited span, for streaming) and MMA-IL (infinite lookback, for quality). [S] — [MMA](https://openreview.net/pdf?id=Hyg96gBKPS)
- **EMMA** (Ma et al., 2023, arXiv:2312.04515): Efficient Monotonic Multihead Attention, applied in its infinite-lookback variant. SeamlessStreaming adopts EMMA as its policy and "each attention head operates an individual simultaneous policy". [S] — [EMMA](https://arxiv.org/pdf/2312.04515); [Seamless](https://arxiv.org/pdf/2312.05187); [seamless_communication README](https://github.com/facebookresearch/seamless_communication/blob/main/docs/streaming/README.md)
- **EDAtt** (Papi et al., ACL 2023, arXiv:2212.07850), [V-code] from the agent code:
  - From an *offline* ST model's cross-attention (`--extract-attn-from-layer`), "for each token, it sums the attention scores of the last *frame_num* frames and verify if it exceeds *attn_threshold*. If this condition is verified, the emission at the current time step is stopped".
  - α ∈ {0.6, 0.4, 0.2, 0.1, 0.05, 0.03}.
  - Intuition: if a token attends to the most recent audio, that audio is still incomplete, so wait.
  - — [FBK EDAtt agent](https://github.com/hlt-mt/FBK-fairseq/blob/master/examples/speech_to_text/simultaneous_translation/agents/v1_0/simul_offline_edatt.py); [EDATT doc](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/EDATT_SIMULST_AGENT_ACL2023.md)
- **AlignAtt** (Papi et al., Interspeech 2023, arXiv:2305.11408), [V-code]:
  - Computes `most_attended_idxs = argmax(attn)` per new token, with `first_invalid_frame = n_frames − frame_num`.
  - Emits tokens up to the first one whose argmax-aligned frame is ≥ first_invalid_frame. Otherwise READ.
  - f ∈ {2, …, 14}.
  - — [FBK AlignAtt agent](https://github.com/hlt-mt/FBK-fairseq/blob/master/examples/speech_to_text/simultaneous_translation/agents/v1_0/simul_offline_alignatt.py); [ALIGNATT doc](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/ALIGNATT_SIMULST_AGENT_INTERSPEECH2023.md)
- **LocalAgreement-n** (Whisper-Streaming, Macháček et al., 2023, arXiv:2307.14743): "if n consecutive updates … agree on a prefix, it's confirmed". Whisper-Streaming uses n = 2 and the reference is "CUNI-KIT at IWSLT 2022". The buffer is trimmed at confirmed sentences or segments, and the system reports "3.3 seconds latency on unsegmented long-form speech". [V-code] — [ufal/whisper_streaming README](https://github.com/ufal/whisper_streaming/blob/main/README.md)
- **StreamSpeech** (Zhang et al., ACL 2024, arXiv:2406.03049): a multi-task model with CTC decoders for ASR and S2TT, whose alignments "guide the policy". It READs until the source text in the source speech is recognised, then WRITEs the corresponding target. Target speech comes from a non-autoregressive text-to-unit module plus a unit HiFi-GAN vocoder. One model covers streaming ASR, Simul-S2TT and Simul-S2ST "under any latency". [S] — [StreamSpeech arXiv HTML](https://arxiv.org/html/2406.03049v1); [V-code] — [ictnlp/StreamSpeech README](https://github.com/ictnlp/StreamSpeech/blob/main/README.md)
- **Hibiki contextual alignment**:
  - Per the README: "Word-level matching is made between source and target transcripts using a *contextual alignment* weakly-supervised method that leverages an off-the-shelf MADLAD machine translation system. The derived alignment rule (a word should only appear in the target once it's predictable from the source) is applied either by inserting silences or by synthesizing targets with a voice controlled, alignment-aware TTS." At inference, "Hibiki relies on simple temperature sampling". [V-code] — [kyutai-labs/hibiki README](https://github.com/kyutai-labs/hibiki/blob/main/README.md)
  - Per search summaries: for each target word, the source word whose appearance most increases the MT log-likelihood of that target word (argmax of the log-likelihood difference between consecutive source prefixes) is taken as its alignment. [S] — [Hibiki OpenReview PDF via search](https://openreview.net/pdf?id=fgjN8B6xVX)
  - Formally, per my reading: a(i) = argmax_j [ log p(y_i | x_{≤j}, y_{<i}) − log p(y_i | x_{≤j−1}, y_{<i}) ]. [M/S]

### Inferences
**How each policy decides it has "enough":**

| Policy | READ/WRITE signal |
|---|---|
| wait-k | a fixed count, so it never adapts |
| wait-if-* | output stability under one more source word |
| LocalAgreement | output stability across consecutive chunks |
| MILk, MMA, EMMA | a learned per-step stopping probability, trained with a latency loss |
| EDAtt, AlignAtt | where the decoder's cross-attention points; attention on the newest audio means wait |
| StreamSpeech | the number of source tokens recognised so far by CTC |
| Hibiki | no explicit policy; the lag is learned from training data aligned by an MT model's "predictability" gain |

- For a Hibiki/Moshi-like base checkpoint, which is plausible here, the *training targets' timing* is the policy. "Teaching the model to wait" means shaping where English words sit in time in the Bulbul target audio.

### Gaps
- Exact Hibiki details could not be verified from the paper text: the minimum-lag margin added after the aligned source word, the silence-insertion procedure, the alignment-aware TTS conditioning, and its CFG/speaker-similarity tricks. Check arXiv:2502.03382 §3.

---

## Q8. Self-corrections ("पाँच लाख… नहीं नहीं, पाँच करोड़।"): when to speak the amount, and how to teach "wait only when it must"

### Takeaway
The utterance has the classic repair structure: **reparandum** "पाँच लाख", interruption point "…", **interregnum / editing term** "नहीं नहीं", **repair** "पाँच करोड़". The amount is safe to speak only once the repair's unit word "करोड़" has been heard and the next cue shows that no further correction follows: the sentence end "।", a pause, or the next content word. Teach this with training targets that drop the reparandum and timings that place the whole entity span after the repair end, plus balanced counter-examples where uncorrected amounts are spoken promptly.

### Cited Findings
- FAQ: the reference holds the corrected meaning, and the abandoned value spoken as fact is wrong. [V-PS] — [Sarvam PS FAQ](file:///home/user/interiit_minips/Sarvam.pdf)
- **Disfluency in end-to-end ST** (Salesky et al.):
  - SLT 2018 released fluent (cleaned) references for Fisher Spanish–English.
  - NAACL 2019 trained end-to-end models to translate "from noisy, disfluent speech to fluent text".
  - The cleaned references were made by sentence-level rewriting rather than token deletion, because "simply removing disfluencies from an utterance can create ill-formed output".
  - [S] — [Salesky et al. 2019, arXiv:1906.00556](https://arxiv.org/abs/1906.00556); [Salesky et al. 2018, arXiv:1811.03189](https://arxiv.org/abs/1811.03189); [ACL N19-1285](https://aclanthology.org/N19-1285/)
- A 2026 paper, "The Role of Disfluencies in Speech Translation" (arXiv:2608.02138), exists, but I could not read its findings. [S] — [arXiv:2608.02138](https://arxiv.org/html/2608.02138)
- **Repair terminology** (reparandum / interruption point / interregnum or editing phase / repair) is from Shriberg's 1994 dissertation, *Preliminaries to a Theory of Speech Disfluencies* (UC Berkeley). The Switchboard disfluency annotation (Meteer et al., 1995) uses the bracket notation `[ RM + RR ]` with `{E …}` editing terms. [M] — not fetched; search returned no primary source.
- Hibiki's alignment rule is "a word should only appear in the target once it's predictable from the source". [V-code] — [hibiki README](https://github.com/kyutai-labs/hibiki/blob/main/README.md)

### Inferences
**When is it safe to speak the amount?**
- "पाँच" alone is *not* safe if the English renders lakh/crore in Western digits: 5 lakh = "500,000" and 5 crore = "50 million" share no spoken word. It is only "shared" if the English keeps Indian units ("five … crore").
- Even then, the number word itself can be corrected ("पाँच… नहीं, छह लाख"). Speaking "five" early saves at most about 300 ms and risks entity accuracy (18 points).
- The earliest safe point is the end of "करोड़" (end of the repair's unit) plus a short look-ahead (about one word or a pause, a few hundred ms) that rules out a second interregnum.
- After "पाँच लाख", the token "नहीं" is ambiguous:
  - Negation of the predicate: "पाँच लाख नहीं दिए" = "did not give five lakh". This is itself an intent-critical item.
  - Self-correction: "नहीं नहीं, पाँच करोड़".
  - The disambiguators come *after* "नहीं": a repeated "नहीं नहीं", a comma or pause, then a same-type entity, means repair; "नहीं" followed by a verb means negation. Prosody (pause or filled pause before "नहीं") marks the interruption point.
  - So even the first "नहीं" should not be translated immediately after an amount.
- Non-entity parts of the sentence (for example a preceding "मुझे … चाहिए" → "I need …") can still be spoken without waiting. Only the entity span is held.

**How to teach "wait only when it must"** (for a Hibiki-style model whose lag is learned from target timings):
1. **Target text**: corrected meaning only ("… five crore"), matching the FAQ. Optionally keep a natural fluent English that does not voice the reparandum.
2. **Entity-atomic timing.**
   - Compute Hibiki-style contextual alignment, but treat each entity (amount + unit, name, phone digit group, date) as one span.
   - Place the *first* English word of the span no earlier than the end time of the *last* Hindi word of the corresponding final (repair) span, plus a margin δ_e (for example 200 to 500 ms, tuned).
   - Pitfall: plain per-word contextual alignment would align "five" to the reparandum "पाँच" (that is where its likelihood jumps) and teach early "five".
   - For non-entity words keep the small default lag, so the model does not learn to wait everywhere.
3. **Negation and verb polarity** get the same treatment: the English verb and its polarity are placed after the Hindi verb group plus any "नहीं".
4. **Data balance.**
   - Inject synthetic self-corrections (reparandum + "नहीं नहीं" / "मतलब" / "sorry" + repair) into Part A scripts and audio.
   - Include confusable negatives: amount + negated verb with no correction, and amounts with no correction at all, where the timing is prompt.
   - The model then learns that the trigger for waiting is the presence of an interregnum cue or an incomplete entity, not entities in general.
   - Measure entity accuracy against lag per category, as in Part E.
5. **Decoding-time guard** (allowed, since the checkpoint still translates): bias pad/silence tokens up while the text stream is inside an unfinished entity span. This is a "hold" mechanism in the inner-monologue text stream.

### Gaps
- No source found that specifically studies *simultaneous* (streaming) translation of self-repairs, or the latency cost of waiting for repairs. This is an open question worth stating in the interview.
- Typical durations of Hindi interregna ("नहीं नहीं") are unknown; δ_e must be tuned empirically.
- Synthetic disfluency-generation methods were not checked (e.g. LARD, Passali et al. 2022 [M]).

---

## Q9. Architectures (Hibiki, Moshi, StreamSpeech, Seamless) and what the Sarvam base checkpoint may look like

### Takeaway
The Sarvam checkpoint is most likely Hibiki/Moshi-style. The evidence: synthetic training data, required English word-level timings in training pairs, 24 kHz audio in 80 ms frames (exactly one Mimi frame), and a fine-tune-only regime. Such a model is a decoder-only multistream transformer over 12.5 Hz codec tokens, with a target-text "inner monologue" stream and a lag learned from time-aligned targets. StreamSpeech and Seamless are the explicit-policy alternatives.

### Cited Findings
- **Hibiki** [V-code] — [hibiki README](https://github.com/kyutai-labs/hibiki/blob/main/README.md):
  - "a decoder-only model for simultaneous speech translation".
  - It "leverages the multistream architecture of Moshi to model source and target speech jointly" and "produces text and audio tokens at a constant framerate of 12.5Hz … along with timestamped text translation".
  - Trained on synthetic aligned data (contextual alignment + silence insertion or alignment-aware TTS).
  - CFG controls voice transfer.
  - French→English only. Hibiki-M runs on-device. "trained on sequences up to 120 seconds and use a context size of 40 seconds".
  - Inference code lives in the moshi repo (`--hf-repo kyutai/hibiki-1b-pytorch-bf16`).
  - Paper: Labiausse et al., ICML 2025, [arXiv:2502.03382](https://arxiv.org/abs/2502.03382). A model card reports about 40k hours of synthetic Fr–En parallel data. [S] — [search summary](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16)
- **Moshi** [V-code] — [moshi README](https://github.com/kyutai-labs/moshi/blob/main/README.md):
  - Two audio streams (model and user) plus "text tokens corresponding to its own speech, its inner monologue".
  - "A small Depth Transformer models inter-codebook dependencies for a given time step, while a large, 7B-parameter Temporal Transformer models the temporal dependencies".
  - The Mimi codec runs at 12.5 Hz and 1.1 kbps, at 24 kHz, with 80 ms latency.
  - Paper: [arXiv:2410.00037](https://arxiv.org/abs/2410.00037).
  - Loader delays: text and semantic codebook at delay 0, acoustic codebooks at delay 1. [V-code] — [loaders.py](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/loaders.py)
- **StreamSpeech** [V-code + S] — [README](https://github.com/ictnlp/StreamSpeech/blob/main/README.md); [arXiv:2406.03049](https://arxiv.org/html/2406.03049v1):
  - An "All in One" model doing streaming ASR, Simul-S2TT and Simul-S2ST with intermediate text outputs.
  - CTC-guided policy, NAR text-to-unit, unit HiFi-GAN vocoder.
  - Checkpoints for Fr/Es/De→En (CVSS).
- **SeamlessStreaming** [V-code] — [docs/streaming/README](https://github.com/facebookresearch/seamless_communication/blob/main/docs/streaming/README.md):
  - 2.5B parameters: a "monotonic decoder checkpoint" plus a "streaming UnitY2 checkpoint". The EMMA policy is evaluated with SimulEval.
  - Speech input in 101 languages, speech output in 36.
  - "Seamless model is simply the SeamlessStreaming model with the non-expressive vocoder_v2 swapped out with the expressive vocoder_pretssel". [V-code] — [seamless_communication README](https://github.com/facebookresearch/seamless_communication/blob/main/README.md)

### Inferences
- **If Hibiki-like**, the fine-tuning levers are:
  1. Target text, including the inner-monologue text stream, which can be supervised directly.
  2. Placement in time of the English Bulbul audio and text tokens relative to the Hindi, which is the learned lag. This is why the PS demands English word timings.
  3. Possibly CFG and sampling temperature at inference.
- **Latency floor**: about 80 to 160 ms architectural. Real latency comes from learned linguistic lag and end-of-utterance flushing.
- **Compute**: a 1 to 2B temporal transformer plus depth transformer at 12.5 Hz is real-time on an H100. The per-session KV cache resets matter for 60 s sessions, given that Hibiki uses a 40 s context.
- **If StreamSpeech or Seamless-like instead**, the policy is explicit (CTC or EMMA thresholds), so decode-time knobs such as chunk size and thresholds trade lag against accuracy without retraining.

### Gaps
- The base checkpoint's architecture is undisclosed until after the mid-term (PS FAQ), so all of the above is inference.
- The size of Hibiki (2B vs 1B for Hibiki-M) and its text tokenizer were not verified this session.

---

## Q10. Claims that turned out wrong or misleading when checked (for the "one claim that was wrong" requirement)

### Takeaway
Four checks turned up claims that were wrong or misleading. Two were in AI search summaries (an inconsistent LAAL description, invented "query" content), one was the common "wait-k gives AL = k", and one was an off-by-one in the mini PS's own wait-k description.

### Cited Findings
- **"Wait-k has AL = k."** This holds only when |x| = |y|. For the PS sentence (|x| = 9, |y| = 8), wait-3 gives AL = 2.625, and wait-k AL is 0.5625, 1.5625, 2.625 … for k = 1, 2, 3 (Q6, script-verified with SimulEval's formula). Secondary summaries state "wait-k gets AL ≈ k" without the equal-length caveat. — [STACL search summary](https://arxiv.org/pdf/1810.08398); [SimulEval ALScorer](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)
- **The AI web-search summary of Hibiki contradicted itself.** It said Hibiki's LAAL is computed "without requiring word-level alignments" and in the next sentence that "the paper computes it from word-level emission timestamps produced by Whisper". LAAL as implemented (SimulEval/FBK) needs per-word emission delays. For speech output these come from forced or ASR word timestamps (SimulEval's `*_SpeechAlign_*` uses MFA). — [search result for Hibiki](https://openreview.net/pdf?id=fgjN8B6xVX); [SimulEval latency_scorer.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)
- **The AI search assistant invented content in my query.** It "corrected" a claim I never made, warning that "τ = |y|/|x|" is wrong. My query listed τ and r = |y|/|x| as separate items. The underlying point is correct, though: τ is the cut-off index argmin_t{g(t) = |x|} and r = |y|/|x| is the rate. Good to state precisely in the interview, because the two are easily conflated. — [search summary](https://ar5iv.labs.arxiv.org/html/1810.08398)
- **The mini PS's operational wait-k rule is off by one relative to STACL.** It says "English word i is spoken once k + i Hindi words have been heard", while STACL has g(t) = k + t − 1 for 1-indexed t. With 1-indexed i, the PS rule is wait-(k+1). It is consistent only if i is 0-indexed. [V-PS] — [mini PS Part E](file:///home/user/interiit_minips/Inter%20IIT%20Tech%20Meet%2015.0_%20Sarvam.ai%20Mini%20PS.pdf); [M] — [STACL](https://arxiv.org/abs/1810.08398)
- **Reference vs hypothesis length.**
  - STACL defines r = |y|/|x| with y the system output [M].
  - SimulEval's default (`use_ref_len=True`) uses the *reference* length for AL and LAAL's |Y|, but the *hypothesis* length for DAL (`target_length = len(delays)`).
  - "AL" numbers from different tools are therefore not always comparable, which is why LAAL was proposed.
  - [V-code] — [SimulEval latency_scorer.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)

### Inferences
- In the interview, give the formula, say which length convention you use, and state the equal-length caveat. This pre-empts the most common follow-up questions.

### Gaps
- Unverified Hibiki ablation numbers appeared in a search summary: ASR-BLEU/LAAL of 4.2/2.46 (no lag), 10.0/2.49 (constant 2 s lag), 22.5/9.02 (constant 10 s lag), 25.6/21.49 (sentence alignment), 17.1/14.34 (no inner monologue). They are not used anywhere above. Verify against arXiv:2502.03382 before citing.
