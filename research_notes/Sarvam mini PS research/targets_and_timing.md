# English training targets and their timing for streaming Hindi→English S2ST (Mini PS Part B Q2 + Q3)

> **How these sources were checked.** This sandbox's egress proxy blocked arxiv.org, openreview.net, aclanthology.org, huggingface.co, kyutai.org, Semantic Scholar and WebFetch. **I could not open any paper PDF or HTML directly.** Paper claims below come from web-search extracts of the original pages; the extract usually names the arXiv/ACL page it quotes. Where only a secondary summary (pith.science, MarkTechPost, liner, Lacuna) carried a claim, I label it **[secondary]**. GitHub READMEs and code were fetched raw from raw.githubusercontent.com and **are** primary-verified (**[verified]**). The Sarvam PS PDFs were read directly with `pdftotext -layout` (**[verified]**). A writer should re-check any number marked [secondary] before putting it in the final report.

## Q0. What the Sarvam PS actually constrains (ground truth for every recommendation below)

### Takeaway
The PS leaves target style and timing entirely to the team. The scoring weights intent (18), entity (18) and latency (16) far above translation quality against human references (12) and naturalness (6). So a target that is somewhat interpretation-like, causally timed and entity-exact is favoured, as long as it does not drift too far from human references.

### Cited Findings
- Part A translations "become the targets your model learns to speak. Think about what makes the best target for a streaming speech-to-speech model: how literal the translation is, and how sentence boundaries map." Numbers are written in words (digits accepted), amounts keep lakh/crore, "A sentence may carry several English translations", and "How you translate is up to you." — [Sarvam PS PDF, §3 Part A] (local: /home/user/interiit_minips/Sarvam.pdf) [verified]
- Part B: "How you arrange the English against the Hindi is up to you." All English target audio must be Bulbul `bulbul:v3`, voice `shubh`, `en-IN`, 24 kHz; "Pace and temperature are free, but declare them." English word timings must be "ordered, non-negative and inside the English target audio". Records and hours cannot be added after the mid-term. — [Sarvam PS, §3 Part B] [verified]
- The PS's own example of the core difficulty: in "मैंने कल अपना पुराना बैंक खाता बंद करवा दिया" the verb phrase "बंद करवा दिया" comes last, and "until it is heard, the system cannot tell whether the account was opened, closed or only checked." Also: "No take-backs. Speech that has been played cannot be withdrawn." — [Sarvam PS, §2] [verified]
- KPI weights: Translation quality "How well the English speech matches human reference translations" = 12; Intent accuracy (asked/refused/confirmed/corrected, negations, conditions, who does what to whom) = 18; Entity accuracy = 18; Latency ("How far the English speech trails the speaker, and how soon it finishes after the speaker stops") = 16; Audio quality and naturalness = 6. "Speech after the speaker stops counts toward latency." — [Sarvam PS, §6–§9.1] [verified]
- Self-corrections FAQ: "The references encode the corrected meaning. Speaking the abandoned value as fact is wrong." Scorer integrity: "filler speech, padding, or reading out alternatives the speaker did not offer" can lead to disqualification. Numbers: "25,000 and 25k both count as correct." — [Sarvam PS, §10–11] [verified]
- G2 requires training-pair manifests listing "its Part A record ID, the target text used and the Bulbul settings". — [Sarvam PS, §8] [verified]
- External models may be used during training "to align the English audio, as reward models or as teachers". — [Sarvam PS FAQ] [verified]
- Mini PS Q2: "How literal should the English be, and how should Hindi and English sentences map, for a model that cannot take back speech?" Q3: "How should English words be placed in time against the Hindi so the model learns when it is safe to speak? How do existing systems build such aligned pairs?" Each proposal needs a recommendation, evidence, main risk, cheapest experiment, and one question no source answers. Part E's wait-k definition: "English word i is spoken once k + i Hindi words have been heard." — [Mini PS PDF] (local: /home/user/interiit_minips/Inter IIT Tech Meet 15.0_ Sarvam.ai Mini PS.pdf) [verified]

### Inferences
- "Several English translations" per sentence lets us store **both** a fluent offline-style reference (for evaluation sanity checks and as a hedge) **and** a monotonic "interpretation-style" variant used as the training target. The G2 manifest records which text was used per pair, so the choice is auditable.
- The evaluation compares English *speech* against human references whose style we do not know. Text-only BLEU against our own fluent variant is a reasonable proxy for how much the monotonic variant costs on the 12-point quality KPI.

### Gaps
- The PS does not say whether the human references are offline-style or interpretation-style, or whether quality is scored with BLEU, COMET or an LLM judge. That decides how much a monotonic target is penalised (see Q2).

## Q1. Hibiki (arXiv 2502.03382) contextual alignment, synthetic aligned pairs, and Hibiki-Zero (arXiv 2602.11072)

### Takeaway
Hibiki aligns each target word with an off-the-shelf MT model (MADLAD-3B). It feeds the model progressively longer source prefixes and picks the source position where the target word's log-likelihood gain is largest. It then forces each target word to come **at least ~2 s after** that source word, by inserting silence or by re-synthesising with an alignment-aware TTS. Its 2026 successor Hibiki-Zero drops word-level alignment. It trains on sentence-level alignments (from punctuation, made with silence insertion) at high latency, then lowers latency with GRPO.

### Cited Findings
- Training uses "aligned source speech and target speech and text, from the same speaker". Data is synthetic. "Word-level matching is made between source and target transcripts using a *contextual alignment* weakly-supervised method that leverages an off-the-shelf MADLAD machine translation system." Rule: "a word should only appear in the target once it's predictable from the source". It is "applied either by inserting silences or by synthesizing targets with a voice controlled, alignment-aware TTS." — [kyutai-labs/hibiki README](https://github.com/kyutai-labs/hibiki) [verified]
- Formula (as extracted from the paper): for each target word, MADLAD-3B is given progressively longer truncated source input, and the alignment is the source position with the **largest log-likelihood gain**, i.e. roughly a_i = argmax_j [log p(y_i | x_≤j, y_<i) − log p(y_i | x_≤j−1, y_<i)]. — [Hibiki, arXiv 2502.03382](https://arxiv.org/abs/2502.03382) (via search extract of arXiv PDF); same description in [Pith review](https://pith.science/paper/2502.03382) [secondary]. *I could not see the exact equation typography.*
- Robustness constraints: the target is required "to lag by at least 2 seconds compared to the contextual alignment", and they "eliminate spikes higher than 25% of the average delay over a window of 5 words". — [Hibiki, arXiv 2502.03382 PDF](https://arxiv.org/pdf/2502.03382) (search extract quoting text)
- **Conflicting lag claim:** another extract (bytez mirror of the ICML version) says the authors vary "a fixed lag between aligned words, from 0.5 to 3 seconds" to set the quality/latency trade-off at training time. — [bytez ICML copy](https://bytez.com/docs/icml/44512/paper) [secondary]. Unresolved. Both may hold, e.g. a 2 s default with ablations over the lag.
- Silence-insertion limitation: it "can lead to hard cuts when the timestamps are inaccurate or no pause exists". This motivates the alignment-aware TTS, which synthesises target speech that already respects the alignment. — [Hibiki, arXiv 2502.03382](https://arxiv.org/pdf/2502.03382) (search extract)
- Speaker similarity: training examples are **labelled with discrete speaker-similarity categories rather than filtered**, and classifier-free guidance at inference favours high similarity. — [Hibiki arXiv HTML](https://arxiv.org/html/2502.03382) (search extract). README: "a larger coefficient will increase voice similarity, but excessive coefficients can lead to worse translations"; default `--cfg-coef` 1, "A typical value to use is 3." — [hibiki README](https://github.com/kyutai-labs/hibiki) [verified]
- Architecture: decoder-only, Moshi multistream; "produces text and audio tokens at a constant framerate of 12.5Hz" with "timestamped text translation"; "trained on sequences up to 120 seconds and use a context size of 40 seconds"; inference is "simple temperature sampling", no explicit policy. Fr→En only. — [hibiki README](https://github.com/kyutai-labs/hibiki) [verified]
- Results [secondary, conflicting]: Pith reports short-form Fr→En ASR-BLEU 39.2 vs Seamless 37.0, speaker similarity 0.41 vs 0.30, human ratings 3.78/5 quality and 3.73/5 naturalness vs ground-truth 4.18/4.12 ([Pith](https://pith.science/paper/2502.03382)). MarkTechPost reports 30.5 ASR-BLEU and 0.52 vs 0.43 speaker similarity ([MarkTechPost](https://www.marktechpost.com/2025/02/08/kyutai-releases-hibiki-a-2-7b-real-time-speech-to-speech-and-speech-to-text-translation-with-near-human-quality-and-voice-transfer/)), probably the long-form setting, but the article does not say. An extract says the short-form comparison shows Hibiki adding ~0.7 s lag over baselines, and defines End Offset as the time from the end of the last source word to the end of the last output word ([arXiv HTML](https://arxiv.org/html/2502.03382)). StreamSpeech "fails to produce intelligible long-form output" and is excluded from long-form ([ResearchGate copy](https://www.researchgate.net/publication/388754494_High-Fidelity_Simultaneous_Speech-To-Speech_Translation)).
- Critique: the target text was itself produced by MADLAD, so the likelihood gains are measured by the same model that wrote the target. A biased argmax could teach speaking with too little context or too much latency. — [Pith review](https://pith.science/paper/2502.03382) [secondary; reviewer opinion]
- Hibiki-Zero (Feb 2026, ICML 2026 oral): "only requires sentence-level translation alignment easily obtained from punctuation". It is first trained on sentence-level aligned data "so that the model learns to perform speech translation but with high latency", then a GRPO RL stage with a BLEU-based reward and **no KL regularisation** reduces latency. A secondary summary says BLEU is computed at multiple intermediate points, so waiting forfeits early reward. It can be adapted to a new input language "with less than 1000 hours of speech". — [Hibiki-Zero arXiv 2602.11072](https://arxiv.org/abs/2602.11072) (search extract); [Lacuna summary](https://lacuna.tiptreesystems.com/work/simultaneous-speech-to-speech-translation-without-aligned-data/wrk_dacd2cba54e585c18ef9cf3e839666e8) [secondary for the reward detail]
- Hibiki-Zero uses **silence insertion to build its coarse sentence-level alignments**. It reports ~3 ASR-BLEU over Hibiki on short-form and >5 over Seamless on long-form. — [arXiv 2602.11072 PDF](https://arxiv.org/pdf/2602.11072) (search extract). Languages: fr/es/pt/de→en, 3B parameters, 8–12 GB VRAM ([hibiki-zero README](https://github.com/kyutai-labs/hibiki-zero) [verified]). CC BY-NC-SA 4.0 per the [HF card](https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16). The ICML poster says the benchmark is 15 h while the arXiv paper says 45 h ([ICML poster](https://icml.cc/virtual/2026/poster/66094)).
- An independent long-form evaluation found Seed LiveInterpret 2.0 had the best quality but >2 s more latency than Hibiki-Zero. — [A Practical Evaluation Method for Long-Form Simultaneous S2ST, arXiv 2606.15059](https://arxiv.org/abs/2606.15059) (search extract)

### Inferences
- The Sarvam base checkpoint is "a streaming Hindi→English S2ST model trained on synthetic data" with a reference WebSocket client. That strongly suggests a Hibiki/Hibiki-Zero-style design (Kyutai's public recipe is the only well-known open synthetic-data S2ST recipe), but this is unconfirmed until the mid-term reveal.
- Two families of target timing therefore exist: (i) Hibiki word-level causal alignment plus a lag floor, giving low latency out of supervision; (ii) Hibiki-Zero sentence-level, giving high-latency supervision with latency then reduced by RL. Our budget (≤200 h, $1k Modal) makes supervised word-level alignment the more realistic main path. A sentence-level fallback (English sentence starts after its Hindi sentence ends) is the safest baseline because it can never be premature within a sentence.
- Hindi is verb-final. Under contextual alignment, English verbs (and "not") will align to the last Hindi tokens, so per-word lags will be strongly non-uniform. That is exactly the case where an adaptive rule beats uniform wait-k.

### Gaps
- Exact equation, any monotonicity post-processing (cummax), how the alignment-aware TTS is conditioned (does it take per-word target times?), how many TTS candidates are generated and selected, and training-hour counts could not be read from the PDF. One search summariser claimed a "900-hour synthetic dataset", but no primary source confirmed it, so treat it as unverified.
- Whether MADLAD's log-likelihood-gain alignment is reliable for Hindi→English, or code-mixed Hindi with Devanagari-written English words, has not been tested anywhere I found.

## Q2. Moshi "inner monologue" (arXiv 2410.00037) and how word timestamps become text tokens; the moshi-finetune data format

### Takeaway
Moshi and Hibiki predict a time-aligned text stream (word-timestamped transcript tokens padded to the 12.5 Hz audio frame rate) as a prefix to each audio frame's tokens. In the public fine-tuning code, English word timings come from `whisper_timestamped` and are stored as `[word, (start, end), speaker]` JSON alongside a stereo wav. Our Bulbul word timings will play exactly this role if the base checkpoint is Moshi-like.

### Cited Findings
- The text stream is "SentencePiece tokens applied to the Whisper transcription of the audio corresponding to Moshi". Each word's start index is its Whisper start timestamp at the 12.5 Hz frame rate, and text tokens are a per-timestep prefix before the semantic/acoustic tokens. A single delay hyper-parameter between text and audio switches the model between ASR and TTS behaviour "with no changes to the loss, architecture, or training data". — [Moshi, arXiv 2410.00037](https://arxiv.org/abs/2410.00037) (search extract; [kyutai.org/Moshi.pdf](https://kyutai.org/Moshi.pdf))
- moshi-finetune expects stereo audio where "the left channel is used for the audio generated by moshi, whereas the second channel is used for the user's input". Each wav has a `.json` "transcript with timestamps". `annotate.py` uses `whisper_timestamped` and writes `"alignments": [[word, (start, end), "SPEAKER_MAIN"], ...]`, with `--lang` (default `en`) and a note to use Whisper medium with VAD for stereo. — [moshi-finetune README](https://github.com/kyutai-labs/moshi-finetune) and [annotate.py](https://github.com/kyutai-labs/moshi-finetune/blob/main/annotate.py) [verified]

### Inferences
- If the base model is Moshi/Hibiki-like, a training pair is likely a 2-channel file: source Hindi on the input channel, delayed Bulbul English on the output channel, plus English word timestamps (on the delayed timeline) for the text stream. **Silence inserted into the English channel must shift the word timestamps too.** Aligning once on contiguous Bulbul audio and then adding the offsets is cleaner than re-aligning the silence-padded file.
- Text-token timing errors of ~1 frame (80 ms) are probably harmless, because the text stream is quantised at 80 ms anyway (see Q6 on aligner accuracy).

### Gaps
- I could not verify the exact PAD/EPAD token scheme or the default text–audio delay value in the paper text (search could not surface Appendix C).

## Q3. How literal / monotonic should the English be? Evidence on monotonic or interpretation-style references

### Takeaway
Across several groups, training simultaneous models on **more monotonic** references (pseudo-references from wait-k decoding, LLM-generated chunk-wise monotonic translations, RL-"tailored" references) improves the latency–quality trade-off, and gains are largest for grammatically distant pairs. The flip side is that offline references and standard BLEU/COMET **under-score** monotonic output. Since the Sarvam quality KPI uses human references of unknown style, aim for "chunk-wise monotonic but grammatical" targets, not word-by-word ones.

### Cited Findings
- Chen et al. 2021 (EMNLP): full-sentence parallel data has long-distance reorderings that make simultaneous models "either make bold hallucinations (for fixed-latency policies) or introduce long delays (for adaptive ones)". They generate pseudo-references by **test-time wait-k** decoding, select with beam search plus BLEU against gold, and report up to **+2.7 BLEU** on Zh→En and Ja→En. — [arXiv 2010.11247](https://arxiv.org/abs/2010.11247); [ACL 2021.emnlp-main.473](https://aclanthology.org/2021.emnlp-main.473/)
- A follow-up (Anticipation-Free Training) uses the pseudo-reference approach as a baseline, with a wait-9 model and k=9 only "since it has the best quality". — [arXiv 2201.12868](https://arxiv.org/abs/2201.12868)
- Sakai et al. 2024 (EMNLP), **LLM-SI-Corpus**: an LLM chunks the source, translates the chunks and joins them under the chunk-wise monotonic translation (CWMT) guideline, "keeping word order and preserving full source content". Fine-tuning SiMT models on it "reduces latency while achieving better quality" than other corpora, in both speech-to-text and text-to-text, for En→Ja (distant pair). — [arXiv 2404.12299](https://arxiv.org/abs/2404.12299); [ACL 2024.emnlp-main.1248](https://aclanthology.org/2024.emnlp-main.1248/)
- Simul-MuST-C (EMNLP 2024, GPT-4o-generated segment-monotonic targets): "the benefit of segment-based monotonicity depends on the grammatical distance" between source and target; distant pairs benefit most. — [ACL 2024.emnlp-main.1238](https://aclanthology.org/2024.emnlp-main.1238/)
- Guo, Zhang & Feng 2023, "Tailored Reference": an RL-trained "tailor" rephrases the ground truth per latency level. Reported effects include anticipation rate on De→En dropping from 28.17% to 19.84% at k=1, and En→Vi Wait-k+Tailor 29.49 vs 28.88 BLEU. — [arXiv 2310.13588](https://arxiv.org/abs/2310.13588) (numbers from [liner summary](https://liner.com/review/simultaneous-machine-translation-with-tailored-reference) [secondary])
- Han et al. 2021 (WMT), monotonic SimulMT with chunk-wise reordering and refinement. — [arXiv 2110.09646](https://arxiv.org/abs/2110.09646); [ACL 2021.wmt-1.119](https://aclanthology.org/2021.wmt-1.119/)
- **Evaluation risk:** NAIST (IWSLT 2024) report that scoring SimulST against an offline-based test set **underestimates** model performance, and that CMT references give higher scores to SimulST models than to offline ST models. — [ACL 2024.iwslt-1.30](https://aclanthology.org/2024.iwslt-1.30.pdf). Simul-COMET (Findings ACL 2026): standard COMET "tends to assign higher scores to offline translations with long-distance reordering". — [ACL 2026.findings-acl.2110](https://aclanthology.org/2026.findings-acl.2110/). Zhao et al. 2021 report a **13.83-BLEU** gap when the same output is scored against interpretation vs translation references (single study, via search extract). Analysis of En–Ja SI word order with CWMT: [arXiv 2406.08940](https://arxiv.org/abs/2406.08940).
- Disfluent source, fluent target: Salesky et al. train end-to-end models from disfluent Spanish speech directly to **fluent** English using copy-edited Fisher references, as a baseline for implicit disfluency removal. — [NAACL 2019, arXiv 1906.00556](https://arxiv.org/abs/1906.00556); corpus paper [arXiv 1811.03189](https://arxiv.org/abs/1811.03189)

### Inferences
- Hindi→English is a distant (SOV→SVO, postpositions→prepositions) pair, so the Simul-MuST-C finding predicts that monotonic targets help **more** here than for European pairs.
- Recommended target style: **chunk-wise monotonic**. Keep the Hindi chunk order (time adverbials, topics, objects) wherever English grammar tolerates fronting, but never produce ungrammatical word-by-word English. All entities, negations and conditions must be explicit and come out exactly. Keep a fluent variant as an alternative in Part A.
- Quantify the cost: score the monotonic variant against the fluent variant (BLEU/chrF/COMET) on a held-out slice. If the drop is large (e.g. >10 BLEU), move toward fluency, because the quality KPI uses human references, probably offline-style.

### Gaps
- No study I found measures monotonic targets for **Hindi→English** or for **speech output** (all the evidence is S2T/T2T, mostly En→Ja and Zh/Ja→En).

## Q4. SOV Hindi → SVO English: concrete re-ordering strategies and their risks

### Takeaway
English has cheap, grammatical "verb-deferring" constructions that let it follow Hindi order: fronted time adverbials, causative "got/had X Ved", object topicalisation with a resumptive pronoun, and "what happened is…". These delay the commitment to the predicate without hallucinating it. Overusing them hurts naturalness (6 pts) and reference-match (12 pts). Use them only where the alternative is a long silent wait or a guessed verb.

### Cited Findings
- The PS's own gloss "Yesterday I got my old bank account closed" already defers the lexical verb ("closed") to the end via a causative, matching Hindi "बंद करवा दिया". — [Sarvam PS §2] [verified]
- Verb-final sources force a simultaneous translator either to predict the unseen verb or to wait until sentence end. — [Sarkar PACLIC 30 keynote](https://www.cs.sfu.ca/~anoop/2016/10/30/paclic30-keynote/); [Shonan seminar No. 068 report](https://shonan.nii.ac.jp/docs/No-068.pdf)
- Anticipation-heavy targets teach hallucination under fixed policies. — [Chen et al. 2021, arXiv 2010.11247](https://arxiv.org/abs/2010.11247)

### Inferences
- Example ladder for "मैंने कल अपना पुराना बैंक खाता बंद करवा दिया":
  1. Fluent: "Yesterday I got my old bank account closed." Already monotonic for everything except the subject/adverb swap; "got" commits only to a causative, and "closed" waits for "बंद".
  2. Topicalised (more monotonic, slightly marked): "Yesterday, my old bank account, I got it closed."
  3. Word-by-word (reject): "I yesterday my old bank account closed got."
  Prefer (1) when it is already causal under the alignment rule, (2) only when the object is long (an entity-heavy noun phrase the listener would otherwise wait several seconds for), never (3).
- **Negation and conditions are the biggest intent risk.** Hindi negation ("नहीं/मत") sits immediately before the final verb, so any English that commits to "I closed…" before hearing it is a no-take-back intent error. The target style should let "didn't/not" come after the Hindi negation, and the timing rule must anchor the English verb **and** its auxiliary/negator to the Hindi verb group, not to the subject.
- Quotative/complement clauses ("…कि …") and relative-correlatives (जो…वो) are other places where a monotonic paraphrase ("He said: …") gives large latency savings.
- A cheap way to generate both variants: prompt an LLM with explicit CWMT-style guidelines (as LLM-SI-Corpus did) plus an entity/negation checker (Mini PS Part D) as a filter.

### Gaps
- No source evaluates listener naturalness (MOS) of topicalised/monotonic English **speech**. NaturalFlow (arXiv 2606.13121) studies pauses, not syntax. I found no Hindi-specific SimulMT/SimulST paper from 2025–26.

## Q5. Sentence-boundary mapping and disfluencies / self-corrections

### Takeaway
Map **one Hindi sentence → one English sentence (or a short sequence of English sentences)** and never pull content forward across a Hindi sentence boundary. Split long Hindi run-ons into several English sentences rather than merging. For self-corrections, the English encodes only the corrected value, and its timing must be anchored to the correction, not to the abandoned value.

### Cited Findings
- Hibiki-Zero gets its sentence-level alignment "from punctuation" and uses it as the first (high-latency) training stage. — [arXiv 2602.11072](https://arxiv.org/abs/2602.11072) (search extract)
- The Part A transcript rule "Every sentence ends with sentence-final punctuation" gives a ready sentence segmentation, and "A sentence may carry several English translations." — [Sarvam PS §3] [verified]
- "The references encode the corrected meaning. Speaking the abandoned value as fact is wrong." — [Sarvam PS FAQ] [verified]
- Fluent-target training from disfluent speech is an established approach. — [Salesky et al. 2019, arXiv 1906.00556](https://arxiv.org/abs/1906.00556)

### Inferences
- Splitting is safe for streaming because shorter English sentences can be committed earlier. Merging two Hindi sentences into one English sentence creates cross-sentence dependencies that the timing rule must respect, so allow it only when the second Hindi sentence is a fragment.
- Per-sentence correspondence should be stored explicitly (Hindi sentence span ↔ English sentence span). It enables the sentence-level fallback timing, sentence-level BLEU checks, and Hibiki-Zero-style stage-1 training.
- **Self-correction timing rule (critical):** for "पाँच लाख… नहीं, छह लाख", the English "six lakh" must be placed after the Hindi "छह लाख" is heard. An automatic aligner (awesome-align/SimAlign) may link "lakh" to the first "लाख", which would teach the model to speak the number early, i.e. to speak the abandoned value. Detect correction patterns (नहीं/मतलब/sorry/sorry-variants, repeated entity type) and force those English entity tokens to align to the **last** matching Hindi span. This is a cheap rule with a large intent/entity payoff.
- Filled pauses and false starts in the Hindi produce no English, so their duration is free latency budget during which the English channel can catch up.

### Gaps
- No source evaluates whether training on corrected-meaning targets causes a streaming S2ST model to wait (good) or to omit early content (bad) when it hears hesitation.

## Q6. English word-level aligners for Bulbul TTS audio

### Takeaway
On clean read English, MFA (english_us_arpa) remains the boundary-accuracy leader, at ~19 ms mean word-boundary error and ~68% within 20 ms on TIMIT. Neural multilingual aligners (MMS, Qwen3-ForcedAligner) sit around 30 ms. All are far below the 80 ms Moshi frame, so the aligner choice matters less for precision than for **robustness** to numbers, Indian names and lakh/crore. Normalise numbers to words before aligning, and validate with Bulbul-specific hand checks.

### Cited Findings
- TIMIT comparison: MFA english_us_arpa places 68.2% of word boundaries within 20 ms (mean 19.1 ms); Qwen3-ForcedAligner 30.1 ms, MMS 29.2 ms; a fine-tuned Whisper aligner 11.6 ms. — ["Where Words Begin and End", arXiv 2610.17782](https://arxiv.org/abs/2610.17782) (very recent preprint; search extract only). An overview cites earlier MFA figures of ~65.7% within 20 ms and 21.9 ms mean deviation on TIMIT. — [Emergent Mind MFA overview](https://www.emergentmind.com/topics/montreal-forced-aligner) [secondary]
- Human ceiling: Buckeye transcribers agreed on only 43% of word boundaries exactly, ~26 ms apart on average. An 80 ms timestamp grid caps F1@20 ms near 0.5. — [FuseAlign, arXiv 2609.33650](https://arxiv.org/abs/2609.33650); [FA-Bench, arXiv 2609.32396](https://arxiv.org/abs/2609.32396) (search extracts)
- Qwen3-ForcedAligner-0.6B: AAS 42.9 ms vs WhisperX 133.2 ms, NeMo Forced Aligner 129.8 ms and Monotonic-Aligner 161.1 ms on **MFA-labelled** data. It is trained on MFA pseudo-labels, so the benchmark favours it. It claims a 67–77% relative AAS reduction on human-labelled sets. — [Qwen3-ASR Technical Report, arXiv 2601.21337](https://arxiv.org/abs/2601.21337) (search extract)
- WhisperX limitation: "Transcript words which do not contain characters in the alignment models dictionary e.g. '2014.' or '£13.60' cannot be aligned and therefore are not given a timing." — [WhisperX README](https://github.com/m-bain/whisperX) [verified]
- torchaudio MMS_FA multilingual tutorial: the transcript "must be normalized beforehand", romanised with `uroman`, with a `<star>` token for missing transcript. **Its banner says: "The APIs described in this tutorial are deprecated in 2.8 and will be removed in 2.9."** — [torchaudio forced_alignment_for_multilingual_data_tutorial.py](https://github.com/pytorch/audio/blob/main/examples/tutorials/forced_alignment_for_multilingual_data_tutorial.py) [verified]
- moshi-finetune's own annotator uses `whisper_timestamped`. — [annotate.py](https://github.com/kyutai-labs/moshi-finetune/blob/main/annotate.py) [verified]

### Inferences
- We know the Bulbul input text exactly, so **forced** alignment (not ASR timestamps) is the right tool. Writing numbers as words in the target (allowed by the PS) removes the WhisperX digit problem. "lakh"/"crore" and Indian names may be out-of-vocabulary for MFA dictionaries, so they need G2P fallback or custom pronunciations (e.g. `lakh L AA1 K`), and should be validated.
- Practical pick: MFA english_mfa/english_us_arpa as primary, with a CTC aligner (MMS_FA via a pinned torchaudio ≤2.8, or Qwen3-ForcedAligner) as a cross-check. Flag words where the two disagree by >80 ms, or where an ASR round-trip (Whisper on the Bulbul audio) disagrees with the text. This doubles as the PS-required TTS QC ("skip or repeat words").
- Bulbul output is clean, single-speaker TTS, so alignment should be easier than TIMIT/Buckeye, though Indian-English vowels may mismatch US-English acoustic models.

### Gaps
- No source reports aligner accuracy on **Indian-English TTS** (or on Bulbul specifically), or on lakh/crore/Indian names. That must be measured by hand-labelling (~30–50 boundaries, as in Part D).

## Q7. How other systems built aligned S2ST / SimulST training pairs

### Takeaway
There are three main approaches: (1) explicit causal data alignment (Hibiki's MT-likelihood alignment plus silence/TTS; TransLLaMa/Hikari's WAIT-token insertion based on word alignment); (2) latent alignments learned during training from sentence-level pairs (StreamSpeech CTC, NAST-S2X CTC, Seamless EMMA); (3) sentence-level pairs plus RL for latency (Hibiki-Zero). Only (1) gives the model an explicit "safe to speak" signal in the data, which suits a no-take-back setting with a small fine-tuning budget.

### Cited Findings
- TransLLaMa (Koshkin, Sudoh, Nakamura; Findings EMNLP 2024): fine-tunes a decoder-only LLM on "causally aligned" pairs by inserting `<WAIT>` tokens so that "target content words never appear earlier than their closest equivalents in the source". A target word can sit anywhere between neighbouring aligned words. — [arXiv 2402.04636](https://arxiv.org/abs/2402.04636); [ACL 2024.findings-emnlp.27](https://aclanthology.org/2024.findings-emnlp.27/) (search extract)
- Hikari (Koshkin et al., March 2026): a policy-free S2T model that encodes READ/WRITE as probabilistic WAIT tokens, built on causal alignment. It adds "Decoder Time Dilation" against over-represented filler tokens, plus SFT to recover from delays. It reports SOTA BLEU on En→Ja/De/Ru. — [arXiv 2603.11578](https://arxiv.org/abs/2603.11578) (abstract-level)
- StreamSpeech (ACL 2024): learns translation and the simultaneous policy jointly via multi-task learning. Its READ/WRITE policy comes from CTC decoders (ASR, non-autoregressive S2TT, speech-to-unit) trained on CVSS. — [arXiv 2406.03049](https://arxiv.org/abs/2406.03049) (search extract)
- NAST-S2X: non-autoregressive speech-to-text/speech over fixed-length chunks, trained with "a CTC-based non-monotonic latent alignment loss together with a glancing mechanism", on CVSS-C Fr→En. — [HF ICTNLP/NAST-S2X](https://huggingface.co/ICTNLP/NAST-S2X)
- SeamlessStreaming uses EMMA, a monotonic-attention policy with "numerically stable, unbiased estimation of monotonic alignments", learned without word-level supervision. — [EMMA arXiv 2312.04515](https://arxiv.org/abs/2312.04515); [Seamless arXiv 2312.05187](https://arxiv.org/abs/2312.05187)
- NaturalFlow (Interspeech 2026): low-latency S2ST tends to produce chunk-by-chunk output with frequent pauses that listeners find unnatural. It reduces inter-chunk silences using model-internal signals. — [arXiv 2606.13121](https://arxiv.org/abs/2606.13121) (abstract-level)
- Hindi–English word alignment is hard for unsupervised aligners: AER on en-hi is SimAlign 0.44 and PMIAlign/OTTAWA 0.39. — [OTTAWA, arXiv 2406.01919](https://arxiv.org/abs/2406.01919) (search extract). Naamapadam used GIZA++ and awesome-align fine-tuned with TLM/self-training, intersected in both directions, for Indic projection. — [arXiv 2212.10168](https://arxiv.org/abs/2212.10168)

### Inferences
- Since en-hi AER is ~0.4 for unsupervised aligners, a pure word-alignment lag rule will be noisy. Combine it with Hibiki-style MT-likelihood alignment (MADLAD-400 covers Hindi, and IndicTrans2 or an LLM could also score) and take the **later** of the two, which is conservative and no-take-back-safe. Alternatively, use an LLM to output alignments directly and spot-check them.
- Silence insertion produces exactly the "frequent pauses" NaturalFlow criticises. Mitigate by inserting silence only at existing Bulbul prosodic breaks, by synthesising per chunk, or by raising the Bulbul pace for chunks that must catch up (pace is a free declared parameter).

### Gaps
- No public system builds aligned pairs for Hindi or code-mixed Hindi; no source compares data-level causal alignment against learned latent alignment at equal data size.

## Q8. Concrete recipe, comparison with wait-k and Hibiki, cheapest experiment; one misleading claim; one unanswered question

### Takeaway
Recommended pipeline: (i) chunk-wise monotonic English target (fluent variant kept as an alternative). (ii) For each English word, an earliest-safe Hindi time = the end of its latest dependency (the later of MT-likelihood alignment and word alignment, with correction/negation overrides) plus a margin δ. (iii) Place one contiguous Bulbul rendering by inserting silence only at its prosodic breaks, so that each word's onset ≥ its safe time, with timestamps shifted accordingly. (iv) Compare against wait-k and sentence-level schedules first in a **text-and-timestamp simulation** before spending GPU time.

### Cited Findings
- Hibiki's ≥2 s lag floor and spike smoothing (>25% of the 5-word average delay) — [arXiv 2502.03382](https://arxiv.org/pdf/2502.03382) (search extract); the WAIT-insertion causality rule — [TransLLaMa arXiv 2402.04636](https://arxiv.org/abs/2402.04636); the Mini PS wait-k definition "English word i is spoken once k + i Hindi words have been heard" — [Mini PS PDF] [verified].
- **Claim that turned out misleading on checking:** Hibiki-Zero's title, "Simultaneous Speech-to-Speech Translation **Without Aligned Data**", and summaries saying it "eliminates the need for word-level alignments entirely" read as if no alignment is needed. The paper itself says it "only requires sentence-level translation alignment" and builds those coarse alignments **with silence insertion** before RL. Aligned (sentence-level) data is still a prerequisite. — [arXiv 2602.11072](https://arxiv.org/abs/2602.11072); [MarkTechPost headline](https://www.marktechpost.com/2026/02/13/kyutai-releases-hibiki-zero-a3b-parameter-simultaneous-speech-to-speech-translation-model-using-grpo-reinforcement-learning-without-any-word-level-aligned-data/). A second check: the torchaudio multilingual forced-alignment tutorial, which the Mini PS recommends, carries its own warning that its APIs are deprecated in torchaudio 2.8 and removed in 2.9 ([tutorial source](https://github.com/pytorch/audio/blob/main/examples/tutorials/forced_alignment_for_multilingual_data_tutorial.py) [verified]). Pin torchaudio ≤2.8 or use another aligner. Hibiki's headline numbers also conflict between secondary sources (39.2 vs 30.5 ASR-BLEU; 0.41 vs 0.52 speaker similarity), see Q1.

### Inferences
**Recipe (per Part A sentence pair):**
1. Inputs: Hindi word times (Part A), the Hindi transcript, and the chosen English target (monotonic variant).
2. Dependencies: for each English word e_i compute
   - a_i^MT = Hibiki-style argmax_j of the log-likelihood gain under an MT model that covers Hindi (MADLAD-400-3B, or IndicTrans2 rescoring, or an LLM). A more conservative variant: the smallest j at which log p(e_i | x_≤j, e_<i) comes within ε of its full-source value.
   - a_i^WA = max aligned Hindi index from awesome-align/SimAlign (content words only; function words inherit from their head).
   - Overrides: negators/auxiliaries go to the Hindi verb group; entities in self-corrections go to the last matching span; numbers go to the last number word of the Hindi amount (e.g. "lakh" in "पाँच लाख पचास हज़ार" waits for "हज़ार").
   - a_i = cummax(max(a_i^MT, a_i^WA)), which keeps the schedule monotone.
3. Safe time s_i = end_time(Hindi word a_i) + δ, with δ swept over {0.5, 1, 2} s (Hibiki's floor is ~2 s). Smooth spikes as Hibiki does.
4. Synthesise the full English sentence with Bulbul once (declared pace/temperature), force-align it (MFA plus a CTC cross-check), and get word onsets o_i in TTS time.
5. Placement: choose the global offset and the silences inserted at Bulbul pause points (or between words when no pause exists, accepting a cut) such that offset + o_i + inserted_before_i ≥ s_i for all i, minimising total delay. This is a simple greedy pass, left to right. Optionally re-synthesise long chunks separately, or raise the pace, when the cumulative lag exceeds a cap (e.g. 4 s), to stop latency drifting (the PS counts "speech after the speaker stops" toward latency).
6. Shift the English word timestamps by the same offsets. Check "ordered, non-negative, inside audio". Write the pair (record ID, target text, Bulbul settings, δ) to the G2 manifest.

**Comparisons:**
- *Uniform wait-k* (Mini PS definition) ignores dependencies. On verb-final Hindi it is either premature on verbs and negations (small k) or slow on everything (large k).
- *Sentence-level* (Hibiki-Zero stage 1) is never premature within a sentence but has maximal latency.
- *Hibiki contextual alignment* is the MT-only version of step 2. Ours adds word-alignment, negation and correction overrides, because a no-take-back error on intent/entities costs more here (36 pts) than latency (16 pts).

**Cheapest experiment (no GPU, ~1 day):** take ~100 Part A sentences (plus the 6 Part D clips) and hand-mark, for 50 of them, the Hindi word after which each English content word is first "safe". Compute schedules for wait-k (k = 2…8), sentence-level, Hibiki-MT-only, and the combined rule at δ ∈ {0.5, 1, 2}. Measure:
- (a) **premature-commit rate**, the fraction of English content words scheduled before their hand-marked safe point, reported separately for negations, verbs and entities;
- (b) LAAL / end offset of the schedule on the Hindi timeline;
- (c) total inserted silence and the number of pauses longer than 300 ms (naturalness proxy).

The rule that minimises (a) at a given (b) wins. Then run a 2-arm LoRA fine-tune (identical data, only the timing rule differs) once the checkpoint arrives, with intent/entity/latency measured on a held-out set. Run the aligner check in parallel: hand-label 30–50 Bulbul word boundaries, including lakh/crore and names, and compare MFA, CTC and Whisper errors.

**Main risks:** monotonic targets diverge from human references (quality KPI); silence insertion yields choppy speech (naturalness KPI); en-hi word-alignment noise (AER ~0.4) makes schedules either premature or over-conservative; the base checkpoint may have been trained with a different alignment convention, so fine-tuning on a new one could first raise errors.

### Gaps
- **Unanswered question that matters most:** what timing convention (word-level contextual alignment and its lag floor, or sentence-level plus RL) and what target style was the Sarvam base checkpoint trained on, and does fine-tuning with a different per-word lag rule help or destabilise it with only ≤200 h? No source can answer this before the mid-term reveal. Hibiki-Zero's "<1000 h to add a new language" result is the nearest evidence and concerns a different setting.
- A second open question: how accurately standard aligners place word boundaries on Bulbul v3 `shubh` en-IN audio, especially "lakh", "crore", Indian names and code-mixed English terms. No published measurement exists, so it must be measured.
