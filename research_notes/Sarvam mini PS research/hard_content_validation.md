# Hard Content (Entities, Numbers, Negations, Self-Corrections) and Validation of TTS/Translation Data for a Hindi Code-Mixed S2ST Dataset

Scope: Mini PS Part B Q4 (hard content), Part C "Validation" and "Failure modes", and the Part D checker. PS facts used below: PS (Sarvam.pdf) requires verbatim IndicVoices Level 1 Hindi transcripts with "numbers written as spoken words", English words in Devanagari, disfluencies kept; English targets "write numbers in words. Digits are also accepted", "amounts keep the speaker's units (lakh, crore)"; "If you use TTS-generated speech anywhere, check it and filter or regenerate bad output. TTS can mispronounce names and numbers, skip or repeat words, or add artefacts"; the FAQ says equivalent forms (25,000 / 25k) count as correct, and for self-corrections "the references encode the corrected meaning. Speaking the abandoned value as fact is wrong." Intent accuracy and entity accuracy are 18 points each. The Mini PS Part D wants "a checker that verifies amounts, numbers, names and negations in the English", in "plain Python with regular expressions and a table of Hindi number words".

Method note (read this first): in this session arxiv.org, huggingface.co, aclanthology.org, isca-archive.org and docs.sarvam.ai could not be fetched (the proxy returned 403 / DNS failure). Paper findings below come from search-engine extracts of the primary pages, linked to the primary URL. Where an extract was ambiguous I say so. The tool findings (num2words, indic-num2words, NeMo text-processing) were checked by hand: I installed or downloaded the packages from PyPI on 2026-10-14 and ran them.

## Q1. Getting entity-rich, code-mixed content into the data (templates, LLM scripts, entity samplers, code-mixing generation)

### Takeaway
Prior work agrees: TTS-synthesised, entity-dense text does fix rare-entity and number errors (Google numerics gave up to 8x lower WER on long digit strings; Amazon OOV). It only works when the speakers and the lexicon are diverse, and it transfers worst to Hindi. The one Indic entity-flywheel study found its Hindi model behind commercial ASR. So entity-heavy synthetic speech should be a targeted slice built from samplers and LLM call scripts, plus real recordings, and not the bulk of the 200 h.

### Cited Findings
- Google (Peyser, Zhang, Sainath, Wu, Interspeech 2019) generated extra numeric training utterances with TTS and added a small neural "denorming" network (spoken to written form). WER on the longest numeric sequences fell by up to 8x. A preprint version reports relative WER gains of 65% (small numbers) and 35% (large numbers) from TTS data in the written domain. The largest gains came from TTS data plus spoken-domain targets plus denorming together. — [arXiv 1907.01372](https://arxiv.org/pdf/1907.01372); [ISCA](https://www.isca-archive.org/interspeech_2019/peyser19_interspeech.html). (The per-class figures come from a ResearchGate copy of the preprint, so check them against the final version.)
- Amazon (Zheng, Liu, Gunceler, Willett, ICASSP 2021) used TTS audio for out-of-vocabulary words and named entities. An RNN-T was fine-tuned on mixed real and synthetic audio with EWC regularisation, to keep accuracy on non-OOV words. The dataset and the TTS are proprietary. A 2023 replication found that a spelling-correction model beat TTS-data fine-tuning. — [arXiv 2011.11564](https://arxiv.org/pdf/2011.11564); [Amazon Science](https://www.amazon.science/publications/using-synthetic-audio-to-improve-the-recognition-of-out-of-vocabulary-words-in-end-to-end-asr-systems); [arXiv 2302.09723](https://arxiv.org/pdf/2302.09723)
- Rosenberg et al. (Google, ASRU 2019): TTS augmentation helps ASR, but speaker diversity and lexical diversity of the synthesised sentences matter. A single-speaker TTS with better audio quality gave much worse ASR results than a multi-speaker one. — [arXiv 1909.11699](https://arxiv.org/pdf/1909.11699)
- "The TTS–STT Flywheel" (Menta, arXiv 2605.03073, May 2026): about 22k LLM-written, entity-dense utterances (digit strings, currency, addresses, code-mixed phrases) in Telugu, Hindi and Tamil, voiced by several Indic TTS systems. It used per-class CER filtering and "spelled-digit text rewriting" so that the labels match what the TTS actually said. Telugu Entity-Hit-Rate was 0.473 against 0.027 for the open baseline and 0.16 for Deepgram Nova-3. Hindi EHR was only 0.337, below its pre-registered 0.65 target, and on Hindi it "underperforms commercial" systems. The real-speech check used only n=20 Telugu utterances (0.516). It is a single-author preprint and not peer-reviewed. Code is listed at github.com/praxelhq/stt-flywheel. — [arXiv 2605.03073](https://arxiv.org/abs/2605.03073)
- Cornell, Darefsky, Duan, Watanabe (SynData4GenAI 2024): Llama-3-written two-speaker conversations, voiced by the conversational TTS Parakeet, were used to fine-tune Whisper for Fisher and Mixer-6. This beat classical multi-speaker synthesis from non-conversational corpora but did not match real in-domain data. Parakeet struggled with long multi-speaker conversations. — [arXiv 2408.09215](https://arxiv.org/abs/2408.09215)
- GCM toolkit (Rizvi, Srinivasan, Ganu, Choudhury, Sitaram; EACL 2021 demo, microsoft/CodeMixed-Text-Generator) generates code-mixed sentences from parallel sentence pairs using Equivalence Constraint and Matrix Language theories. Under EC theory, switches are allowed only where the two languages' surface structures map onto each other. The toolkit samples among candidates to keep natural ones. X-RiSAWOZ used GCM for English-Hindi and fell back to word-alignment substitution when GCM produced no candidate. — [ACL Anthology](https://aclanthology.org/2021.eacl-demos.24.pdf); [GitHub](https://github.com/microsoft/CodeMixed-Text-Generator); [X-RiSAWOZ arXiv 2306.17674](https://arxiv.org/pdf/2306.17674)
- Quality of synthetic Hinglish is hard to predict automatically. In HinglishEval (INLG 2022), two annotators rated synthetic Hinglish 1–10, and the best quality-prediction systems were weak (one reports F1 0.11 and MSE 6.0). — [arXiv 2108.01861](https://arxiv.org/pdf/2108.01861); [HinglishEval overview](https://aclanthology.org/2022.inlg-genchal.3.pdf)
- An EMNLP 2025 paper generated code-mixed text by word substitution plus GPT-4 prompting. It filtered with M-index and I-index thresholds of 0.1, which removes monolingual or barely mixed sentences. These indices measure how much mixing there is, not how natural it is. — [ACL Anthology 2025.emnlp-main.109](https://aclanthology.org/2025.emnlp-main.109.pdf)
- MASSIVE localises the English SLURP NLU set in parallel into 51 languages, with 60 intents, 55 slot types and 18 domains. Speech-MASSIVE adds crowd-sourced (not TTS) speech in 12 languages under CC BY-NC-SA 4.0. — [MASSIVE arXiv 2204.08582](https://arxiv.org/pdf/2204.08582); [Speech-MASSIVE arXiv 2408.03900](https://arxiv.org/abs/2408.03900v1)
- NeMo's Hindi ITN telephone grammar carries context-cue words that announce a number type: नंबर, मोबाइल, फोन, कॉल, लैंडलाइन, पिन, कोड, पिनकोड, कार्ड, क्रेडिट. It also has tables for English digit words and English teens/tens spoken inside Hindi. So phone numbers spoken with English digit words is a pattern the grammar was built to handle. — [nemo_text_processing 1.2.0 on PyPI](https://pypi.org/project/nemo-text-processing/1.2.0/) (files `inverse_text_normalization/hi/data/telephone/*.tsv`, inspected locally)

### Inferences
- Recommended generator: an LLM writes call-centre scenario scripts (banking, UPI, telecom recharge, e-commerce returns, insurance, loans, travel). The LLM receives slots filled by programmatic entity samplers and does not invent the entities. This way every utterance carries gold entity values for later checks. The samplers should cover:
  - Indian person names across regions and religions, cities and PIN codes, Indian organisations and products (brand names spoken in Devanagari).
  - Amounts drawn log-uniformly from ₹10 to ₹100 crore, spoken in lakh/crore with साढ़े/सवा/पौने/डेढ़/ढाई.
  - 10-digit mobile numbers starting with 6–9, spoken digit-by-digit, in pairs or groups, with "डबल" or "ट्रिपल", and with English digit words.
  - Account, IFSC and OTP strings; dates and times (e.g., "पंद्रह अगस्त", "सवा तीन बजे").
  - Negations (नहीं, मत, ना, कभी नहीं), conditionals (अगर…तो, वरना), and repairs.
- Generate the Devanagari Level-1 spoken form and a structured gold record (entity type, value, span) together, never by post-hoc ITN. The Google and Flywheel results both depend on the labels matching the spoken form.
- Use LLM prompting rather than GCM for natural Hinglish, but keep GCM/EC-style constraints in mind, and filter by M-index / CMI > ~0.1 so that some utterances really mix. No source shows that GCM beats LLM prompting on naturalness. The HinglishEval results warn that automatic naturalness scoring is unreliable, so a human must spot-check.
- Rosenberg's diversity result implies many TTS voices, plus voice cloning from real-speaker prompts, rather than one high-quality voice.

### Gaps
- No source reports how much entity-dense synthetic speech (hours or ratio) is optimal inside a mixed real+synthetic budget, or how it affects a speech-to-speech translation model, as opposed to ASR.
- No corpus documents how Hindi speakers actually say phone numbers (pairs vs digits vs English digits); a sampler distribution has to be guessed or measured on real data (e.g., IndicVoices conversational subset).
- I could not confirm whether MASSIVE hi-IN exists or whether Speech-MASSIVE includes Hindi audio. Check the HF cards (not fetchable here).

## Q2. Disfluency and self-correction injection

### Takeaway
Rule-based injection (LARD) and LLM-based injection (LLM-Dys, the 2412.12710 insertion model, DRIVE) are both established. LLM injection gives more natural, context-appropriate disfluency than random rule-based insertion. Their known failures are that rules insert one disfluency per utterance and that LLM-based TTS over-stresses or mishandles repetitions. For this PS the important disfluency is the content repair (abandoned value → corrected value), and it must be carried as metadata so that the English target says only the corrected value.

### Cited Findings
- LARD (Passali et al., LREC 2022; the correct arXiv ID is 2201.05041, not 2211.11863) generates repetitions, replacements and restarts from fluent text, using the reparandum/interregnum/repair scheme. It was motivated by class imbalance in existing disfluency data. — [ACL Anthology 2022.lrec-1.249](https://aclanthology.org/2022.lrec-1.249/); [arXiv 2201.05041](https://arxiv.org/abs/2201.05041); scheme description in [arXiv 2211.09235](https://arxiv.org/html/2211.09235)
- LARD inserts only one disfluency per utterance, according to a Stanford CS224N report. The same report says the newer NLP-based LARD version was not public. — [CS224N report](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/final-reports/final-report-169616647.pdf)
- 2412.12710 fine-tunes an LLM on Switchboard annotations to insert typical disfluencies (filled and silent pauses) and atypical ones (repetition, substitution, insertion), then voices them with Bark TTS. — [arXiv 2412.12710](https://arxiv.org/html/2412.12710)
- LLM-Dys (Interspeech 2025) is an LLM-written dysfluent text corpus synthesised with TTS, covering 11 categories at word and phoneme level. Its size is reported inconsistently (">10,000 h" in the intro, about 12,790 h in the statistics). LLMs tend to repeat pronouns and prepositions and to use fillers for insertions. — [arXiv 2505.22029](https://arxiv.org/abs/2505.22029)
- DRIVE generates disfluency-rich in-car dialogues by prompting, including self-corrections, and beats LARD-style post-hoc insertion on naturalness and contextual appropriateness. — [arXiv 2507.19867](https://arxiv.org/html/2507.19867v1)
- Koel-TTS (EMNLP 2025) notes that LLM-based TTS can over-emphasise scripted repetitions and lose the intended flow. — [ACL Anthology 2025.emnlp-main.1076](https://aclanthology.org/2025.emnlp-main.1076.pdf)
- Hossain et al. (Findings EMNLP 2020): negation hurts MT across 17 directions, with quality reductions above 60% in some cases. Tang & Rönchen (TACL 2021) find that NMT models are more confused by deleted negation than by inserted negation, using the LingEval97 polarity category with 26,803 instances. — [arXiv 2010.05432](https://arxiv.org/pdf/2010.05432); [TACL / arXiv 2107.12203](https://arxiv.org/pdf/2107.12203)

### Inferences
- Use a Hindi repair grammar driven from the structured record. The record lists entity slots {abandoned value, corrected value, interregnum}. The interregnum comes from a set of Hindi editing terms such as "नहीं नहीं", "सॉरी", "मतलब", "मेरा मतलब", "एक मिनट", "अरे", "माफ़ कीजिए", "नहीं, रुको". The repair types:
  - value repairs: "पाँच लाख… नहीं नहीं, पाँच करोड़"
  - digit repairs inside a phone number: "नौ आठ सात… सॉरी, नौ आठ छह…"
  - name repairs
  - negation flips: "हाँ कर दो… नहीं, मत करो"

  The English target and the checker's gold always use the corrected value. Store the abandoned value as a forbidden entity, so the checker can flag "abandoned value spoken as fact" (PS FAQ rule).
- Fillers (उम्म, अं, हम्म, मतलब, वो, ना) should be inserted at a few per utterance, and not just one as in LARD. The rate should be calibrated against a real Hindi conversational sample, not set by hand.
- TTS-rendered fillers and repetitions are the items most likely to be dropped or over-stressed. Treat them as entities in the round-trip check (Q4) and accept a filler mismatch more leniently than an entity mismatch.
- Negation needs dedicated minimal pairs (same sentence with and without नहीं/मत) in training and in the checker test set, because MT literature shows negation is a disproportionate error source.

### Gaps
- No published study evaluates disfluency or self-repair synthesis in Hindi or Hinglish, or how Indic TTS (Bulbul, IndicF5, Indic Parler-TTS) renders Hindi fillers like "उम्म"/"अं". This must be measured empirically.
- Real Hindi conversational disfluency and repair rates (per 100 words) were not found in any source.

## Q3. Hindi number verbalisation, ITN, and a number-word ↔ value parser for the checker

### Takeaway
Use NVIDIA NeMo text-processing (1.2.0) Hindi TN/ITN as the reference and as the source of the checker's word table. It covers lakh/crore/arab/kharab, साढ़े/सवा/पौने/डेढ़/ढाई, telephone, money, dates and time, and it lists 220 spelling variants for 10–99. Python `num2words` has no Hindi. `indic-num2words` uses non-standard spellings and breaks at or above 100 crore. Always NFC-normalise, because nukta letters come in two Unicode forms.

### Cited Findings
- `num2words` 0.5.14 (latest on PyPI at test time) raises `NotImplementedError` for `lang='hi'`. Its converter list includes bn, kn, te and en_IN but no hi. `en_IN` verbalises the English side in the Indian system, e.g. 2500000 → "twenty-five lakh" and 50000000 → "five crore". Tested locally. — [PyPI num2words](https://pypi.org/project/num2words/); [GitHub savoirfairelinux/num2words](https://github.com/savoirfairelinux/num2words)
- `indic-num2words` 1.3.2 (`from num_to_words import num_to_word`), tested locally:
  - Gives 87 → सत्तासी, 69 → उन्हत्तर, 2500000 → पच्चीस लाख, 50000000 → पाँच करोड़.
  - Uses non-standard spellings: 39 → उन्तालिस, 99 → निन्यान्बे, 78 → अठ्हत्तर.
  - Inserts commas ("एक सौ, एक").
  - For 1,234,567,890 it returns digit-pair groups ("बारह, चौतीस, छप्पन, अठ्हत्तर, नब्बे") rather than अरब.
  - Rejects decimals (1.5).
  — [PyPI indic-num2words](https://pypi.org/project/indic-num2words/); [GitHub sutariyaraj/indic-num2words](https://github.com/sutariyaraj/indic-num2words/)
- `nemo_text_processing` 1.2.0 (requires pynini 2.1.6.post1) ships Hindi TN and ITN. Contents inspected locally:
  - ITN taggers: cardinal, date, decimal, fraction, measure, money, ordinal, telephone, time, whitelist.
  - The magnitude table lists हज़ार, लाख, करोड़, अरब, खरब, नील, पद्म, शंख.
  - `teens_and_ties.tsv` has 220 spellings for the 90 values 10–99, up to 7 variants per value (e.g., 98: निन्यान्बे / निन्यानबे / निन्यानवे / निन्यान्वे).
  - The cardinal grammar handles साढ़े, सवा, पौने (via `paune.tsv`), डेढ़ and ढाई.
  - `currency.tsv` maps रुपया/रुपये/रुपए/रुपिया/रुपेया and पैसा/पैसे to ₹/p.
  - ITN output uses Devanagari digits (१, २…).
  — [PyPI nemo-text-processing 1.2.0](https://pypi.org/project/nemo-text-processing/1.2.0/); [GitHub NVIDIA/NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing)
- NeMo's own grammar lists डेढ़ twice: once with precomposed ढ़ (U+095D) and once with ढ + nukta (U+0922 U+093C). Python's `unicodedata.normalize('NFC', 'ढ़')` returns the decomposed pair, because these letters are composition exclusions. NFC on both sides therefore unifies them. Verified locally. — [nemo_text_processing 1.2.0](https://pypi.org/project/nemo-text-processing/1.2.0/) (`inverse_text_normalization/hi/taggers/cardinal.py` line 145)
- Sarvam's Bulbul TTS docs: `enable_preprocessing` normalises numbers, dates and English words ("Rs. 1,00,000" → "rupees one lakh"). It defaults to false for bulbul:v2 and is "not available for bulbul:v3". Third-party docs (Pipecat) say bulbul:v2 is now deprecated and rejected by the API, so the source is contradictory. — [Sarvam how-to](https://docs.sarvam.ai/api/api-guides-tutorials/text-to-speech/how-to/enable-text-preprocessing); [Pipecat docs](https://reference-server.pipecat.ai/en/stable/api/pipecat.services.sarvam.tts.html); [LiveKit Sarvam plugin](https://docs.livekit.io/agents/models/tts/plugins/sarvam/)
- A third-party Indic ASR benchmark notes that IndicVoices conversational audio has many spoken phone numbers and OTPs, and that models outputting digits rather than spelled-out words are penalised against the references. — [sthanika.ai benchmark](https://sthanika.ai/research/what-it-actually-said)

### Inferences
- **Checker design.** I prototyped a parser locally (~70 lines, NeMo tables plus small dictionaries). It parsed all 12 test phrases correctly: पच्चीस हज़ार → 25000; ढाई लाख → 250000; साढ़े तीन करोड़ → 35000000; पौने दो लाख → 175000; सवा पाँच सौ → 525; एक करोड़ बीस लाख पचास हज़ार → 12050000; नौ आठ सात छह पाँच चार तीन दो एक शून्य → DIGITS 9876543210; नाइन एट डबल सात फाइव → 98775; सत्तासी हज़ार उनहत्तर → 87069; "पाँच लाख नहीं नहीं पाँच करोड़" → two amounts (the repair logic must then pick the last one). Design points:
  - NFC-normalise everything and map Devanagari digits to ASCII.
  - Load the value table from NeMo `digit.tsv` + `teens_and_ties.tsv` (all variants).
  - Add English-in-Devanagari digit words (ज़ीरो/वन/टू/…/नाइन) and डबल/ट्रिपल.
  - Add the fraction prefixes साढ़े +0.5, सवा +0.25, पौने −0.25, and the specials डेढ़ 1.5 and ढाई 2.5.
  - Treat सौ/हज़ार/लाख/करोड़/अरब as multipliers.
  - Read a run of three or more bare digits as a digit string (phone/account), and anything else as an amount.

  Known hard cases for this design are below.
- **Hard cases.**
  - "दो तीन लाख" is an approximate range ("two–three lakh"), not 23 lakh. It must be flagged as a range.
  - "एक एक" can mean digit 11 or "one each".
  - Years spoken in pairs (उन्नीस सौ सत्तासी, दो हज़ार चौबीस).
  - Times: साढ़े चार बजे is 4:30, not 4.5.
  - Ordinals and dates: पंद्रह तारीख़, पहली.
  - Ambiguous homographs: नौ (nine / "new" in Hindi-Urdu usage is नया, so low risk), सौ.
  - Code-mixed amounts: "फाइव लाख", "टेन के" (10k), "वन पॉइंट फाइव करोड़".
- **English-side checker.** Parse English digits, words ("five crore"), and "k/lakh/crore/million/billion" to a canonical integer, e.g. with `num2words` en_IN for generation and a small regex parser for "25k"/"2.5 lakh". Compare values, not strings, because the PS judges equivalent forms as correct.
- **Normalisation mismatch prevention.** Generate both the TTS input text and the Level-1 transcript from the same spoken-form string, with numbers already as words. Never send digits to TTS with `enable_preprocessing`, because then the TTS decides the verbalisation (e.g. "rupees one lakh" vs "एक लाख रुपये") and the transcript may not match the audio.

### Gaps
- Whether Sarvam's evaluator treats "5 crore" and "50 million" as equivalent. The PS says amounts keep the speaker's units, and the FAQ only gives 25,000/25k.
- No published accuracy figures for NeMo Hindi ITN on conversational or code-mixed text were found.

## Q4. TTS / audio validation: ASR round-trip, quality, duration and speaker filters, and which Hindi ASR to use

### Takeaway
Published filters and thresholds:
- IndicVoices-R: SNR ≥ 25 dB, C50 ≥ 30 dB, duration 0.2–30 s, pitch mean ≤ 350 Hz, pitch SD ≤ 150 Hz, speaking rate ≤ 30 chars/s, CER ≤ 5%.
- Emilia: DNSMOS P.835 OVRL > 3.0, Whisper LID confidence ≥ 80%, IQR outlier removal (1.5×IQR) on average character duration.
- Seed-TTS / F5-TTS evaluation convention: Whisper-large-v3 WER plus WavLM-large speaker-verification cosine.

For Hindi the round-trip ASR itself has about 10–13% WER and about 4–5% CER on benchmarks. A plain WER threshold therefore over-rejects, so use CER plus exact entity-span matching with two ASRs.

### Cited Findings
- IndicVoices-R thresholds: C50 ≥ 30 dB, SNR ≥ 25 dB (both from Brouhaha), duration 0.2–30 s, pitch mean ≤ 350 Hz and pitch SD ≤ 150 Hz (PENN via Dataspeech), speaking rate ≤ 30 characters/second. Utterances with CER > 5% "between the two levels" are excluded. The extract does not make clear which two transcripts are compared. The HF card lists clarity/SNR/pitch/rate filters but not C50 or CER. — [NeurIPS 2024 D&B paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/7dfcaf4512bbf2a807a783b90afb6c09-Paper-Datasets_and_Benchmarks_Track.pdf); [arXiv 2409.05356](https://arxiv.org/html/2409.05356); [HF card](https://huggingface.co/datasets/ai4bharat/indicvoices_r)
- Emilia-Pipe thresholds:
  - Drops segments whose Whisper LID confidence is below 80% or whose language is outside the target set.
  - Keeps DNSMOS P.835 OVRL > 3.0, justified as the P.835 "fair" anchor.
  - Removes average character (or, in one passage, "phone") duration outliers outside Q1 − 1.5·IQR / Q3 + 1.5·IQR, computed per source.
  - On a ~600 h subset it kept 29.43% (176.22 h) and raised mean DNSMOS from 2.50 to 3.26.
  - A later version used a DNSMOS threshold of 2.4 for incremental data.
  — [arXiv 2407.05361](https://arxiv.org/pdf/2407.05361); [arXiv 2501.15907](https://arxiv.org/html/2501.15907v1)
- Seed-TTS-eval convention: WER from Whisper-large-v3 (English) and Paraformer-zh (Chinese); SIM is the cosine between WavLM-large speaker-verification embeddings of the generated and reference audio. On Seed-TTS test-en, ground-truth audio scores WER 2.06 and SIM 0.73 (F5-TTS paper). Some later papers use Whisper-large-v3-turbo or ERes2Net, so numbers are comparable only when the same judge models are used. — [Seed-TTS arXiv 2406.02430](https://arxiv.org/html/2406.02430v1); [F5-TTS arXiv 2410.06885](https://arxiv.org/html/2410.06885v1); [Qwen-Audio-3.0-TTS arXiv 2607.23938](https://arxiv.org/pdf/2607.23938)
- NVIDIA NeMo Curator's generic guide suggests UTMOS ≥ 4.0 for TTS-grade data. This is not Hindi-validated. — [NeMo Curator UTMOS filter](https://docs.nvidia.com/nemo/curator/curate-audio/process-data/quality-filtering/utmos)
- Hindi ASR reference points:
  - IndicWhisper (Vistaar, 2023): Hindi WER 10.3 on Kathbath, 12.0 on Kathbath-Hard, 15.0 on FLEURS; lowest WER on 6 of 7 Hindi benchmarks at release. — [GitHub AI4Bharat/vistaar](https://github.com/AI4Bharat/vistaar); [arXiv 2305.15386](https://arxiv.org/pdf/2305.15386)
  - Sarvam Saarika v2.5 (vendor-reported, Vistaar): Hindi WER 11.81%, CER 4.42%. The docs say v2.5 will be deprecated. The blog gives about 22% WER on IndicVoices across all languages, and about 19% for Saaras v3. — [Sarvam Saarika docs](https://docs.sarvam.ai/api/getting-started/models/saarika); [Sarvam ASR blog](https://www.sarvam.ai/blogs/asr)
  - IndicConformer-600M multilingual (MIT, 22 languages, CTC/RNNT decoding): the model card reports Hindi WER 13.2 on Vaani-Benchmark-V1.0. An unofficial mirror reports dropped words on continuous audio over about 18 s, mitigated with ~15 s windows. — [HF ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual); [DictionLabs mirror](https://huggingface.co/DictionLabs/indic-conformer-600m-multilingual)
  - Logatra independent Hindi benchmark: Saarika v2.5 WER 0.373 on the public split and 0.197 on the held-out split. — [GitHub Nemukai/logatra-benchmark](https://github.com/Nemukai/logatra-benchmark)
  - Whisper large-v3 on code-mixed Hindi: Oriserve (a vendor) reports 50.8 WER on FLEURS for Hinglish output. In a Common Voice English study, Whisper-large-v3's hallucinations on Indian-accent audio were 43% repetition loops, 48% syntactic completions and 9.3% content hallucinations. — [Oriserve GitHub](https://github.com/OriserveAI/Whisper-Hindi2Hinglish); [arXiv 2502.12414](https://arxiv.org/pdf/2502.12414)
- The TTS-STT Flywheel used per-class CER filtering and rewrote digits as spelled words so that labels match the TTS audio. That is the closest published analogue to an entity-specific round trip. — [arXiv 2605.03073](https://arxiv.org/abs/2605.03073)
- No source gave Hindi numbers for Meta Omnilingual ASR. — (search returned no results)

### Inferences
Recommended TTS gate. Each clip must pass all checks; regenerate up to N=3 times with another seed or voice, then drop.

1. **Round-trip ASR with two Hindi ASRs** (IndicConformer-600M RNNT, plus Saarika/Saaras or Whisper-large-v3-Vaani-Hindi). Normalise both hypothesis and reference: NFC, strip punctuation, nukta folding, chandrabindu/anusvara folding, English-in-Devanagari spelling variants.
   - Keep if CER ≤ 8–10% on the best of the two ASRs. This is my threshold, set about 2× the ~4.4% CER the ASRs achieve on real Hindi. The IndicVoices-R 5% bar was for human-transcribed real speech and is too tight with ASR noise.
   - Review WER > 25% or CER > 15%.
2. **Entity exact match (zero tolerance).** Run the Q3 parser on the ASR output. Every gold entity value (amount, digit string, date) must appear exactly in at least one ASR hypothesis, and no forbidden or abandoned value may appear in a "corrected" position.
   - Names are matched by phonetic or transliteration distance, e.g. normalised edit distance ≤ 0.2 on a romanised form, because ASR spelling of names varies.
   - An entity-mismatched clip is regenerated, not relabelled, unless the Flywheel-style option of relabelling to what was actually said is used. Relabelling changes the gold entity and the English target, so it must be propagated.
3. **Skip/repeat detection.**
   - Characters per second within the IQR fence per voice (Emilia rule), and a hard cap of ≤ 30 chars/s (IndicVoices-R).
   - Insertion count from the alignment > 2 words, or an n-gram repeated more than 2 times in the ASR output where the script has none, means a repeat.
   - Forced-alignment confidence (CTC/MMS aligner) per word: flag words whose mean log-prob falls in the bottom 1–2%.
4. **Quality.** DNSMOS OVRL ≥ 3.0 on the clean TTS output, before telephony/noise augmentation is applied. Clipping: flag if more than 0.1% of samples are at full scale. Leading/trailing silence ≤ 1 s, and no internal silence > 2 s unless scripted (VAD).
5. **Speaker consistency.** WavLM-SV cosine to the prompt voice. No published threshold exists; calibrate on real same-speaker pairs and drop clips below the 5th percentile.
6. **LID for code-mixed clips.** Do not use Whisper LID ≥ 80% (Emilia) as a hard filter, because Hinglish is legitimately mixed. Instead check that the English words in the script come back as Devanagari English words from the ASR.

Human audit: listen to 1–2% of accepted clips, stratified by voice and entity type. Report precision and recall of the automatic gate on that audit.

### Gaps
- No source gives Hindi-validated DNSMOS/UTMOS thresholds or speaker-similarity cut-offs. UTMOS and DNSMOS are trained mostly on non-Indic data.
- No published Hindi WER for Omnilingual ASR or for Saaras v3 Hindi specifically was found.
- No source quantifies how often Indic TTS systems (Bulbul v3, IndicF5, Indic Parler-TTS) mispronounce numbers or names. This must be measured.

## Q5. Translation validation (QE, back-translation, LLM-judge, entity/number/negation consistency)

### Takeaway
Neural MT metrics (COMET/xCOMET/COMETKiwi) are used as quality-estimation filters without a standard threshold. Challenge-set work (ACES) shows that number and named-entity errors are a dedicated weakness category. Use deterministic Hindi↔English entity and negation consistency checks as the hard gate, with COMETKiwi or xCOMET and an LLM judge as soft signals.

### Cited Findings
- COMETKiwi (Unbabel/wmt22-cometkiwi-da) is used as a reference-free filter for synthetic parallel data. One study adds a minimum score-difference threshold because many pairs differ only by metric noise. No universal cut-off is reported. — [arXiv 2508.14951](https://arxiv.org/pdf/2508.14951); [COMETKiwi 2023 arXiv 2309.11925](https://arxiv.org/pdf/2309.11925)
- xCOMET combines a sentence score with error-span detection, deriving the sentence score from the predicted spans. In xCOMET analyses, localised errors caused large score drops, the largest for negation and for number/named-entity mismatches. This comes from a search extract of the xCOMET TACL paper, so check it. — [xCOMET TACL 2024](https://aclanthology.org/2024.tacl-1.54.pdf); [arXiv 2310.10482](https://arxiv.org/pdf/2310.10482)
- The ACES challenge set has 36,476 examples across 146 language pairs and 68 phenomena. It has a dedicated number and named-entity category, with character-level and word-level edits. Its hardest level uses a good translation that is lexically far from the reference, to test whether metrics still catch the entity error. — [ACES arXiv 2401.16313](https://arxiv.org/pdf/2401.16313); [GitHub EdinburghNLP/ACES](https://github.com/EdinburghNLP/ACES)
- GEMBA-MQM (GPT-4, 3-shot, MQM error spans) scored 96.5% system-level pairwise accuracy at WMT23. The authors caution about its black-box dependence. No per-category (entity/number) detection rate is reported. — [ACL Anthology 2023.wmt-1.64](https://aclanthology.org/2023.wmt-1.64)
- Sarvam evaluates ASR with LLM-WER, LLM-CER, an Intent Score and an Entity Preservation score (names, numbers, locations, organisations retained). It argues that identical WER can hide very different semantic accuracy. — [Sarvam blog](https://www.sarvam.ai/blogs/evaluating-indian-language-asr); [Sarvam Audio](https://www.sarvam.ai/blogs/sarvam-audio)
- Negation is a disproportionate MT error source: up to more than 60% quality reduction (Hossain et al.), and deleted negation is especially hard (Tang & Rönchen). — [arXiv 2010.05432](https://arxiv.org/pdf/2010.05432); [arXiv 2107.12203](https://arxiv.org/pdf/2107.12203)

### Inferences
Recommended translation gate. Hard checks must pass; soft checks route to review.

**Hard checks**
- **Entity multiset equality.** Compare the Hindi gold record (or the Q3 parser on the Level-1 text) with the English-side parser output, by canonical value:
  - amounts in rupees as integers;
  - digit strings exactly;
  - dates as (day, month, year);
  - names and orgs by transliteration match against the gold Latin form;
  - lakh/crore units preserved in the surface form (PS rule).
- **Negation polarity.** Count of Hindi negators (नहीं/मत/ना/न/कभी नहीं/बिना) vs English negators (not/n't/no/never/don't/without). Apply a mismatch rule: a mismatch triggers an LLM yes/no check for "is the polarity of each clause preserved?"
- **Conditional markers.** अगर/तो/वरना/जब तक ↔ if/then/otherwise/unless/until must be present on both sides.
- **Self-repair.** The English contains the corrected value and does not contain the abandoned value as an assertion. Allowed: "five lakh, no, five crore" in a literal variant. Not allowed: a fluent variant stating only "five lakh".

**Soft checks**
- COMETKiwi/xCOMET-QE: route the bottom 5–10% of a batch to review rather than applying an absolute threshold, since none is established.
- Back-translation (EN→HI, IndicTrans2) followed by the entity checker again. This catches English that is fluent but has drifted.

**Disfluency handling.** If the target is literal, the LLM translator must not "fix" disfluencies silently: an English filler or repair should exist when the Hindi has a content repair. If the target is fluent, it must collapse to the corrected meaning. Keep both as `en_alt` (the PS allows several translations).

### Gaps
- No published COMETKiwi/xCOMET threshold for Hindi→English filtering, and no meta-evaluation of how well QE metrics detect lakh/crore or digit-string errors in Hindi→English specifically.
- No source measured LLM-judge accuracy for negation-polarity checking in Hindi.

## Q6. Failure modes in similar datasets, and what passes the audit but trains a worse model

### Takeaway
The documented failures are:
- TTS repetition, skips and hallucination;
- labels that do not match the spoken verbalisation;
- low speaker and lexical diversity;
- synthetic data that does not transfer to real speech;
- automatic metrics blind to entity and negation errors;
- Hindi-specific underperformance of synthetic-entity methods.

An audit that checks transcript format and timings can pass while the model still fails on real, noisy, conversational Hindi.

### Cited Findings
- **Single-speaker TTS.** Even at higher quality it gave much worse ASR than multi-speaker TTS, so diversity beats polish. — [arXiv 1909.11699](https://arxiv.org/pdf/1909.11699)
- **Too little real data.** LLM+TTS conversational data beat older synthesis but did not match real in-domain data (Cornell et al.). — [arXiv 2408.09215](https://arxiv.org/abs/2408.09215)
- **Hindi transfer.** The entity flywheel failed its Hindi target (EHR 0.337 vs 0.65 target), and its real-speech validation was only n=20. — [arXiv 2605.03073](https://arxiv.org/abs/2605.03073)
- **Labels vs verbalisation.** The flywheel had to rewrite labels to spelled digits to match the TTS. ASR benchmarks penalise digit output against word references in IndicVoices. — [arXiv 2605.03073](https://arxiv.org/abs/2605.03073); [sthanika.ai](https://sthanika.ai/research/what-it-actually-said)
- **ASR used as a judge.** Whisper-large-v3 hallucinations are mostly repetition loops and syntactic completions, so it can hide or invent TTS errors in a round trip. — [arXiv 2502.12414](https://arxiv.org/pdf/2502.12414)
- **Long inputs.** Dropped words on long continuous audio were reported for IndicConformer beyond about 18 s (unofficial). — [DictionLabs mirror](https://huggingface.co/DictionLabs/indic-conformer-600m-multilingual)
- **Entity errors in speech translation.** In NEuRoparl-ST (Gaido et al., EMNLP 2021), ST systems got about 3/4 of terms and about 2/3 of NEs right, but only 37–40% of person names. NE accuracy correlates only loosely with overall translation quality (TACL "Hearing to Translate"). — [ACL Anthology 2021.emnlp-main.128](https://aclanthology.org/2021.emnlp-main.128/); [TACL](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.787/138660/Hearing-to-Translate-The-Effectiveness-of-Speech)
- **Entity injection helps names.** Detection-and-injection gave up to 7.1% higher NE accuracy and a 31.3% reduction in person-name errors. — [arXiv 2210.11981](https://arxiv.org/pdf/2210.11981)
- **Over-filtering.** Emilia's filters discarded about 70% of raw audio on its subset. Aggressive filtering shrinks data a lot. — [arXiv 2407.05361](https://arxiv.org/pdf/2407.05361)

### Inferences
Ways the dataset could pass the audit but train a worse model:

1. **Entity templates overfit.** The same 50 names, round amounts (5 lakh, 10 lakh) and fixed carrier phrases ("मेरा नंबर है…") teach the model to copy priors, not to listen. Sample entities from long-tailed distributions, cap reuse (e.g., each name ≤ 5 uses), and hold out entity values and templates between train and dev.
2. **Too clean.** Read-style TTS at DNSMOS > 3 without telephony/noise/codec simulation does not match the hidden test of "real Hindi recordings". Filter the clean render, then augment.
3. **Synthetic entity distribution mismatch.** Over-representing phone numbers and under-representing dates, products or orgs.
4. **References too literal or too free.** Literal targets that keep "no no" repairs could teach the streaming model to voice abandoned values. Free targets could drop hedges and negation scope.
5. **Speaker leakage.** Cloned voices of dev or test speakers in train inflate dev scores.
6. **Number-form inconsistency in English targets.** Mixed "5 lakh" / "500,000" / "half a million" teaches the model unstable outputs. Pick one canonical form for training, keeping lakh/crore per the PS, even though evaluation accepts equivalents.
7. **Round-trip ASR shares biases with the TTS.** If both were trained on similar Indic corpora, they can agree on a mispronunciation. Use two ASRs and a human audit.
8. **Translation LLM "fixes" disfluency or invents entities.** The entity and negation gates in Q5 catch this.

### Gaps
- No published study measures how much these specific failures (template overfitting, literal vs fluent targets) hurt a streaming S2ST model's entity or intent accuracy. Each needs an ablation.

## Q7. How entity and intent accuracy are likely measured, so data choices target the metric

### Takeaway
Existing practice:
- NE accuracy (case-insensitive) and NER-F1 for ST (NEuRoparl-ST);
- SLU-F1 for entities (SLURP);
- an Entity-Hit-Rate (Flywheel);
- named-entity error rate (IndicContextEval);
- Sarvam's own LLM-based Intent Score and Entity Preservation score.

The PS evaluator likely applies something similar to an ASR transcript of the English speech. This is an inference: the PS says only that KPIs are "measured on your system's English speech".

### Cited Findings
- NEuRoparl-ST scores NE handling by case-insensitive NE accuracy, i.e. whether the reference entity appears in the output, plus F1 for joint ST+NER. Some papers use case-sensitive accuracy, so check the variant before comparing. — [Interspeech 2023 Gaido et al.](https://www.isca-archive.org/interspeech_2023/gaido23_interspeech.pdf); [arXiv 2210.11987](https://arxiv.org/pdf/2210.11987); [arXiv 2210.11981](https://arxiv.org/pdf/2210.11981)
- SLURP introduced a dedicated entity-labelling metric (SLU-F1) for spoken entity evaluation. — [SLURP](https://astro.paperswithcode.com/paper/slurp-a-spoken-language-understanding-1)
- Sarvam uses an "Intent Score" and an "Entity Preservation" score alongside WER, LLM-WER and COMET. Entity preservation checks whether names, numbers, locations and organisations are retained. — [Sarvam STT page](https://www.sarvam.ai/speech-to-text); [Sarvam blog](https://www.sarvam.ai/blogs/evaluating-indian-language-asr)
- IndicContextEval reports a named-entity error rate alongside WER, with Indic NLP Library normalisation. — [arXiv 2606.19157](https://arxiv.org/pdf/2606.19157)
- PS (Sarvam.pdf §7, FAQ): intent covers asked/refused/confirmed/corrected, negations, conditions and who-does-what-to-whom. Entity covers names, places, orgs, amounts, numbers, dates and products. Equivalent number forms are accepted. Speaking an abandoned self-corrected value as fact is wrong. — local file /home/user/interiit_minips/Sarvam.pdf

### Inferences
- **Internal metric (Entity).** Run Whisper or another English ASR on the model's English speech, then the English number/date parser, then compare canonical values with the gold record. Report per-type exact-match accuracy, plus a "forbidden-value rate" for self-corrections.
- **Internal metric (Intent).** An LLM judge with a fixed rubric: speech act (ask/refuse/confirm/correct), polarity per clause, condition present, agent/patient roles. Calibrate it on about 100 human-labelled pairs.
- **Dev set.** Build from real recordings (Part D style), not TTS, so the internal metric matches the real hidden test.
- **Speech-act coverage.** Data should carry speech-act labels (asked/refused/confirmed/corrected) in the script generator, so coverage can be balanced. Refusals and confirmations are probably rare in found data.

### Gaps
- The exact Sarvam KPI implementation (ASR model on English output, LLM judge, matching rules for names and units) is not public in the PS or in any source found.

## Verification log: a claim that turned out wrong or misleading, and a question no source answers

### Takeaway
Several claims did not hold when checked:
- "num2words supports Hindi (lang='hi')" is false for the current release.
- The LARD arXiv ID given in the brief (2211.11863) is wrong; LARD is 2201.05041.
- Speech-MASSIVE is crowd-sourced real speech, not a synthetic or TTS entity dataset.

The open question: what entity-level round-trip pass rate on TTS data actually predicts entity accuracy on real Hindi test speech.

### Cited Findings
- **num2words.** It is commonly suggested (including by AI assistants) for Hindi number verbalisation. Version 0.5.14 raises `NotImplementedError` for `lang='hi'` and has no `hi` in `CONVERTER_CLASSES`, although it has bn/kn/te and en_IN. Tested locally on 2026-10-14. — [PyPI num2words](https://pypi.org/project/num2words/)
- **indic-num2words.** It is often recommended as the Hindi alternative, but it produced non-standard spellings (उन्तालिस, निन्यान्बे, अठ्हत्तर), inserted commas, and gave a digit-pair reading for numbers ≥ 100 crore. Tested locally. — [PyPI indic-num2words](https://pypi.org/project/indic-num2words/)
- **LARD ID.** LARD's arXiv ID is 2201.05041 (LREC 2022). 2211.09235 is a different paper ("Artificial Disfluency Detection, Uh No, Disfluency Generation for the Masses"). — [arXiv 2201.05041](https://arxiv.org/abs/2201.05041); [arXiv 2211.09235](https://arxiv.org/html/2211.09235)
- **Speech-MASSIVE.** The brief listed it under synthetic entity data, but it was "curated through crowd-sourcing", i.e. real speech. — [arXiv 2408.03900](https://arxiv.org/abs/2408.03900v1)
- **Emilia unit.** The duration-outlier rule is stated per "average phone duration" in one sentence and "average character duration" elsewhere, so the unit is ambiguous. — [arXiv 2407.05361](https://arxiv.org/pdf/2407.05361)
- **Bulbul preprocessing.** Sarvam docs describe `enable_preprocessing` for bulbul:v2, while Pipecat docs say bulbul:v2 is rejected by the API. Advice to "just turn on enable_preprocessing" may be stale. — [Sarvam how-to](https://docs.sarvam.ai/api/api-guides-tutorials/text-to-speech/how-to/enable-text-preprocessing); [Pipecat](https://reference-server.pipecat.ai/en/stable/api/pipecat.services.sarvam.tts.html)

### Inferences
- Recommended "wrong claim" for the report: the num2words Hindi claim. It is concrete, reproducible in two lines of Python, and directly relevant to the Part D checker.
- Recommended "no source answers" question: **at what entity-exact-match round-trip threshold, and what fraction of entity-dense TTS hours in a 200 h mix, does real-speech entity accuracy of a streaming Hindi→English S2ST model stop improving or start degrading?** The Flywheel paper hints at poor Hindi transfer but measured ASR, not S2ST, with n=20 real utterances in Telugu only. Test it in week 1 with a small sweep: 0/5/10/20% entity-TTS hours, scored on 30–50 real recorded entity clips.

### Gaps
- Could not read full PDFs (arXiv, HF and ACL were blocked in this environment). Exact numbers from IndicVoices-R, Emilia, xCOMET and GCM were taken from search-tool extracts of the primary pages and should be spot-checked against the PDFs before final citation.
