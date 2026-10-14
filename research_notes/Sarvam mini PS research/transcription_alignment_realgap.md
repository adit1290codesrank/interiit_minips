# Verbatim Hindi transcription, Hindi forced alignment, code-mixed Hindi→English translation, and the synthetic-vs-real speech gap

Scope: Sarvam.ai PS (Inter IIT Tech Meet 15.0), Part A record requirements (from `/home/user/interiit_minips/Sarvam.pdf`, read with `pdftotext -layout`). Each record needs: a verbatim Hindi transcript following IndicVoices Level 1 ("transcribed as spoken": disfluencies, repetitions and fillers kept; English words written in Devanagari as spoken, with no bracketed Latin forms; numbers as spoken words; sentence-final punctuation; noise tags optional; manual correction allowed). It also needs start and end times for every Hindi word, including the English words written in Devanagari ("audited at the mid-term"), and sentence-level English translations ("numbers in words, digits also accepted"; "amounts keep the speaker's units (lakh, crore)"; several translations allowed per sentence). The PS also says: "If you use TTS-generated speech anywhere, check it and filter or regenerate bad output… TTS can mispronounce names and numbers, skip or repeat words, or add artefacts." The hidden test set is real Hindi recordings that may contain "noise and telephone audio", and the audio is 24 kHz mono 16-bit.

Method note (research environment): direct fetches of arxiv.org, huggingface.co, docs.sarvam.ai, pytorch.org and readthedocs were blocked by the egress proxy. So the facts below come from two places. Most were checked first-hand in GitHub READMEs and source files (raw.githubusercontent.com). The rest come from web-search result extracts of the primary pages; these are marked "(search extract)" and should be re-checked before they are quoted as exact numbers in the report.

---

## 1. IndicVoices Level 1 / Level 2 transcription: definitions, tool, process, noise tags, code-mixing

### Takeaway
IndicVoices uses exactly two transcription levels. Level 1 is verbatim and phonetically faithful: "transcribe as spoken" and do not worry about spelling. Level 2 standardises that text to the nearest textbook spelling. No higher levels exist. Transcription was done in AI4Bharat's open-source Shoonya tool, with a maker → checker → super-checker workflow, and 89 bracketed non-speech tags such as [barking] and [hmm]. The PS's rule "English in Devanagari, no bracketed Latin" is stricter than some IndicVoices-derived conventions, which keep English named entities in brackets. The team must apply it themselves.

### Cited Findings
- IndicVoices (arXiv 2403.01926; ACL Findings 2024) describes a two-level strategy. "In Level 1, transcription is performed exactly as spoken, while Level 2 involves standardizing words to their textbook form." Level 1 is described as easier to write rules for, because transcribers "just need to listen to the sound and transcribe it using the corresponding grapheme". Level 2 needed detailed standardisation rules written with linguists — [IndicVoices arXiv PDF](https://arxiv.org/pdf/2403.01926) (search extract)
- Appendix A wording: in the first level "the transcribers were asked to transcribe verbatim without worrying about correctness of spellings (thus emphasising only on phonetic fidelity)". In the second level "the transcribers were asked to standardise the transcription to convert the phonetically correct representations to nearest standard spellings in textbooks" — [IndicVoices arXiv HTML v1](https://arxiv.org/html/2403.01926v1) (search extract)
- Non-speech events use bracketed tags (e.g., [baby_crying], [barking], [hmm], [horn]). There are 89 tags, listed in Table 13 of Appendix C. Guidelines were written separately for each of the 22 languages, with examples and several review rounds — [IndicVoices arXiv HTML v1](https://arxiv.org/html/2403.01926v1) (search extract)
- Quality control uses a maker–checker–superchecker process with two review rounds. IndicVoices-R (the TTS derivative) excluded utterances where the CER between the Level-1 and Level-2 transcripts was above 5% — [IndicVoices PDF](https://arxiv.org/pdf/2403.01926); [IndicVoices-R, NeurIPS 2024 D&B](https://proceedings.neurips.cc/paper_files/paper/2024/file/7dfcaf4512bbf2a807a783b90afb6c09-Paper-Datasets_and_Benchmarks_Track.pdf) (search extract)
- Shoonya is AI4Bharat's open-source annotation platform for parallel text, OCR, ASR and TTS data. The IndicVoices repo names it as the transcription platform, and Kathbath (built on Microsoft's Karya) as the collection app — [IndicVoices README (GitHub)](https://github.com/AI4Bharat/IndicVoices) (read directly). In Shoonya, an item is annotated, then reviewed, then super-checked, and the tool can pre-fill transcripts with IndicASR — [AI4Bharat Shoonya blog](https://ai4bharat.iitm.ac.in/blog/shoonya) (search extract)
- Dataset size per the current README: 12,000 h in total (8% read, 76% extempore, 15% conversational), from 22,563 speakers in 208 districts and 22 languages. Of this, 3,200 h are transcribed, with a median of 122 h per language. (The arXiv v1 abstract says 7,348 h and 16,237 speakers; the README figures are the later release) — [IndicVoices README](https://github.com/AI4Bharat/IndicVoices) (read directly)
- Code-mixing conventions differ by dataset. A secondary IndicVoices-derived benchmark describes verbatim native-script transcription where English named entities are "transliterated into the native script and additionally provided in brackets in English" — (search extract via [IndicVoices search result](https://arxiv.org/pdf/2403.01926); secondary attribution, unverified). Indic DiarBench (2026) defines two forms. One is native-script, with English words written in the Indic script. The other is normalised, with English in Roman script and digits as Arabic numerals — [Indic DiarBench arXiv 2607.23808](https://arxiv.org/pdf/2607.23808) (search extract)

### Inferences
- The PS's "Level 1" is IndicVoices Level 1, with two extra rules added: no bracketed Latin forms, and numbers as words. A Level-2-style standardised spelling (e.g., correcting colloquial or phonetic spellings) is not required and arguably violates "as spoken". Example: if the speaker says "मतबल", keep "मतबल" and do not correct it to "मतलब".
- Any IndicVoices-derived training text, or ASR trained on it, may emit "[English]" brackets or Level-2 spellings. A post-filter that strips brackets and Latin script is needed.
- Noise tags are optional in the PS. If they are used, they must be excluded from word-alignment targets, or be aligned as non-word segments, so the auditor does not count them as "words".

### Gaps
- The full Appendix (Table 13 tag list and the per-language Hindi rules, e.g., how Level 1 writes English words and digits) could not be read directly (arXiv blocked). Whether IndicVoices Level 1 itself writes English words in Devanagari or in Latin brackets is not confirmed from the primary text.

---

## 2. Verbatim Hindi ASR options, reported Hindi WER, disfluency preservation, script and number handling

### Takeaway
No off-the-shelf model meets the PS's Level-1 format directly. The most usable starting points are:
- **Sarvam `saaras:v3` in `mode="verbatim"`**: vendor-documented to keep fillers and spoken numbers, but code-mix mode returns English in Latin script.
- **AI4Bharat IndicConformer-600M** (MIT, CTC+RNNT, trained on IndicVoices-style data): open, and also usable for alignment.
- **IndicWhisper**: lowest open Hindi WER on Vistaar, 13.6% average.

Whisper large-v3 and turbo tend to normalise away fillers and are weak on Hindi. A post-processing stage is unavoidable: Latin→Devanagari for English words, digits→Hindi number words, and sentence punctuation. Human correction is still needed for fillers, repetitions, false starts and English loanword spellings.

### Cited Findings
- **IndicWhisper / Vistaar** (arXiv 2305.15386). Hindi WER: Kathbath 10.3, Kathbath-Hard (noisy) 12.0, MUCS 12.0, IndicTTS 7.6, Gramvaani (telephone-quality community radio) 26.8, average 13.6. On the Hindi subset, the average baselines are Google STT 23.9, IndicWav2vec 21.0, Azure STT 20.0, NVIDIA-medium 19.4 and NVIDIA-large 18.6 — [Vistaar README](https://github.com/AI4Bharat/vistaar) (read directly). Gramvaani is the closest public proxy for telephone or low-bandwidth Hindi, and every system's WER there is 2–3× its clean WER (Google STT 59.9, IndicWav2vec 42.1, IndicWhisper 26.8) — same source
- **IndicConformer**: AI4Bharat suite for all 22 scheduled languages, released under the MIT licence. It ships as a 600M multilingual hybrid RNNT model (`indicconformer_stt_multi_hybrid_rnnt_600m.nemo`) plus per-language "hybrid_rnnt_large" models, including Hindi. It needs the AI4Bharat NeMo fork (`nemo-v2` branch) — [IndicConformerASR README](https://github.com/AI4Bharat/IndicConformerASR) (read directly). The HF card describes it as a hybrid CTC+RNNT Conformer with language-specific heads — [HF CoreML port](https://huggingface.co/phequals/indic-conformer-600m-multilingual-coreml-rnnt) (search extract). The IndicVoices repo also ships the "IndicASR" NeMo checkpoint trained on IndicVoices — [IndicVoices README](https://github.com/AI4Bharat/IndicVoices)
- No official Hindi WER for the 600M multilingual IndicConformer was retrievable. A community GGUF repack of the monolingual Hindi IndicConformer reports Kathbath 13.5 and FLEURS-hi 15.2 WER (self-reported) — [Singla0009/IndicConformer-GGUF](https://huggingface.co/Singla0009/IndicConformer-GGUF) (search extract)
- **Sarvam Saaras v3** (2026) offers five output modes: `transcribe`, `translate`, `verbatim`, `translit` and `codemix`. Verbatim is "exact word-for-word transcription without normalization, preserving filler words and spoken numbers as-is". `translit` gives Roman script only. `codemix` gives "English words in English and Indic words in native script". The mode parameter only works with `saaras:v3`; `saarika:v2.5` is legacy, transcription-only, and ignores mode. Since June 2026 the REST and batch STT endpoints default to `saaras:v3`, and streaming defaults to Saaras v4 — [Sarvam STT REST docs](https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/rest-api); [Saaras model page](https://docs.sarvam.ai/api-reference-docs/models/saaras); [Sarvam changelog](https://docs.sarvam.ai/api/getting-started/changelog) (search extracts)
- Sarvam reports Saaras v3 at about 19.31% WER on its 10-language IndicVoices subset, down from about 22% in v2. It claims to beat GPT-4o-transcribe, Gemini 3 Pro, Deepgram Nova-3 and ElevenLabs Scribe v2. This is vendor-reported, aggregated across 10 languages, and not a Hindi-only figure — [Sarvam ASR blog](https://www.sarvam.ai/blogs/asr); [Sarvam STT page](https://www.sarvam.ai/speech-to-text) (search extract)
- **Whisper large-v3**: the Trelis Hinglish card reports base large-v3 WER of 27.50 on FLEURS-hi, 30.82 on CommonVoice-hi, 33.92 on cs-FLEURS and 29.74 on CoSHE-500. These are the author's own numbers with a script-safe normaliser. The card warns that Whisper's default English normaliser strips Devanagari matras, which inflates WER. It also warns that Devanagari references unfairly penalise Latin-script loanwords and digits — [Trelis/whisper-hinglish-preview](https://huggingface.co/Trelis/whisper-hinglish-preview) (search extract). The Vaani benchmark reports whisper-large-v3 at 34.1 WER on its Hindi set — [Vaani Benchmark arXiv 2606.21408](https://arxiv.org/pdf/2606.21408) (search extract)
- **Whisper and disfluencies**:
  - Whisper "performs implicit disfluency removal", attributed to its training-data normalisation — [Kid-Whisper arXiv 2309.07927](https://arxiv.org/pdf/2309.07927) (search extract)
  - Off-the-shelf Whisper behaves "more in line with intended speech transcription" — [arXiv 2311.00867](https://arxiv.org/pdf/2311.00867) (search extract)
  - CrisperWhisper (verbatim fine-tune) reports AMI WER of 9.72 against Whisper's 16.82 on verbatim references — [CrisperWhisper arXiv 2408.16589](https://arxiv.org/pdf/2408.16589) (search extract)
  - Informal tests found Whisper drops ums and uhs but keeps most other disfluencies — [Cortico blog](https://cortico.ai/news/insights/evaluating-openai-s-whisper-on-community-conversations/) (non-peer-reviewed)
- **Controllable verbatim ASR**: "Transcription Policy as a Latent Variable" (Interspeech 2026) argues that mixing verbatim and intended-style annotations makes the transcription style an uncontrolled latent variable. This causes decoding instability, inflated WER and unreliable word timing. Task tokens trained on parallel verbatim/intended pairs raised German disfluency F1 from 10% to 79%. Supervised cross-attention fine-tuning gave better word timestamps on disfluent speech than forced-alignment baselines — [arXiv 2607.18934](https://arxiv.org/pdf/2607.18934) (search extract of abstract)
- **Whisper script confusion**: a Hindi telephony pilot found Faster-Whisper `base` returned Hindi speech in Urdu/Arabic script, while `small` returned Devanagari. This is a 2-utterance pilot, not a benchmark — [CallClear README](https://github.com/Siddharth-21s/callclear) (read directly)
- **Meta Omnilingual ASR** (Nov 2025, 1,600+ languages; CTC and LLM-ASR families from 300M to 7B, plus v2 and "Unlimited"-length variants). The 7B LLM-ASR per-language table lists `hin_Deva` with 236.7 training hours and a CER of 3.6. Note that this is CER, not WER. Inference accepts only audio shorter than 40 s, except in the Unlimited models. Languages are coded `{lang}_{script}` — [omnilingual-asr README](https://github.com/facebookresearch/omnilingual-asr); [per_language_results_table_7B_llm_asr.csv](https://github.com/facebookresearch/omnilingual-asr/blob/main/per_language_results_table_7B_llm_asr.csv) (read directly)
- **Meta MMS**: MMS-1B-all covers 1,162 languages. The code and weights are CC-BY-NC 4.0, which is non-commercial. Per-language token dictionaries are provided, e.g., Hindi for FL102 — [fairseq MMS README](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) (read directly). No Hindi WER figure was retrievable.
- **Qwen3-ASR** (0.6B/1.7B, Jan 2026) supports Hindi among 30 languages and is evaluated on FLEURS (hi in the "Fleurs†" group) and News-Multilingual. **Qwen3-ForcedAligner-0.6B does NOT support Hindi**: it covers 11 languages (zh, en, yue, fr, de, it, ja, ko, pt, ru, es). Streaming mode does not return timestamps — [Qwen3-ASR README](https://github.com/QwenLM/Qwen3-ASR) (read directly)
- **NVIDIA**: the NIM support matrix lists a Parakeet 1.1B RNNT Multilingual model and Nemotron ASR Streaming as supporting hi-IN — [NVIDIA ASR NIM support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html) (search extract). Canary-1B-v2 and Parakeet-TDT-0.6B-v3 cover 25 European languages; Hindi coverage was not confirmed — [arXiv 2509.14128](https://arxiv.org/html/2509.14128v1) (search extract). A NeMo Curator PR adds a Hindi "Indic Canary" FLEURS benchmark, with no WER given — [NeMo Curator PR #2456](https://github.com/NVIDIA-NeMo/Curator/pull/2456) (search extract)
- **Google Chirp 3**: the Vaani benchmark reports a substitution penalty on code-switched tokens of only 2.6 points for Chirp 3, against 15.6 for Shrutam — [Vaani Benchmark arXiv 2606.21408](https://arxiv.org/pdf/2606.21408) (search extract)
- **Transliteration (Latin English word → Devanagari)**: IndicXlit (about 11M params, transformer, trained on the 26M-pair Aksharantar corpus, 21 languages). Hindi En→Indic top-1 accuracy is 60.56 on Dakshina, 55.59 on Aksharantar native words and 58.87 on Aksharantar named entities — [IndicXlit README](https://github.com/AI4Bharat/IndicXlit) (read directly)

### Inferences
- **Recommended pipeline**:
  1. Get a first-pass hypothesis from Saaras v3 `verbatim`, or IndicConformer, or both. Two-system disagreement can flag segments for review.
  2. Convert any Latin tokens to Devanagari. Use an LLM with a lookup cache for English loanwords (e.g., "meeting" → मीटिंग). IndicXlit is a weaker fit, because it is trained on romanised *Indic* words, not English words, and its top-1 accuracy is only about 56–61%.
  3. Expand digits into Hindi number words, keeping "lakh/crore" grouping, e.g., 2,50,000 → "ढाई लाख" or "दो लाख पचास हज़ार", whichever was said. Rule-based libraries (e.g., `num2words` with `lang='hi'`) give a reading but not necessarily *the spoken* one, so this must be checked against the audio.
  4. Add danda or "?" at the end of each sentence.
- **Where humans are still needed**: restoring fillers (अं, हम्म, मतलब), repetitions and false starts (Whisper-family models delete them; even "verbatim" modes are unverified for Hindi); choosing the spoken form of numbers and dates; spelling English loanwords consistently; and sentence segmentation.
- Because the training references are verbatim with Devanagari English, WER of public models against these references will be inflated. Use a transliteration-aware CER when comparing ASR systems on the team's own dev set.

### Gaps
- No independent, Hindi-only, verbatim-reference comparison of Saaras v3, IndicConformer-600M, Omnilingual, Qwen3-ASR and Chirp 3 was found. All Sarvam numbers are vendor-reported.
- Whether Saaras v3 `verbatim` writes English words in Devanagari or in Latin for Hindi is not stated in the retrieved docs. Only `codemix` is documented, as Latin English. This needs a quick API test.
- The official IndicConformer-600M Hindi WER (HF model card) and the MMS-1B-all Hindi FLEURS WER could not be retrieved.

---

## 3. Hindi forced alignment: tools, reported boundary accuracy, failure modes, measuring error, human agreement

### Takeaway
Hindi has no MFA pretrained acoustic model, only a Hindi CV pronunciation dictionary. Qwen3-ForcedAligner excludes Hindi. The practical open options are CTC-based:
- MMS-based `ctc-forced-aligner`, with uroman romanisation and `<star>` tokens for unknown material;
- torchaudio MMS_FA, deprecated or removed in torchaudio ≥2.9, so pin 2.8;
- WhisperX's default Hindi wav2vec2 (`theainerd/Wav2Vec2-large-xlsr-hindi`);
- NeMo Forced Aligner with an IndicConformer CTC head.

On English benchmarks, CTC aligners (MMS, WhisperX) put far fewer word boundaries within 25 ms than MFA does: about 43–53% against 70–73%. Hand checks will typically find errors of tens of ms, and large failures around fillers, long silences and noise. Human labellers themselves disagree by about 10–20 ms, so 20 ms is the conventional tolerance.

### Cited Findings
- **MFA**: the mfa-models catalog has a "Hindi CV dictionary v2.0.0" (Epitran IPA phoneset, CC BY 4.0) but **no Hindi acoustic model**. The acoustic-model list includes Tamil as the only Indic language, alongside English, French, Mandarin and others. G2P models also include Tamil but not Hindi — [mfa-models docs source: dictionary/Hindi](https://github.com/MontrealCorpusTools/mfa-models/tree/main/docs/source/dictionary/Hindi); [acoustic index](https://github.com/MontrealCorpusTools/mfa-models/blob/main/docs/source/acoustic/index.rst) (read directly). MFA can train a new acoustic model with `mfa train` and generate dictionaries with `mfa g2p`, but the dictionary phone set must match the acoustic model — [MFA docs](https://montreal-forced-aligner.readthedocs.io/en/latest/reference/g2p/index.html); [Corpus Phonetics Tutorial arXiv 1811.05553](https://arxiv.org/pdf/1811.05553) (search extract)
- **ctc-forced-aligner (MahmoudAshraf97)**:
  - It uses HF CTC models; the default is `MahmoudAshraf/mms-300m-1130-forced-aligner`, covering 1,130 languages.
  - It needs `--romanize` (uroman) for non-Latin scripts with the default model.
  - It aligns at word, sentence or char level, with a `<star>` token at "edges" or per "segment", and chunks audio with a 30 s window and 2 s context.
  - It claims at least 5× less memory than the torchaudio API.
  - Native-vocabulary models can be swapped in via `--alignment_model`.
  — [ctc-forced-aligner README](https://github.com/MahmoudAshraf97/ctc-forced-aligner) (read directly)
- **torchaudio MMS_FA**: the tokenizer uses lowercase Latin letters, apostrophe and `<star>`. Transcripts must be romanised (uroman) and normalised first. Digits are not romanised by uroman and get stripped: the tutorial's "1882" is removed — [torchaudio multilingual FA tutorial source](https://github.com/pytorch/audio/blob/main/examples/tutorials/forced_alignment_for_multilingual_data_tutorial.py) (read directly). The torchaudio 2.9 docs state that the forced-alignment APIs "were deprecated in 2.8 and have been removed in 2.9", as TorchAudio entered maintenance. Docs for 2.11 still show the tutorial, so the docs are inconsistent — [torchaudio 2.9 CTC FA tutorial](https://docs.pytorch.org/audio/2.9.0/tutorials/ctc_forced_alignment_api_tutorial.html) (search extract)
- **WhisperX**: the default alignment model for `hi` is `theainerd/Wav2Vec2-large-xlsr-hindi`, from `DEFAULT_ALIGN_MODELS_HF` in alignment.py. The documented limitation: "Transcript words which do not contain characters in the alignment models dictionary e.g. '2014.' or '£13.60' cannot be aligned and therefore are not given a timing." Overlapping speech is handled poorly — [whisperX README](https://github.com/m-bain/whisperX); [whisperx/alignment.py](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py) (read directly)
- **NeMo Forced Aligner (NFA)**: produces token-, word- and segment-level timestamps from any NeMo **CTC** model (or a hybrid model's CTC head). It accepts a user reference text and handles audio of 1 h or more — [NFA README](https://github.com/NVIDIA/NeMo/tree/main/tools/nemo_forced_aligner) (read directly). Since IndicConformer is hybrid CTC+RNNT in NeMo format (§2), NFA can in principle align Devanagari text natively, without romanisation. This is an inference; no Hindi NFA evaluation was found.
- **Qwen3-ForcedAligner-0.6B**: non-autoregressive, aligns up to 5 min, 11 languages, **no Hindi**. Its reported accumulated average shift (AAS, ms, lower is better) is about 33–76 ms. Examples: English 37.5 (against NFA 107.5 and WhisperX 92.1), Chinese 33.1, French 41.7 and Italian 75.5. On the same table, NFA scores 100–143 ms and WhisperX 92–165 ms — [Qwen3-ASR README](https://github.com/QwenLM/Qwen3-ASR) (read directly)
- **English word-boundary accuracy** (TIMIT / Buckeye; % within 10/25/50/100 ms), as reported in a FALCON paper table drawing on Rousso et al. 2024:
  - TIMIT: MFA 41.6/72.8/89.4/97.4; WhisperX 22.4/52.7/82.4/94.2; MMS 18.6/43.5/75.7/94.7.
  - Buckeye: MFA 39.8/69.9/84.9/91.8; WhisperX 18.8/43.1/67.4/77.4; MMS 25.0/52.7/75.0/87.9.
  - Rousso et al. scored with F1 at a 20 ms threshold and excluded words off by more than 500 ms.
  — [Rousso et al., "A Comparison of Modern ASR Methods for Forced Alignment", arXiv 2406.19363](https://arxiv.org/pdf/2406.19363); [arXiv 2606.10675](https://arxiv.org/pdf/2606.10675) (search extract)
- **2026 MFA evaluation**: MFA 3.0 mean boundary error is below 15 ms on four benchmarks, scored on word starts rather than ends — [arXiv 2606.18466](https://arxiv.org/html/2606.18466v1) (search extract). A Whisper fine-tuned on human boundaries reports 11.6 ms TIMIT word-boundary error, against 18.2 ms for the best prior system, and MFA at 68.2% within 20 ms — [arXiv 2610.17782](https://arxiv.org/html/2610.17782) (search extract). Different protocols give different numbers, so always state the protocol.
- **Human agreement and tolerance**:
  - More than 10% of boundaries differed by over 20 ms between human labellers (Cosi et al. 1991, via Pauws et al. 1996).
  - The average inter-labeller deviation in Buckeye was about 16 ms (Pitt et al. 2005).
  - Experts are not expected to agree to the millisecond, so automatic annotation should aim at about 10–15 ms.
  - Automatic phone boundaries on the UCLA archive: about 71% within 20 ms of gold.
  — [UCLA Phonetics Lab Archive segmentation, arXiv 2403.19509](https://arxiv.org/pdf/2403.19509); [Interspeech 2009 Ogbureke](https://www.isca-archive.org/interspeech_2009/ogbureke09_interspeech.pdf) (search extracts)

### Inferences
- **Recommended Part D protocol**:
  1. Run the aligner and export word intervals.
  2. Open each clip in Audacity, import the aligner output as a label track (Audacity labels are tab-separated `start end label`), and hand-place 30 boundaries, e.g., 15 starts and 15 ends spread across fillers, English-in-Devanagari words, numbers and words after pauses.
  3. Export the hand labels the same way.
  4. Report the mean and median absolute error in ms, the percentage within 20, 50 and 100 ms, and the count of gross errors (more than 200 ms).
  Keep starts and ends separate, because CTC aligners are biased differently on each; CTC "peaky" outputs often place word ends late or early around blanks. Use the human-agreement floor of about 10–20 ms as context.
- **Expected Hindi failure modes**:
  - (a) Fillers and noise not in the transcript get absorbed into neighbouring words. Mitigate with `<star>` tokens, or by transcribing the fillers, which Level 1 requires anyway.
  - (b) uroman romanises Devanagari English loanwords phonetically (e.g., मीटिंग → "miting"). This is fine for MMS, but char-vocab Hindi wav2vec2 models need everything in Devanagari.
  - (c) Digits, Latin text or punctuation outside the vocabulary get no timing (the WhisperX limitation), or are silently dropped (MMS_FA normalisation). This is another reason to expand numbers to words *before* alignment.
  - (d) Long silences or 30 s chunk boundaries stretch the timing of adjacent words.
  - (e) Telephone and noisy audio degrade CTC posteriors.
  Since clips are ≤60 s and records need every word timed, use an aligner that never drops words: `ctc-forced-aligner`, or NFA with an IndicConformer CTC head. Upsample-matching matters too: resample 24 kHz to 16 kHz for these models.
- Use the disagreement between two aligners (e.g., MMS against IndicConformer/NFA) as an automatic QC signal. Flag words whose boundaries differ by more than 100 ms for human review.

### Gaps
- **Unanswered question for this task**: no source reports word-boundary accuracy of *any* aligner on Hindi, let alone on code-mixed Hindi with English words written in Devanagari, against hand labels. In particular, no source measures whether uroman romanisation of Devanagari-transliterated English words, or of disfluencies, degrades MMS boundaries relative to native-script CTC models (IndicConformer/NFA, theainerd wav2vec2). The team's own 30-boundary check will be the only evidence.
- Whether torchaudio ≥2.9 truly removed `forced_align` and `MMS_FA` is inconsistent across the docs. Pin torchaudio 2.8, or use ctc-forced-aligner, which ships its own implementation.
- No published word-level human inter-annotator agreement figure (as opposed to phone-level) was found.

---

## 4. Translating code-mixed, disfluent Hindi to English: IndicTrans2, Sarvam, NLLB and LLMs; known failure modes

### Takeaway
IndicTrans2 is the strongest open dedicated MT system for Hindi→English. It reportedly beats GPT-3.5 by about 12 points in Indic→En. However, it was benchmarked on clean, formal and conversational text (FLORES-200, IN22-Gen and IN22-Conv), not on disfluent, Devanagari-code-mixed ASR transcripts. Sarvam offers `sarvam-translate:v1`, which is formal and has `numerals_format`, and `mayura:v1`, which has formal, colloquial and code-mixed modes. LLMs handle disfluency and the "keep lakh/crore" rule best when explicitly prompted. The documented risk is numeric and unit errors: large-unit conversion error rates exceed 20% for some open LLMs, and BLEU barely penalises these errors.

### Cited Findings
- IndicTrans2 (arXiv 2305.16307; TMLR 12/2023) covers all 22 scheduled languages. Its primary benchmarks are IN22 (IN22-Gen = Wiki+Web; IN22-Conv = 1,503 conversational sentences) and FLORES-200 devtest (1,012 sentences). chrF++ is the primary metric, with BLEU and COMET also reported. The paper reports +5.8 (En-Indic) and +7.4 (Indic-En) points over IndicTrans1 on FLORES — [IndicTrans2 README](https://github.com/AI4Bharat/IndicTrans2) (read directly); [arXiv 2305.16307](https://arxiv.org/abs/2305.16307) (search extract)
- The NTREX Hindi→English BLEU for IndicTrans2 is 41.3. In the Indic→En direction, its gains over the best commercial and open baselines on IN22-Gen were small, about +0.7 to +1.1 (TMLR version) — [IndicTrans2 TMLR/arXiv](https://arxiv.org/pdf/2305.16307) (search extract). The exact hin_Deva→eng_Latn chrF++ on FLORES and IN22 could not be retrieved; the model card points to a Google Drive metrics folder — [HF indictrans2-indic-en-1B card](https://huggingface.co/ai4bharat/indictrans2-indic-en-1B) (search extract)
- For Hinglish→English (romanised code-mixed text, WMT22 MixMT), an IndicTrans-based system using transliteration and pseudo-translation reached ROUGE of about 0.55 — [WMT22 paper](https://aclanthology.org/2022.wmt-1.113.pdf) (search extract)
- **Sarvam translation APIs**:
  - `mayura:v1` modes: formal, modern-colloquial, classic-colloquial and code-mixed (`mode="code-mixed"`); 1,000-char input limit.
  - `sarvam-translate:v1`: formal style only, `numerals_format` ("international" or "native"), 2,000-char limit.
  — [Sarvam Mayura docs](https://docs.sarvam.ai/api-reference-docs/getting-started/models/mayura); [Sarvam-Translate docs](https://docs.sarvam.ai/api/getting-started/models/sarvam-translate.md) (search extracts)
- **Numeric and large-unit errors**: in a study of LLM number translation (zh↔en, with large units such as 亿/million/billion), "even the latest llama3.1 8b model can have error rates as high as 20%" on large-unit cases. Google Translate and Tower-Instruct-7B were more accurate. BLEU is "inadequate" because a wrong magnitude is often a single-token error — [Investigating Numerical Translation with LLMs, arXiv 2501.04927](https://arxiv.org/pdf/2501.04927) (search extract). This is the closest analogue to lakh/crore → million conversion; no lakh/crore-specific study was found.

### Inferences
- **Recommended approach**:
  1. Translate from a *cleaned* view of the Level-1 transcript: optionally strip fillers and repetitions before MT, but keep the original for the record.
  2. Use an LLM, or IndicTrans2 as a baseline, with explicit instructions: keep lakh and crore as "lakh/crore" (the PS requires the speaker's units); write numbers in words; keep negation and hedges; do not add content; preserve sentence boundaries one-to-one, so that sentence-level streaming targets align.
  3. Run automatic checks:
     - a number/unit consistency check, comparing extracted Hindi numerals against English numerals and keeping lakh/crore;
     - a negation check (नहीं/मत/न ↔ not/no/never);
     - a length-ratio check;
     - back-translation or round-trip COMET/chrF for hallucination screening.
- **Literal vs fluent**: for a streaming S2ST target, a more monotonic, literal translation is easier to learn with low latency. Hibiki (Kyutai) built its synthetic parallel data with per-word delays chosen using a text MT system's perplexity — [Hibiki arXiv 2502.03382](https://arxiv.org/html/2502.03382v1) (search extract). The PS allows several translations per sentence, so one could store both a literal and a fluent rendering.
- Devanagari-written English words (e.g., "मीटिंग") may confuse NMT models trained on clean Hindi. LLMs typically recover them, but this is unverified for IndicTrans2. Test on a small set.

### Gaps
- No benchmark was found that measures any MT system (IndicTrans2, Sarvam, NLLB, Gemma, Llama, Qwen, GPT-class) on *disfluent, verbatim, Devanagari-code-mixed* Hindi ASR transcripts. No measured negation-drop or hallucination rates for Hindi→English were found either.
- The exact IndicTrans2 hi→en chrF++ (FLORES-200 / IN22-Gen / IN22-Conv) and NLLB-200 Hindi numbers were not retrievable (arXiv and HF blocked).

---

## 5. The synthetic-vs-real speech gap, methods to close it, and how to measure it

### Takeaway
Models trained only on TTS speech consistently do much worse on real speech. The main stated causes are limited speaker and prosody variability and missing channel effects; one older study reported a roughly 10× WER gap. The gap shrinks with:
- better speaker embeddings and more diverse voices;
- mixing in real recordings;
- classic augmentation: speed perturbation 0.9/1.0/1.1 (about 4.3% relative WER gain), noise and RIR, and real telephone codec chains (G.711 / AMR-NB / AMR-WB / Opus / packet loss).

The most direct way to measure the gap is train-on-synthetic, test-on-real WER (or ASR-BLEU) on a held-out real Hindi dev set. Fréchet-style distances are only a supplementary proxy, and they depend on the embedding.

### Cited Findings
- Rossenbach et al. (ASRU 2021, arXiv 2104.05379) studied how TTS pre-processing, speaker embedding and input encoding affect synthetic data for AED ASR, with comparisons to hybrid, CTC and transducer systems. Internal LM subtraction gave up to 38% relative improvement; for non-AED systems, adding synthetic data had "no relevant effect" on LibriSpeech-100h — [arXiv 2104.05379](https://arxiv.org/pdf/2104.05379) (search extract)
- Hilmes, Rossenbach & Schlüter (arXiv 2407.17997) trained ASR *purely* on synthetic LibriSpeech-100 reproductions (FastSpeech-2-style TTS) for AED, hybrid DNN-HMM and GMM-HMM systems. The real–synthetic gap depends strongly on the speaker embedding, and it is smaller for traditional (HMM) systems — [arXiv 2407.17997](https://arxiv.org/pdf/2407.17997) (search extract; exact WERs not retrieved)
- An earlier Rossenbach et al. (2019) LibriSpeech study reportedly found synthetic-only training at about 50% dev-clean WER, against about 5% for natural speech, attributed largely to limited speaker variety — (search extract summarising [awesomepapers: rossenbach2019generating](https://awesomepapers.io/speech-audio/papers/rossenbach2019generating); secondary, verify against the original)
- An Edinburgh thesis (Minixhofer) finds that ASR trained only on synthetic data has much higher error on real speech, because TTS misses real-speech variability. Its scaling study suggests even diffusion TTS would need very large amounts of synthetic data to match real-data ASR. It proposes the TTSDS distributional metric, which correlates with human ratings over 35 systems — [Edinburgh ERA thesis](https://era.ed.ac.uk/handle/1842/44684) (search extract). A related paper proposes training ASR on synthetic speech and testing on real speech as a distance measure, which correlates with MOS — [arXiv 2310.00706](https://export.arxiv.org/pdf/2310.00706) (search extract)
- Fréchet Speech Distance and its MMD variant: WavLM Base+ embeddings agree best with human ratings, but results shift with the setup and there is no agreed embedding. The metrics are presented as complements to listening tests — [arXiv 2601.21386](https://arxiv.org/abs/2601.21386v1) (search extract). No study was found validating FAD/FSD as a predictor of ASR training gains — same search.
- Synth2Aug (SLT 2021, arXiv 2011.11818) used multi-speaker TTS from target-domain text to help cross-domain speaker recognition, combined with multi-style training. The authors note that channel effects are hard for TTS to capture — [arXiv 2011.11818](https://arxiv.org/abs/2011.11818) (search extract). "Speech Recognition with Augmented Synthesized Speech" (arXiv 1909.11699) states that synthetic speech still does not deliver the value of real speech for ASR — [arXiv 1909.11699](https://arxiv.org/pdf/1909.11699) (search extract). Cross-lingual multi-speaker TTS plus voice conversion improves low-resource ASR with only one target-language speaker — [arXiv 2204.00618](https://arxiv.org/pdf/2204.00618) (search extract)
- Hibiki (ICML 2025) is a simultaneous S2ST model for Fr→En trained on aligned *synthetic* data; the authors released about 900 h of synthetic data. Some evaluation uses VoxPopuli, with real human interpretation as ground truth. No quantitative real-vs-synthetic ablation was retrieved — [arXiv 2502.03382](https://arxiv.org/html/2502.03382v1); [ICML poster](https://icml.cc/virtual/2025/poster/44512) (search extract)
- Speed perturbation (Ko et al., Interspeech 2015) uses factors 0.9, 1.0 and 1.1 (3× data). It gave an average 4.3% relative WER improvement across 4 LVCSR tasks with 100–960 h of training data — [Ko et al. 2015](https://www.danielpovey.com/files/2015_interspeech_augmentation.pdf) (search extract). It is implemented as `torchaudio.transforms.SpeedPerturbation` — [torchaudio docs](https://docs.pytorch.org/audio/2.3/generated/torchaudio.transforms.SpeedPerturbation.html) (search extract). The RIR datasets of Ko et al. 2017 (simulated and real RIRs) are widely reused, but their WER gains were not retrieved — [Speech Robust Bench arXiv 2403.07937](https://arxiv.org/pdf/2403.07937) (search extract)
- **Telephone simulation tooling**:
  - `phonesim` runs *real* codecs through ffmpeg: G.711 (ITU-T segmented coder in-process), G.722, G.726, Opus, and AMR-NB/AMR-WB (the last needs libopencore-amr and libvo-amrwbenc). It models band edges, frame erasures with concealment, jitter buffers, AGC, noise, multi-transcode chains and clock drift. It is seeded, and input/output **default to 24 kHz**. EVS and G.729 are not offered (no open encoder) — [phonesim README](https://github.com/deepmark/phonesim) (read directly)
  - A Hindi telephony benchmark pipeline uses 8 kHz resampling, a 300–3400 Hz band limit, G.711 μ-law, added noise and packet loss — [CallClear README](https://github.com/Siddharth-21s/callclear) (read directly)
  - An anti-spoofing study's channel augmentation covers landline (G.711/G.726), cellular (AMR, AMR-WB, GSM), VoIP and satellite, with packet loss and 8/16 kHz resampling — [arXiv 2110.10491](https://arxiv.org/pdf/2110.10491) (search extract)
  - A Mandarin E2E ASR paper encoded training audio with AMR-NB and G.711 and reports a clear gain on 8 kHz test data, with negligible change on 16 kHz data — [arXiv 2010.12715](https://arxiv.org/pdf/2010.12715) (search extract; magnitudes not retrieved)
- Gramvaani (real low-bandwidth Hindi) roughly doubles WER for every system relative to clean Kathbath: IndicWhisper 26.8 vs 10.3, and Google STT 59.9 vs 14.3 — [Vistaar README](https://github.com/AI4Bharat/vistaar) (read directly). This is direct evidence of a channel gap for Hindi.

### Inferences
- **Recommended for this PS**:
  - **Inputs**: prefer real Hindi recordings, since real hours count only by unique source interval. Use TTS mainly for voice and text diversity. Zero-shot voice cloning from many real speakers attacks the main documented cause of the gap (speaker variability).
  - **Training-time augmentation** (it adds no dataset hours under PS rules):
    - noise at SNR 0–20 dB from MUSAN or DNS-challenge noise;
    - real and simulated RIRs;
    - speed 0.9/1.1;
    - a telephone chain. Example: 24 k → 8 k, 300–3400 Hz band-pass, G.711 μ-law/A-law or AMR-NB 4.75–12.2 kbps or Opus 6–12 kbps, optional packet loss 1–5%, then back to 24 kHz to match the test condition ("lower-bandwidth upsampled to 24 kHz"). Use `phonesim` or plain ffmpeg (`-ar 8000 -c:a pcm_mulaw` / `-c:a libopencore_amrnb`, then `-ar 24000`).
  - **Disfluencies**: inject them into TTS text (fillers such as "अं", "मतलब", repetitions, false starts), because clean TTS lacks them and the hidden set is real speech.
- **Measuring the gap**: keep a held-out *real* Hindi dev set, including some telephone or noisy clips. Report:
  - train-synthetic / test-real ASR WER (or the downstream S2ST ASR-BLEU) against train-real / test-real;
  - optionally FSD with WavLM embeddings between the synthetic and real pools, as a cheap secondary diagnostic.
- PS-specific: TTS output must be checked, e.g., by ASR round-trip CER against the input text and number-word checks. Mispronounced numbers and skipped or repeated words are explicitly named as problems in the PS.

### Gaps
- No Hindi-specific (or S2ST-specific) quantitative real-vs-synthetic ablation was found.
- No source quantifies how much telephone-codec augmentation reduces Hindi ASR WER, or S2ST ASR-BLEU, on real telephone audio.
- **Claim that turned out wrong or misleading when checked**: the AI4Bharat Vistaar README is internally inconsistent for Hindi. The per-language table gives IndicWhisper Hindi **CommonVoice 11.4 / FLEURS 15.0**. The Hindi model-comparison table in the same README gives **FLEURS 11.4 / CommonVoice 15.0** — the two values are swapped ([Vistaar README](https://github.com/AI4Bharat/vistaar), read directly). Any "IndicWhisper Hindi FLEURS WER" quoted from that README is ambiguous unless checked against the paper (arXiv 2305.15386).
  - Two related search-engine claims also proved misleading. Third-party write-ups cite Saaras v3's "~19% WER" as a Hindi result, but Sarvam's own page gives it as a 10-language IndicVoices aggregate ([Sarvam STT page](https://www.sarvam.ai/speech-to-text)). A search summary also claimed WhisperX achieves "±20 ms" word timing, but independent TIMIT results put only 22.4% of WhisperX word boundaries within 10 ms and 52.7% within 25 ms ([arXiv 2406.19363](https://arxiv.org/pdf/2406.19363)).
