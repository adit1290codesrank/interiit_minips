# Hindi speech/data tooling: is the study's stack current (Oct 2026) and correctly characterised?

Verification labels used throughout: **[P]** = primary source read directly (GitHub raw/clone, PyPI JSON, local wheel inspection, the PS PDF); **[S]** = search-engine extract of the source (the page itself was blocked by the proxy: arxiv.org, aclanthology.org, huggingface.co, docs.sarvam.ai, sarvam.ai, vaani.iisc.ac.in, ai4bharat.iitm.ac.in all returned CONNECT 403); **[S2]** = secondary/third-party source via search; **[I]** = my inference. Date of checks: 14 Oct 2026.

## Q0. Summary verdicts across the whole stack (cross-cutting)

### Takeaway
Most of the stack is still current, but six points need correcting. (1) IndicVoices Hindi is far smaller than 105 h on the primary paper's numbers: about 65 h train + 5 h test transcribed in v1. (2) Saaras **V4** (Aug 2026) and Bulbul **V4** now exist; the PS pins `bulbul:v3`/`shubh`, so v3 must be pinned explicitly. (3) Saaras `verbatim` emits at least some English words in **Latin** script, so its output needs Devanagari transliteration for the PS. (4) Omnilingual ASR has **v2** checkpoints and an unlimited-length variant (Dec 2025), so the "< 40 s" caveat is now model-specific. (5) Nemotron 3.5 ASR (June 2026) and Chatterbox Multilingual V3 plus a Hindi single-language pack are new candidates. (6) The IndicVoices training recipe uses *normalised* text as the ASR target, so IndicConformer/IndicASR are not natively verbatim.

### Cited Findings
| Item | Report's claim | Status Oct 2026 | Verdict | Source / level |
|---|---|---|---|---|
| IndicVoices licence | CC BY 4.0, gated | CC BY 4.0 per AIKosh (from earlier notes); GitHub README still has no licence section | Keep; still verify on the HF card | [P] README; earlier notes |
| IndicVoices Hindi hours | 105 h usable (R1) | Paper v1 Table 6: **Hindi train 65 h / 287 speakers, test 5 h / 63 speakers** (valid not readable); Lahaja paper also says the IV Hindi subset is 65 h. Later releases are bigger overall (12k/3.2k h README; 23.7k/11.2k h AIKosh/HF) but no Hindi figure was found | **Overstated / unverified**: plan for ~70 h or less until the card is checked | [S] [IndicVoices paper](https://arxiv.org/pdf/2403.01926), [Lahaja](https://arxiv.org/html/2408.11440v1) |
| IndicVoices conversational | "single-speaker conversational" | Raw calls are two parties on **one telephone bridge**. Segments carry `speaker_id`, and the official chunker drops `speaker_id == 1` segments | Correctable: use per-speaker segments only (see Q1) | [S] paper; [P] [chunker](https://github.com/AI4Bharat/IndicVoices/blob/master/data_prep_scripts/utils_dataset_create_iv.py) |
| Saaras v3 verbatim | Keeps fillers and spoken numbers; English script undocumented | Docs example: "मेरा फोन नंबर है नौ आठ चार **zero** नौ पांच zero…", i.e. English in **Latin**. **Saaras V4** released 24 Aug 2026, with five modes including verbatim | Update: test V4; transliterate Latin tokens | [S] [Saaras docs](https://docs.sarvam.ai/api/getting-started/models/saaras.md); [S2] [TechStory](https://techstory.in/?p=576600) |
| Bulbul | v3 `shubh` | **Bulbul V4** live alongside V3 (changelog 6 Oct 2026); in V4, "Shubh" is renamed "Shubh – Hinglish Ad Voice"; v3 is called the "stable" model; no v3 deprecation found | Pin `model="bulbul:v3"`, `speaker="shubh"`, `en-IN` as the PS requires | [S] [Sarvam changelog](https://docs.sarvam.ai/changelog/2026/10/6.md), [V4 Flash guide](https://docs.sarvam.ai/api/api-guides-tutorials/text-to-speech/best-practice-guide-for-bulbul-v-4-flash); [P] PS §3 |
| IndicConformer | 600M MIT, needs AI4Bharat NeMo fork | Unchanged: 600M multi plus per-language `hi_hybrid_rnnt_large` .nemo; still needs the AI4Bharat NeMo fork. No "v2" found | Current; caveat on normalised training targets | [P] [README](https://github.com/AI4Bharat/IndicConformerASR) |
| IndicWhisper (Vistaar) | Benchmark reference | No newer release found | Current but superseded as a verbatim tool | earlier notes |
| Omnilingual ASR | hin_Deva CER 3.6, < 40 s | CSV still says hin_Deva CER 3.6, 236.7 training h. **Dec 2025: `_v2` CTC/LLM checkpoints with improved CER, plus `omniASR_LLM_Unlimited_*_v2` for unlimited length**. Code and models Apache-2.0 | Update the caveat | [P] [README](https://github.com/facebookresearch/omnilingual-asr), [CSV](https://github.com/facebookresearch/omnilingual-asr/blob/main/per_language_results_table_7B_llm_asr.csv) |
| torchaudio MMS_FA | "2.11: still shipped" | 2.11.0 is the latest on PyPI. The wheel's `pipelines/__init__.py` exports `MMS_FA` and `functional/__init__.py` exports `forced_align` | **Confirmed** | [P] PyPI JSON + local wheel inspection |
| ctc-forced-aligner | Install from GitHub; PyPI name is a different project | PyPI `ctc-forced-aligner` 1.0.2 → `deskpai/ctc_forced_aligner` (different project). GitHub code BSD-2; default model `MahmoudAshraf/mms-300m-1130-forced-aligner` is **CC-BY-NC 4.0** | Confirmed | [P] PyPI JSON; [P] [README](https://github.com/MahmoudAshraf97/ctc-forced-aligner) |
| Qwen3-ForcedAligner | 11 languages, no Hindi | Still 11 (zh, en, yue, fr, de, it, ja, ko, pt, ru, es). Native Transformers support added 26 Jun 2026. Qwen3-ASR (not the aligner) does support Hindi | Confirmed: not usable for Hindi timing | [P] [Qwen3-ASR README](https://github.com/QwenLM/Qwen3-ASR) |
| IndicTrans2 | MIT, 200M distilled | Models MIT; IN22 CC-BY-4.0; last update **Jan 2025 RoPE long-context (2048-token) variants**. No IndicTrans3 found | Current | [P] [README](https://github.com/AI4Bharat/IndicTrans2) |
| Sarvam-Translate | No public Hindi figures | Open-weights, fine-tuned from Gemma3-4B-IT with AI4Bharat; 22 languages | Add as an open MT candidate | [S] [Sarvam blog](https://www.sarvam.ai/blogs/sarvam-translate) |
| Indic Parler-TTS | Apache-2.0 | Unchanged | Current | earlier notes; [S2] |
| IndicF5 | MIT, permissioned voices | README: 11 languages incl. Hindi; needs ref audio **+ ref text**; 24 kHz; trained on 1,417 h | Current | [P] [README](https://github.com/AI4Bharat/IndicF5) |
| Veena | Hindi/English, 4 voices | Apache-2.0, 3B Llama + SNAC 24 kHz; voices kavya/agastya/maitri/vinaya; Maya Research now calls it an earlier model (Maya1 newer open-weight) | Current; check Maya1 | [S] [HF card](https://huggingface.co/maya-research/Veena) |
| DNSMOS / WavLM-SV | Signal and speaker gates | No Hindi validation for DNSMOS, UTMOSv2 or Distill-MOS found (see Q6) | Keep as soft gates only | [S] |

### Inferences
- The two changes most likely to cost points are: (a) the 105 h IndicVoices plan, which probably cannot be met from Hindi IV alone; and (b) an accidental Bulbul V4 call (the renamed "Shubh" voice) producing non-compliant targets. Both are cheap to fix now.

### Gaps
- The HF IndicVoices card (current Hindi hours by version) and the Sarvam docs pages could not be opened directly (proxy 403). Every Sarvam claim here rests on search extracts.

## Q1. IndicVoices Hindi: hours, splits, speakers, conversational format, licence, and the "one speaker per clip" critique

### Takeaway
The critique is half right. IndicVoices conversations were recorded as two participants on a **single telephone bridge** (one mixed channel), so whole conversational files violate "one speaker per clip". However, the transcripts are **segment-level with a per-segment `speaker_id`**, and AI4Bharat's own chunker already keeps only one speaker's segments (it skips `speaker_id == 1`). Per-speaker segment cuts therefore satisfy the rule, provided cross-talk and backchannels are screened. They also count only their own intervals toward the hour budget. Hindi transcribed hours in the primary v1 paper are about **65 h train + 5 h test**, well short of the 105 h the report allocates.

### Cited Findings
- Paper v1 totals: "7348 hours of read (9%), extempore (74%) and conversational (17%) audio data from 16237 speakers", of which 1,639 h are transcribed — [IndicVoices paper](https://arxiv.org/pdf/2403.01926) [S]. The current README says 12,000 h, 8% read / 76% extempore / 15% conversational, 22,563 speakers, 3,200 h transcribed, "median of 122 hours per language" — [README](https://github.com/AI4Bharat/IndicVoices) [P]. AIKosh says 23.7K h and 51K speakers — [AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/indicvoices.html) [S].
- Hindi split (paper v1, Table 6): **train 287 speakers / 65 h; test 63 speakers / 5 h**; validation column garbled in the extract — [IndicVoices paper](https://arxiv.org/pdf/2403.01926) [S]. Corroborated: the "Hindi subset of the IndicVoices dataset contains 65 hours" — [Lahaja paper](https://arxiv.org/html/2408.11440v1) [S]. A third-party aggregate (`fhai50032/asr-hindi`) lists `indicvoices_hi` at 300 h (unofficial; from the earlier notes).
- Conversational recording method: the mobile app captured single speakers at 44.1 kHz, but for pairs "they set up a telephone bridge and connected two participants on a single call and recorded their conversation". Strangers produced small talk, so role-play scenarios were used — [IndicVoices paper](https://arxiv.org/html/2403.01926) [S].
- Transcript structure, from AI4Bharat's official chunker (repo HEAD d50726d, 22 Feb 2026) [P] — [utils_dataset_create_iv.py](https://github.com/AI4Bharat/IndicVoices/blob/master/data_prep_scripts/utils_dataset_create_iv.py):
  - Each raw file's JSON has parallel lists `verbatim[]` and `normalized[]`. Each element has `start`, `end`, `text` and **`speaker_id`**.
  - The chunker writes one wav per segment (`_chunk_{e}`) and **skips a segment if `verbatim['speaker_id'] == 1`**, if it is longer than 30 s or shorter than 0.05 s, or if it has empty text.
  - Per-chunk metadata is file-level: `speaker_id, scenario, task_name, gender, age_group, job_type, qualification, area, district, state, occupation, verification_report`, plus `verbatim`, `normalized`, `unsanitized_verbatim`, `unsanitized_normalized`.
  - The chunker asserts 16 kHz input after the README's ffmpeg downsampling step. Raw file names look like `<uuid>_0.json` / `<uuid>_1`.
- The ASR training target in this recipe is `'text': ntext`, i.e. the **normalised** (Level-2) transcript. A separate `utils_dataset_create_verbatim_manifest.py` exists for a verbatim manifest — [repo](https://github.com/AI4Bharat/IndicVoices/tree/master/data_prep_scripts) [P].
- PS rules: "One speaker per clip"; real hours = "the union of the source-file intervals you use. Overlapping or repeated cuts of the same recording count once"; clips ≤ 60 s; the test has "One speaker per session. Background noise and telephone audio may occur." — [PS](file:///home/user/interiit_minips/Sarvam.pdf) [P].
- Indic DiarBench (2026) builds on IndicVoices-style collection but recorded **separate microphone streams per speaker**, which the extract contrasts with the single-bridge IndicVoices phone recordings — [arXiv 2607.23808](https://arxiv.org/html/2607.23808v1) [S].

### Inferences
- **"One speaker per clip" is satisfiable.** Take only segments whose `speaker_id` matches the file's metadata speaker (the official recipe's `speaker_id != 1`). Never let a clip or streaming window span another speaker's segment. Then screen each kept segment for the other party's backchannels ("हाँ", "हम्म") and overlap, because the channel is mixed and segment boundaries are human-drawn. Cheap screens: pyannote overlapped-speech detection; or a WavLM-SV cosine check of each segment against the speaker's own centroid; or drop segments within ±0.3 s of a `speaker_id==1` segment.
- **Hour accounting favours segments.** Only the used segment intervals count, not the whole call. Merging consecutive same-speaker segments into ≤ 60 s windows counts the union, but such a window must not contain the other speaker's turn. The `_0`/`_1` suffix may mean each participant has a separate file of the same call. If so, cuts of the same call through two files are the "same recording", and their intervals overlap in time. I could not verify this, so it is a risk to check, not a fact.
- **Telephone-band conversational audio is a feature, not a bug.** The test includes telephone audio, and the PS allows upsampled lower-bandwidth audio. Conversational is only about 15–17% of IV, so even dropping it entirely costs roughly 10 h of Hindi on v1 numbers.
- **The 105 h R1 plan is likely infeasible from IV Hindi v1.** About 70 h transcribed, minus the speaker-1 halves of conversations, invalid segments and quality filtering, probably leaves about 50–65 h. Unless the current HF release has substantially more Hindi (the corpus grew roughly 7×), the shortfall needs other real sources: Kathbath (CC0, 150 h, read), FLEURS, Common Voice, Vaani Hindi if the licence checks out (Q2), and self-recorded speech. Alternatively, raise the synthetic share.
- IndicConformer/IndicASR is trained on normalised text by default, so it may standardise colloquial spellings and drop fillers. It is a draft tool, not a Level-1 oracle.

### Gaps
- The current HF card's Hindi hours per version (v1/v2/v3) and per style could not be read (huggingface.co blocked). A Hindi split by read/extempore/conversational was not found anywhere.
- Whether `_0`/`_1` files are two participants' copies of one call, or two halves, is unverified. So is whether the speaker-1 (unknown-metadata) party's segments are also transcribed in Hindi files.
- The native sample rate of the conversational (telephone) audio was not confirmed; it is likely 8 kHz [I].

## Q2. New 2025–2026 Indic datasets with redistributable licences

### Takeaway
No new **real, conversational, code-mixed Hindi** corpus with a clean redistributable licence appeared in 2025–26. The most important new real source is **Vaani** (IISc/ARTPARK + Google): about 31K h of audio and about 2.0–2.1K h transcribed across 109 languages. Its project site says CC-BY-4.0, but a derived HF subset says CC-BY-NC, and no Hindi-hours figure was found. New synthetic code-mixed corpora (BEHE-CMDisfl) exist, but they are themselves Parler-TTS output. "IndicVoices-ST" remains unconfirmed.

### Cited Findings
- Vaani: about 31,270 h of audio and about 2,067 h transcribed, 109 languages, 165 districts (arXiv March 2026). A later post says 31,255+ h and 2,043+ h — [arXiv 2603.28714](https://arxiv.org/html/2603.28714v2) [S]; [LinkedIn post](https://ee.linkedin.com/in/tvanatalu) [S2].
- Vaani licence: the project site states CC-BY-4.0, but a third-party Odia extraction lists CC-BY-NC 4.0 and defers to the ARTPARK-IISc terms — [vaani.iisc.ac.in](https://vaani.iisc.ac.in/) [S]; [HF Minutor/vaani-or-dataset](https://huggingface.co/datasets/Minutor/vaani-or-dataset) [S2].
- A HF discussion notes "audio quality is poor for English in Vaani" — [SraVaani discussion](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0/discussions/3) [S2].
- BEHE-CMDisfl (WILDRE 2026): a synthetic Bengali–English and Hindi–English speech corpus with disfluencies; text by ChatGPT, voiced with Indic Parler-TTS; licence not seen — [ACL 2026.wildre-1.5](https://aclanthology.org/2026.wildre-1.5.pdf) [S].
- COMI-LINGUA (Hinglish *text*, CC-BY-4.0) — [arXiv 2503.21670](https://arxiv.org/html/2503.21670) [S]. L3Cube-HingCorpus is CC-BY-NC-SA — [L3Cube](https://github.com/l3cube-pune/code-mixed-nlp) [S].
- Lahaja (12.5 h Hindi, multi-accent) and Svarah (Indian English) have no licence in their READMEs; see the earlier notes.
- Omnilingual ASR corpus: `facebook/omnilingual-asr-corpus`, CC-BY-4.0 (mostly low-resource languages) — [README](https://github.com/facebookresearch/omnilingual-asr) [P].
- Indic-ST: Krutrim lists an "Indian Multilingual Translation Corpus for Evaluating Speech LLMs" (IndicST) — [Ola Krutrim](https://ai-labs.olakrutrim.com/models/IndicST-Dataset) [S]. Its licence and Hindi content were not checked.

### Inferences
- If the Vaani Hindi licence is confirmed as CC-BY-4.0 on the official HF card, Vaani is the best top-up for the IV shortfall. It is spontaneous single-speaker phone speech (image-prompted descriptions) from many districts. Its transcription convention is not IV Level-1, so transcripts would need re-normalisation (English-word script, numbers as words). If any official copy says NC, exclude it, because the dataset goes to Sarvam, a company.
- Synthetic corpora such as BEHE-CMDisfl add nothing over generating one's own data, and their hours would count as synthetic.

### Gaps
- Vaani Hindi transcribed hours, its official HF licence text, and its transcription conventions could not be read (blocked).
- No Bhashini/ULCA Hindi conversational dataset with a verified redistributable licence turned up. "Pratinidhi" and "IndicVoices-ST" were not found.

## Q3. Newest Hindi ASR and verbatim/disfluency behaviour

### Takeaway
For drafts of Level-1 transcripts the best current pair is **Saaras V4 (or v3) verbatim**, post-processed to Devanagari, plus **Omnilingual ASR v2** (CTC or LLM) as an independent second opinion. IndicConformer is still useful for CTC alignment but is trained on normalised targets. Qwen3-ASR and Nemotron 3.5 ASR now support Hindi but have no public Hindi verbatim evidence. Canary-1B-v2 and Parakeet-v3 do **not** cover Hindi. No benchmark measures disfluency retention for Hindi, so a human pass remains mandatory.

### Cited Findings
- Saaras v3 modes: transcribe, translate, verbatim, translit, codemix. Verbatim is "exact word-for-word transcription without normalization, preserving filler words and spoken numbers", with the example output "मेरा फोन नंबर है नौ आठ चार zero नौ पांच zero नौ पांच zero" — [Saaras docs](https://docs.sarvam.ai/api/getting-started/models/saaras.md); [output-mode guide](https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/how-to/select-output-mode.md) [S]. In codemix, English words stay in Latin — [Sarvam building-for-India](https://docs.sarvam.ai/api/building-for-india) [S].
- Saaras v3 scores about 19% WER on IndicVoices (vendor, multi-language aggregate) — [Sarvam ASR blog](https://www.sarvam.ai/blogs/asr) [S]. A third-party card lists Saaras v3 at 0.237 WER on a Hindi-female IndicTTS subset (out-of-domain, unverified) — [Friendli card](https://friendli.ai/models/manishehehe/whisper-small-hindi-lora) [S2].
- **Saaras V4** was announced 24 Aug 2026, GA for Voice Agents 2 Sep: an audio encoder plus a 3B hybrid state-space LM, 22 scheduled languages plus global English, outputs "verbatim transcription, normalised text, code-mixed text, transliteration and translation". The evidence is vendor-only — [TechStory](https://techstory.in/?p=576600) [S2]; [Remio analysis](https://www.remio.ai/post/sarvam-ai-saaras-v4-raises-the-bar-for-indic-speech-but-benchmarks-are-only-the) [S2].
- Omnilingual ASR: hin_Deva CER 3.6 (7B LLM-ASR, 236.7 training h). The Dec 2025 update adds `omniASR_{CTC,LLM}_{300M,1B,3B,7B}_v2` with improved CER and `omniASR_LLM_Unlimited_*_v2` for unlimited audio length. The 40 s limit applies to the CTC and LLM suites. Apache-2.0 — [README](https://github.com/facebookresearch/omnilingual-asr); [CSV](https://github.com/facebookresearch/omnilingual-asr/blob/main/per_language_results_table_7B_llm_asr.csv) [P].
- Qwen3-ASR-1.7B/0.6B (Jan 2026) list Hindi among 30 languages. Hindi appears only inside aggregated "Fleurs†" and "News-Multilingual" scores; no per-language Hindi WER is published — [Qwen3-ASR README](https://github.com/QwenLM/Qwen3-ASR) [P].
- NVIDIA Nemotron 3.5 ASR: 600M streaming, 40 language-locales including hi-IN, released 4 Jun 2026 under the OpenMDW licence. A third-party review lists Hindi as "transcription-ready" — [NVIDIA Perspectives FAQ](https://perspectives.nvidia.com/nemotron-speech/task/faq/what-open-speech-models-handle-hindi-english-or-mandarin-english-code-switching/) [S]; [Wavect review](https://wavect.io/blog/nvidia-nemotron-3-5-asr-production-review.md) [S2]; [OpenWhispr](https://openwhispr.com/blog/parakeet-vs-whisper-vs-nemotron) [S2].
- Canary-1B-v2 / Parakeet-TDT-0.6B-v3 report 25 European languages; Hindi is not included — [arXiv 2509.14128](https://arxiv.org/pdf/2509.14128) [S].
- AI4Bharat + Bodhan AI "Indic-Transcribe": 1.2B, 26 Indic languages + English, 1.3M h, aimed at code-mixed speech. Openness and licence are not stated in the extract; date is around Aug 2026 — [Analytics India Mag](https://analyticsindiamag.com/ai-news/ai4bharat-bodhan-ai-build-12-bn-parameter-speech-model-for-26-indian-languages) [S2].
- IndicConformer is unchanged: `indicconformer_stt_multi_hybrid_rnnt_600m.nemo` and `indicconformer_stt_hi_hybrid_rnnt_large.nemo`, AI4Bharat NeMo fork — [README](https://github.com/AI4Bharat/IndicConformerASR) [P]. Its training recipe uses normalised text — [chunker](https://github.com/AI4Bharat/IndicVoices/blob/master/data_prep_scripts/utils_dataset_create_iv.py) [P].
- Whisper-family models delete fillers ("implicit disfluency removal"), as in the earlier report; no change found.

### Inferences
- **Script handling.** Saaras verbatim's "zero" example shows English tokens come back in Latin. The PS requires English words in Devanagari, so add a step that maps Latin tokens to Devanagari (LLM with a cached lexicon, as the report already proposes for loanwords). Gate it by checking that no Latin characters remain.
- **Disfluency ranking (no benchmark; reasoning only).** Saaras verbatim is the only system documented to keep fillers. CTC models (Omnilingual CTC, IndicConformer-CTC) tend to emit what they hear, but they are trained on targets that may omit fillers. LLM-decoders (Omnilingual-LLM, Qwen3-ASR, Saaras V4's LM) may "clean up". Measure filler recall on the Part D clips and report it.
- For the round-trip TTS gate in C3, Omnilingual v2 (different family, Apache) plus IndicConformer remain a good pair. Saaras is a paid API whose outputs are allowed per PS §10.

### Gaps
- No Hindi-only WER for Saaras v3/V4, Qwen3-ASR or Nemotron 3.5. No published filler/repetition retention numbers for any Hindi ASR.
- Google USM/Chirp 3 Hindi numbers and verbatim behaviour were not researched; budget ran out.

## Q4. Aligners for Hindi word timing

### Takeaway
The report's choice holds. **torchaudio 2.11 `MMS_FA` + uroman** is verified present. `ctc-forced-aligner` from GitHub is a sound cross-check. **NeMo NFA + IndicConformer-CTC** is the only native-Devanagari, MIT-weight option. Qwen3-ForcedAligner still excludes Hindi. No published Hindi word-boundary accuracy exists as of Oct 2026. New 2026 aligners (learned-DP, soft-DP) do not report Hindi.

### Cited Findings
- torchaudio 2.11.0 is the latest PyPI release. Inspecting the cp312 manylinux wheel shows `MMS_FA` in `torchaudio/pipelines/__init__.py` and `forced_align` in `torchaudio/functional/__init__.py` — [PyPI torchaudio](https://pypi.org/project/torchaudio/2.11.0/) [P, local].
- ctc-forced-aligner: install with `pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git`; `--romanize` is required with the default model; code BSD-2; "the default model has CC-BY-NC 4.0 License". The PyPI package of the same name (1.0.2) points to `deskpai/ctc_forced_aligner` — [README](https://github.com/MahmoudAshraf97/ctc-forced-aligner) [P]; [PyPI JSON](https://pypi.org/project/ctc-forced-aligner/) [P].
- Qwen3-ForcedAligner-0.6B covers 11 languages (zh, en, yue, fr, de, it, ja, ko, pt, ru, es), aligns up to 5 min, Transformers-native since 26 Jun 2026 — [Qwen3-ASR README](https://github.com/QwenLM/Qwen3-ASR) [P].
- New research aligners:
  - Learned dynamic programming over SSL features beats MFA and MMS on TIMIT/Buckeye and is tested on Dutch/German/Hebrew; no Hindi — [arXiv 2606.10675](https://arxiv.org/pdf/2606.10675) [S].
  - Differentiable soft-DP aligner — [arXiv 2606.25460](https://arxiv.org/pdf/2606.25460) [S].
  - `parseh/aligner-hi`: a CTC aligner derived from IndicWav2Vec-Hindi, "no accuracy guarantee" — [HF](https://huggingface.co/parseh/aligner-hi) [S].
- Prior English comparison: MFA outperformed WhisperX and MMS — [arXiv 2406.19363](https://arxiv.org/html/2406.19363v1) [S]. Code-mixed Hindi-English "presents complex challenges to forced-alignment techniques" — [Pandey et al. 2020](https://slam.phil.uni-duesseldorf.de/papers/2020/PandeyGogoiTang_2020_CodeMixedHindi_CSWorkshop.pdf) [S].

### Inferences
- **Licensing nuance.** MMS_FA and the ctc-forced-aligner default weights are CC-BY-NC. Timestamps are generally not treated as derivative works of model weights, but a cautious team can make NFA + IndicConformer (MIT) the production aligner for the 200 h dataset and keep MMS_FA for the Part D demo and cross-check. I am not aware of any authority on this point.
- Two-aligner disagreement (MMS-romanised vs. IndicConformer-native) is a stronger quality signal than either alone, because the errors come from different tokenisations.

### Gaps
- No Hindi or code-mixed word-boundary accuracy (ms) for any aligner was found, so the report's "your Part D number is the only evidence" stands.
- NFA + IndicConformer hybrid-RNNT CTC-head compatibility with mainline NeMo is untested.

## Q5. Newest TTS for Hindi / code-mixed Hindi, licences, prosody, and the "flat prosody degrades Audio Quality KPI" critique

### Takeaway
The open TTS pool is current, with three additions: **Chatterbox Multilingual V3** (Hindi included, plus a dedicated Hindi fine-tune), **Kokoro** (Apache, Hindi voices, no cloning) and Maya1 (successor to Veena). **CosyVoice3** (no Hindi) and **Fish/OpenAudio** (non-commercial/research licence) should stay out. The critique's KPI claim is **wrong on mechanism**. The PS's "Audio quality and naturalness" (6 pts) rates "how clean and natural the **English** speech sounds", and the English targets are Bulbul v3 `shubh` by rule. Hindi TTS prosody affects only the *input* side. The real risk is an input-domain gap: a model fine-tuned on flat, clean, read-style synthetic Hindi may wait, segment or recognise worse on real spontaneous telephone Hindi. That hits intent, entity and latency (52 pts), not audio quality.

### Cited Findings
- PS: "All English target audio in supervised training pairs must be synthesised with Bulbul: bulbul:v3, voice shubh, language_code="en-IN", 24 kHz". The evaluation row reads "Audio quality and naturalness — How clean and natural the English speech sounds, as rated by listeners", worth 6 points. Intent and entity are 18 each, latency 16 — [PS §3, §9.1](file:///home/user/interiit_minips/Sarvam.pdf) [P].
- Chatterbox Multilingual V3: 0.5B, 23 languages including Hindi, "reduced hallucination", zero-shot cloning. The Single Language Pack includes Hindi (`ResembleAI/Chatterbox-Multilingual-hi`) — [Chatterbox README](https://github.com/resemble-ai/chatterbox) [P]. MIT licence and about 5 s cloning — [hinglish-tts repo](https://github.com/harrrshall/hinglish-tts) [S2]. A paper notes vanilla Chatterbox performs well on Hindi but does not tokenise Telugu/Tamil — [Praxy Voice, arXiv 2604.25441](https://arxiv.org/html/2604.25441v1) [S].
- Kokoro: 82M, "Apache-licensed weights", lang code `'h' => Hindi` — [Kokoro README](https://github.com/hexgrad/kokoro) [P]. Hindi voicepacks are one female (Alpha) and one male (Omega); no voice cloning — [tts.ai listing](https://tts.ai/voices/kokoro/) [S2].
- Fun-CosyVoice 3.0 covers 9 languages (zh, en, ja, ko, de, es, fr, it, ru) plus Chinese dialects. **No Hindi** — [CosyVoice README](https://github.com/FunAudioLLM/CosyVoice) [P].
- Fish Audio: weights under the "FISH AUDIO COMMUNITY LICENSE", which allows "research and non-commercial uses… and… limited uses for a Commercial Purpose by Licensees whose Annual Revenue is less than the Revenue Threshold". S2-Pro weights are under the Research License — [fish-speech README/LICENSE](https://github.com/fishaudio/fish-speech) [P].
- Veena: Apache-2.0, 3B Llama + SNAC 24 kHz, Hindi/English/code-mixed, 4 voices. Maya Research calls Veena an earlier model; Maya1 is its newer open-weight release — [HF Veena](https://huggingface.co/maya-research/Veena) [S].
- IndicF5 needs a reference clip **and its transcript**; 11 languages; trained on Rasa, IndicTTS, LIMMITS, IndicVoices-R — [IndicF5 README](https://github.com/AI4Bharat/IndicF5) [P].
- Prosody evidence:
  - "Open-weight text-to-speech models for Hindi produce fluent narration but fail to adapt prosody to narrative context" (fixed by conditioning IndicF5 on retrieved reference audio) — [ICML 2026 poster](https://icml.cc/virtual/2026/77343) [S].
  - On a self-published Hinglish rubric, Indic Parler-TTS scored 3.40 overall (1.75 on pure-Roman Hindi); IndicF5 scored 2.13 unpatched vs 4.70 after preprocessing and a duration patch — [hinglish-tts](https://github.com/harrrshall/hinglish-tts) [S2].
  - The RASMALAI paper reports Indic Parler-TTS near-human MUSHRA for some languages; Hindi not shown in the extract — [arXiv 2505.18609](https://arxiv.org/pdf/2505.18609) [S].
  - No common-test-set Hindi MOS across open models was found.
- Synthetic-data transfer for Indic ASR: the "TTS-STT Flywheel" fine-tuned Whisper on about 22K synthetic entity-dense Indic–English utterances (under $50). Telugu entity-hit-rate rose from 0.027 to 0.473, but the Hindi model "underperformed the commercial system" and all models missed pre-registered targets. Preprint, self-reported — [arXiv 2605.03073](https://arxiv.org/pdf/2605.03073) [S]; [chatpaper mirror](https://chatpaper.com/pt/chatpaper/paper/275803) [S2]. A 2026 study of synthetic speech for Hindi/Kannada/Telugu ASR fine-tuning exists but was not examined — [arXiv 2606.17662](https://arxiv.org/pdf/2606.17662) [S].

### Inferences
- **What the critique gets wrong.** Hindi TTS cannot directly lower the Audio Quality score, because listeners rate the system's English output, and that output imitates Bulbul `shubh` targets, which are fixed. The only indirect path is that a model confused by out-of-domain input could stutter, insert pauses or emit garbled English. That shows up mainly as intent, entity and latency loss, and only marginally as naturalness.
- **What the real risk is.** Input-side domain gap, which hits intent, entity and latency, plus the audit samples transcripts and timings but not Hindi audio naturalness. Mitigations, roughly in order of value:
  1. Keep real speech the majority of the dataset (but see Q1: IV hours are smaller than planned).
  2. Clone many *consented* real voices: team members, or prompts whose licence and consent permit cloning. Do not clone IndicVoices speakers without consent; CC-BY on recordings is not voice-cloning consent.
  3. Use reference-audio prosody variety: IndicF5 and Chatterbox inherit prosody from the prompt, so take spontaneous, emotional and telephone prompts rather than read-style ones.
  4. Write disfluencies and fillers into the scripts.
  5. Apply channel augmentation (telephone/codec/noise) at training time; it adds no hours.
  6. Measure train-synthetic/test-real vs. train-real/test-real on a speaker-disjoint real dev set.
- Chatterbox Multilingual V3 (MIT, Hindi fine-tune, cloning) is the most significant new open option for voice diversity. Verify its handling of Devanagari-written English words and lakh/crore readings with the two-ASR round-trip gate before use.

### Gaps
- No Hindi MOS for Chatterbox-hi, Kokoro-Hindi, Maya1 or Veena on a shared set. The Chatterbox licence was confirmed only via a secondary source (the README grep found no licence line). Svara (Kenpath) licence was not found. Sesame/Orpheus Indic variants were not researched.

## Q6. Hindi→English MT for code-mixed disfluent input; lakh/crore handling

### Takeaway
IndicTrans2 is still the newest AI4Bharat MT; there is **no IndicTrans3**. Its only update is the Jan 2025 RoPE long-context variants. **Sarvam-Translate** (open weights, Gemma3-4B-IT based) is a credible open second system. No public benchmark tests disfluent, Devanagari-code-mixed Hindi or lakh/crore preservation for any system. The report's "LLM primary, gated by the entity/negation checker; IndicTrans2 baseline" design remains the right one.

### Cited Findings
- IndicTrans2 update log: "Jan 18, 2025 – Long Context Models – RoPE-based variants… upto 2048 tokens". Licence table: model checkpoints MIT; BPCC-H-Daily and IN22 CC-BY-4.0 — [IndicTrans2 README](https://github.com/AI4Bharat/IndicTrans2) [P].
- Older IN22-Conv Hindi→English BLEU: IndicTrans2 about 60.9, Google Translate about 61.8, GPT-3.5 about 58.8 — [arXiv 2311.09216](https://arxiv.org/pdf/2311.09216) [S].
- Sarvam-Translate: open-weights, fine-tuned from Gemma3-4B-IT with AI4Bharat, paragraph-level for 22 languages; vendor human eval claims it beats much larger models — [Sarvam blog](https://www.sarvam.ai/blogs/sarvam-translate) [S]; Gemma was chosen for its Indic tokenisation efficiency — [DeepMind Gemmaverse](https://deepmind.google/models/gemma/gemmaverse/sarvam-ai/) [S].
- Code-mixed MT: a fine-tuned Mistral-7B with context-enriched synthetic data beat IndicTrans2 on Hinglish→English (+14.3%) — [NITK IDR](https://idr.nitk.ac.in/handle/123456789/28559) [S]. COMI-LINGUA: closed LLMs significantly outperform open models zero-shot on Hinglish tasks including MT — [arXiv 2503.21670](https://arxiv.org/html/2503.21670) [S]. LLMs generating code-switched text often default to Devanagari — [arXiv 2601.07153](https://arxiv.org/pdf/2601.07153) [S].
- Gemma 4 31B and Sarvam 30B exist (they appear in a 2026 Indic factual benchmark), but no Hindi→English MT scores were found — [arXiv 2608.15535](https://arxiv.org/pdf/2608.15535) [S].

### Inferences
- **Two-system design.** Use a frontier LLM (or Sarvam-30B / Gemma 4) for literal + fluent translations, with Sarvam-Translate or IndicTrans2-200M as the second, cheaper opinion. IndicTrans2 is trained on clean written Hindi, so expect it to drop fillers and possibly normalise "पाँच लाख" to "500,000". Whatever it does is a reportable Part D finding.
- There is no evidence base for lakh/crore preservation in any system. The report's hard gate ("lakh/crore kept in surface form; canonical value equality") is essential, not optional.

### Gaps
- No IN22 or code-mixed scores for Gemma 4, Llama 4, Qwen3 or Sarvam-30B/105B Hindi→English. No study of number-unit (lakh/crore) behaviour in MT found.

## Q7. Quality metrics: DNSMOS vs. newer predictors; Hindi suitability

### Takeaway
No MOS predictor (DNSMOS, UTMOS/UTMOSv2, Distill-MOS, NISQA) has a published validation on Hindi listener ratings. One multilingual study found the **original UTMOS generalised better across languages than UTMOSv2**. Keep DNSMOS as a coarse noise/clipping gate on clean TTS renders only, never as a naturalness ranker for Hindi. Calibrate any threshold on a small in-house Hindi listening set. WavLM-SV cosine for speaker consistency remains reasonable but needs a Hindi-calibrated threshold, as the report says.

### Cited Findings
- UTMOSv2 is the VoiceMOS Challenge 2024 Track 1 system (1st in 7 of 16 metrics), trained for high-quality synthetic speech naturalness — [UTMOSv2 README](https://github.com/sarulab-speech/UTMOSv2) [P].
- "Our preliminary experiments revealed that the original version [UTMOS] had better generalization ability to different languages" than UTMOSv2. The same paper calls NISQA-TTS the only open multilingual-trained MOS predictor known to the authors — [arXiv 2509.04830](https://arxiv.org/pdf/2509.04830) [S].
- UTMOS correlates well on VoiceMOS but "notably worse on the SOMOS dataset" — [arXiv 2509.20485](https://arxiv.org/pdf/2509.20485) [S].
- On URGENT2024 (English-centric), Distill-MOS reaches 0.93 LCC/SRCC and DNSMOS 0.91/0.87 — [UrgentMOS, arXiv 2601.18438](https://arxiv.org/pdf/2601.18438) [S].
- OpenBibleTTS reports per-language UTMOSv2 for 37 languages including Hindi, but without human-correlation data — [arXiv 2606.09553](https://arxiv.org/pdf/2606.09553) [S]. The PSP Indic accent benchmark defers MOS calibration to v2 — [arXiv 2604.25476](https://arxiv.org/pdf/2604.25476) [S].

### Inferences
- For the *English* side, which is what is actually scored, UTMOS/UTMOSv2 are reasonable proxies to track whether fine-tuning degrades the model's English naturalness relative to Bulbul `shubh` references.
- For the Hindi TTS side, the decisive gates are the two-ASR round trip and entity exact-match, not MOS.

### Gaps
- No Hindi-specific correlation for any predictor. SpeechBERTScore Hindi suitability was not checked.
