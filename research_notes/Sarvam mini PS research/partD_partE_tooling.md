# Parts D and E tooling: recording, alignment, translation, entity checking, wait-k simulation, WebSocket echo server, causal front-end

Scope: tool-level notes (install commands, APIs, versions, pitfalls) for Part D (build and measure) and the three Part E tracks (ML, Dev, Signal Processing) of the Inter IIT 15.0 Sarvam.ai Mini PS. Research date: 14 Oct 2026. Versions come from the live PyPI JSON API on that date. Several claims below were checked by downloading the wheel or sdist and reading its source, because the docs sites (docs.pytorch.org, readthedocs, huggingface.co, arxiv.org, audacityteam.org) could not be reached from this sandbox. Where a claim rests on that kind of source inspection, the citation is the PyPI release page of the package inspected.

Problem-statement facts these notes build on, from the local PDFs (`Inter IIT Tech Meet 15.0_ Sarvam.ai Mini PS.pdf` and §4, §5.1 and §7 of `Sarvam.pdf`):
- Audio is mono 16-bit PCM at 24 kHz in both directions. Input arrives in 80 ms frames of 3,840 bytes, in real time: frame k is sent at t = k × 80 ms and the last frame is zero-padded. Then comes `input_end` with `valid_samples`, followed by silent 80 ms frames until `session_complete`.
- The server sends PCM frames of at most 1 s each and exactly one `session_complete` text frame. Nothing may follow it.
- Playback rule: each frame starts at max(arrival time, end of the previous frame).
- If a frame cannot be delivered within 1 s of its scheduled time because the server is not reading, the session fails.
- `session_complete` must arrive within 30 s of `input_end`. The server must accept a connection within 10 s.
- Container: port 8000, path `/ws`. `GET /health` must return 200 within 300 s of start-up. All weights must be inside the image.
- Sessions are 10–60 s, one speaker, and may contain noise or telephone audio.
- KPIs: Entity accuracy (18 points), Intent accuracy (18), Latency (16), Translation quality (12), Audio quality (6).

---

## Q1. Part D: recording, conversion and forced alignment (torchaudio MMS_FA vs ctc-forced-aligner, uroman)

### Takeaway
Use MMS_FA (the 1,130-language MMS model trained on romanised text) through either `torchaudio.pipelines.MMS_FA` or MahmoudAshraf97's `ctc-forced-aligner` installed from GitHub. Both run on CPU at 16 kHz and need the Hindi transcript romanised with `uroman` (`lcode='hin'`). Two warnings that are easy to miss:
- The tutorial's own banner says these APIs "will be removed in 2.9". That plan was reversed: torchaudio 2.11 (the latest release, Mar 2026) still ships `forced_align`, `merge_tokens` and `MMS_FA`.
- The PyPI package named `ctc-forced-aligner` is not MahmoudAshraf97's project.

### Cited Findings
**torchaudio status**
- torchaudio entered a maintenance phase. APIs marked "Drop" were deprecated in 2.8 (6 Aug 2025) and "most APIs marked as drop are now removed" in 2.9 (15 Oct 2025). Decoding and encoding moved to TorchCodec, and `torchaudio.load()`/`save()` now run on top of TorchCodec. — [torchaudio releases](https://github.com/pytorch/audio/releases); [pytorch/audio#3902](https://github.com/pytorch/audio/issues/3902)
- In 2.8, `forced_align` was "slated for deletion". The 2.10 notes (21 Jan 2026) reversed that: "The C++ and CUDA extensions that were previously marked for deprecation are preserved and will remain in torchaudio (lfilter, RNNTLoss, CUCT, forced_align, overdrive)". The same release "marks the finalization of the TorchAudio migration". 2.11.0 (23 Mar 2026) adds no features and targets torch 2.11 "and future versions of torch". — [torchaudio releases](https://github.com/pytorch/audio/releases); [pytorch/audio#3902](https://github.com/pytorch/audio/issues/3902)
- Inspection of the torchaudio 2.11.0 wheel:
  - It still contains `torchaudio.functional.forced_align`, `merge_tokens`, `TokenSpan` and `torchaudio.pipelines.MMS_FA` (class `Wav2Vec2FABundle` with `get_model(with_star=True)`, `get_tokenizer()`, `get_aligner()`, `get_dict()`).
  - It no longer contains `torchaudio/io` (StreamReader/StreamWriter), `sox_effects` or `torchaudio/backend`.
  - `torchaudio.load` is documented as "relies on TorchCodec's decoding … `normalize`, `buffer_size`, and `backend` are ignored". — [torchaudio 2.11.0 on PyPI](https://pypi.org/project/torchaudio/2.11.0/) (wheel source)
- PyPI on 14 Oct 2026 lists torch 2.14.1 (30 Sep 2026) as the latest, while the latest torchaudio is still 2.11.0. — [torch on PyPI](https://pypi.org/project/torch/); [torchaudio on PyPI](https://pypi.org/project/torchaudio/)

**MMS_FA bundle facts (from the 2.11 source)**
- Weights URL: `dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt`, about 1.18 GB.
- `_sample_rate=16000`, normalised waveform.
- Labels are `('-', 'a', 'i', 'e', 'n', 'o', 'u', 't', 's', 'r', 'm', 'k', 'l', 'd', 'g', 'h', 'y', 'b', 'p', 'w', 'c', 'v', 'j', 'z', 'f', "'", 'q', 'x', '*')`: blank `-`, 27 lowercase Latin letters and apostrophe, plus the star `*`.
- The bundle has "no token for word boundary (like `|`)".
- Licence: CC-BY-NC 4.0.
- The docstring says "31K hours … 1,130 languages", but the tutorial text says "23,000 hours of audio from 1100+ languages". The two sources are inconsistent. — [torchaudio 2.11.0 on PyPI](https://pypi.org/project/torchaudio/2.11.0/); [tutorial source](https://github.com/pytorch/audio/blob/main/examples/tutorials/forced_alignment_for_multilingual_data_tutorial.py)
- `forced_align(log_probs (B,T,C), targets (B,L), input_lengths, target_lengths, blank=0)`:
  - It supports only `batch_size==1`.
  - It requires T ≥ L + N_repeat.
  - It raises an error if the targets contain the blank index. — [torchaudio 2.11.0 on PyPI](https://pypi.org/project/torchaudio/2.11.0/)

**Tutorial recipe** — [tutorial source](https://github.com/pytorch/audio/blob/main/examples/tutorials/forced_alignment_for_multilingual_data_tutorial.py)
- Loading: `from torchaudio.pipelines import MMS_FA as bundle; model = bundle.get_model(); tokenizer = bundle.get_tokenizer(); aligner = bundle.get_aligner()`.
- Alignment: `emission,_ = model(waveform); token_spans = aligner(emission[0], tokenizer(transcript))`, where `transcript` is a list of words.
- Converting frames to seconds: `ratio = waveform.size(1)/emission.size(1)/sample_rate`, then `t = span.start*ratio`.
- Romanisation: the tutorial uses the legacy Perl `uroman.pl`, then normalises with `lower(); replace("’","'"); re.sub("([^a-z' ])"," ",text)`.
- The tutorial warns that numbers like "1882" are not romanised and get deleted ("one needs to romanize numbers, but this is known to be a non-trivial task").
- The `<star>` dimension is included by default and can be disabled with `with_star=False`.

**uroman**
- `uroman` is now a Python package: `pip install uroman`, v1.3.1 (Jun 2024; PyPI shows 1.3.1.1). The last Perl version is 1.2.8.
- API: `ur.Uroman().romanize_string(s, lcode='hin')`. CLI: `python3 -m uroman -l hin -i in.txt`.
- It converts native-script digits to Western Arabic digits. It does not spell numbers out. — [uroman on PyPI](https://pypi.org/project/uroman/); [isi-nlp/uroman](https://github.com/isi-nlp/uroman)
- Local test with uroman 1.3.1.1 (this sandbox):

  | Devanagari | uroman output |
  |---|---|
  | राहुल | raahul |
  | शर्मा | sharmaa |
  | पाँच | paamc |
  | करोड़ | karodd |
  | बैंक | baimk |
  | अकाउंट | akaaumtt |
  | फ़ोन | fon |
  | ढाई | ddhaaii |
  | डेढ़ | ddeddh |
  | 98765 | 98765 |

  The digit string 98765 is deleted by the tutorial normaliser. So digits must be written out as words in the alignment transcript.

**ctc-forced-aligner (MahmoudAshraf97)**
- Install from GitHub: `pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git`. FFmpeg is required.
- The GitHub releases page is empty.
- API: `load_alignment_model(device, dtype)`, `load_audio`, `generate_emissions(model, waveform, batch_size=16)`, `preprocess_text(text, romanize=True, language="hin")`, `get_alignments`, `get_spans`, `postprocess_results`. It returns JSON segments with `start`, `end` and `text`.
- CLI: `ctc-forced-aligner --audio_path a.wav --text_path t.txt --language hin --romanize`, with options `--split_size word` and `--star_frequency edges|segment`.
- Default model: `MahmoudAshraf/mms-300m-1130-forced-aligner`, licensed CC-BY-NC 4.0. — [MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
- Its `requirements.txt` is `uroman, nltk, torch, torchcodec, transformers>=4.34, Unidecode`.
- It builds its own pybind11 C++ `forced_align_impl.cpp`, so it does not depend on torchaudio's `forced_align`.
- `preprocess_text` uses `Uroman().romanize_string(..., lcode=iso)` and inserts `<star>` tokens. — [pyproject/requirements/text_utils on GitHub](https://github.com/MahmoudAshraf97/ctc-forced-aligner/blob/main/requirements.txt)
- **Pitfall:** `pip install ctc-forced-aligner` installs a different project. PyPI 1.0.2 (9 Feb 2025) is authored by "Deskpai.com" (url `github.com/deskpai/ctc_forced_aligner`), is ONNX-runtime based, and depends on `librosa, numpy, onnxruntime`. — [ctc-forced-aligner on PyPI](https://pypi.org/project/ctc-forced-aligner/1.0.2/); [deskpai/ctc_forced_aligner](https://github.com/deskpai/ctc_forced_aligner)

**Decoding audio under torchaudio ≥ 2.9**
- `torchcodec` needs system FFmpeg 4–9.
- `silero-vad[audio]` pins `torchaudio>=0.12.0,<2.10`. — [silero-vad on PyPI](https://pypi.org/project/silero-vad/)
- This sandbox has `ffmpeg 6.1.1-3ubuntu5` and no `sox` binary (local check).

### Inferences
**Conversion commands** (standard ffmpeg/sox usage; not from a fetched page)
```bash
# phone recording (m4a/aac/opus) -> canonical 24 kHz mono s16 (matches §4)
ffmpeg -i raw/clip01.m4a -ac 1 -ar 24000 -sample_fmt s16 -c:a pcm_s16le audio24k/clip01.wav
# 16 kHz copy for MMS_FA / Silero
ffmpeg -i audio24k/clip01.wav -ar 16000 -c:a pcm_s16le audio16k/clip01.wav
# sox equivalent (sox: apt-get install sox)
sox raw/clip01.wav -b 16 -c 1 -r 24000 audio24k/clip01.wav
```

**Aligner choice**
- For explainability in the interview, a ~40-line script on `torchaudio.pipelines.MMS_FA` plus the `uroman` Python package is easier to defend than `ctc-forced-aligner`. Pin `torch==2.11.*`, `torchaudio==2.11.0` and `torchcodec`, or load audio with `soundfile` and pass a tensor, which removes the TorchCodec/FFmpeg coupling.
- Keep `ctc-forced-aligner` (installed from git) as a cross-check: it handles long audio via windowing (`window_size` 30 s), which 20–40 s clips do not need.

**Transcript handling**
- Keep two parallel transcripts:
  - the Level-1 verbatim Devanagari/Latin transcript (the deliverable);
  - an "alignment view" in which each token maps 1:1 to a display token. Digits are expanded to Hindi words (or to the digits as actually spoken). Latin-script English words stay as they are; uroman leaves Latin text unchanged, and the alignment model's alphabet is Latin. Devanagari-written English loanwords (बैंक → baimk) are romanised by uroman.
- Romanising a word can produce an empty string, for example digits, or punctuation such as "।". Drop such words before alignment or map them to `*`, then re-insert them with interpolated times. This keeps word indices aligned with the display transcript.
- Alignment times come in 20 ms emission frames, because the wav2vec2 stride is 320 samples at 16 kHz. Boundary resolution can therefore never be better than about ±10 ms.

### Gaps
- docs.pytorch.org could not be fetched, so the rendered tutorial page for 2.11 is unconfirmed. The GitHub source still carries the "deprecated in 2.8 / removed in 2.9" banner, and I could not confirm whether it was edited after 2.10.
- No published MMS_FA word-boundary error figure for Hindi or code-mixed speech was found (see Q2).

---

## Q2. Part D: hand-checking 30 boundaries (Audacity/Praat) and how to report alignment error

### Takeaway
Mark boundaries in an Audacity label track, export the tab-separated `start\tend\label` file, and compare 30 aligner boundaries (word starts and ends) against the hand marks. Report the mean and median absolute error and the share within 20/50/100 ms tolerances. Literature figures are given as accuracy at a tolerance on English (TIMIT/Buckeye), not Hindi, so the 30-boundary check is the only Hindi evidence.

### Cited Findings
- Audacity label export: one line per label, "start and end times in seconds", the label text in column 3, tab-separated. Point labels have start = end. With multiple label tracks, the uppermost track comes first. Use UTF-8 for non-ASCII labels such as Devanagari. — [Audacity manual: Importing and Exporting Labels](https://manual.audacityteam.org/man/importing_and_exporting_labels.html); [crowsetta Audacity format](https://crowsetta.readthedocs.io/en/latest/formats/seq/aud-seq.html)
- An extended "spectral" format adds frequency ranges on an extra line. Search results only gave the section heading, so verify the format before parsing. — [Audacity manual: Export Other](https://manual.audacityteam.org/man/file_menu_export_other.html)
- `praatio` 6.2.2 (Jan 2026) reads and writes Praat TextGrids, if Praat is preferred. — [praatio on PyPI](https://pypi.org/project/praatio/)
- Reported accuracy levels (English):
  - CTC with label priors reduces word/phone boundary error by 12–40% over standard CTC. It is on par with MFA on Buckeye but behind MFA on TIMIT. — [Huang et al. 2024, arXiv 2406.02560](https://arxiv.org/html/2406.02560v2)
  - MFA reaches 72.8% (TIMIT) and 69.9% (Buckeye) word-boundary accuracy at a 20 ms tolerance. — [BFA, arXiv 2509.23147](https://arxiv.org/html/2509.23147v1)
  - A 2026 preprint compares against MMS-CTC at 10 ms and 100 ms tolerances; I could not see its numbers. — [Multilingual Word Aligner, emergentmind summary](https://www.emergentmind.com/papers/2606.10675)

### Inferences
**Protocol**
- Choose 30 boundaries spread across the 6 clips (5 per clip). Over-sample the hard cases:
  - English–Hindi switch points;
  - word pairs inside amounts (पाँच | करोड़);
  - phone-number digit groups;
  - the self-correction ("…नहीं नहीं…");
  - words next to pauses.
- Mark the boundaries blind: do not import the aligner's times first, to avoid anchoring bias.

**Metrics**
- Per boundary: e = t_aligner − t_hand (signed).
- Report mean |e|, median |e|, mean signed error (bias), and %|e| ≤ 20 / 50 / 100 ms.
- Report word starts and word ends separately. CTC peaky behaviour usually makes ends less reliable, and the label-priors paper's motivation implies the same.
- Give a bootstrap 95% CI. With n = 30 the CI will be wide; say so.

**Fixed limits**
- Frame quantisation of 20 ms gives a floor of about ±10 ms.
- The hand-marking tolerance is roughly ±10–20 ms (an inference). Treat sub-20 ms differences as noise.

**Parsing the Audacity export**
`[l.split('\t') for l in open(f, encoding='utf-8') if not l.startswith('\\')]`. The skip condition ignores spectral lines.

### Gaps
- I could not open the IndicVoices paper's Level-1 rules in detail, because arxiv was unreachable. A search summary says Level 1 is verbatim "as spoken", phonetically faithful spelling (e.g. "muje" kept, not "mujhe"), and Level 2 is standardised. — [IndicVoices arXiv 2403.01926](https://arxiv.org/html/2403.01926v1)
- The tag inventory (noise, laughter, English-word handling) was not confirmed. Another researcher covering IndicVoices should supply it.
- No Hindi MMS_FA boundary-error benchmark was found.

---

## Q3. Part D translation: IndicTrans2 / IndicTransToolkit (CPU), or an LLM; literal vs fluent

### Takeaway
IndicTrans2 Indic→En comes as a 1B model and a distilled 200M model (`ai4bharat/indictrans2-indic-en-dist-200M`). Both are MIT-licensed and run via HF transformers with `trust_remote_code=True` and `IndicProcessor` from `IndicTransToolkit` (v1.1.1, Python ≥ 3.10). The 200M model is the CPU-friendly choice for the "fluent" version. A literal and a fluent version from one MT model are not available by prompt, so use the LLM for the literal (and optionally the fluent) rendering, and use IndicTrans2 as the open baseline.

### Cited Findings
- Model checkpoints are MIT-licensed. Indic-En comes as `ai4bharat/indictrans2-indic-en-1B` and the distilled `ai4bharat/indictrans2-indic-en-dist-200M`. CTranslate2 (CT2) conversions exist "only for efficient inference". — [AI4Bharat/IndicTrans2 README](https://github.com/AI4Bharat/IndicTrans2); [HF interface README](https://github.com/AI4Bharat/IndicTrans2/tree/main/huggingface_interface)
- `pip install indictranstoolkit`. The authors recommend `numpy>=2.1`, `torch>=2.5` and `transformers>=4.51`. It is Linux/macOS only.
- Usage: `ip = IndicProcessor(inference=True)`, then `ip.preprocess_batch(sents, src_lang=..., tgt_lang=...)`, then the tokenizer, then `model.generate(num_beams=5, max_length=256)`, then `ip.postprocess_batch(outputs, lang=tgt)`. If `from IndicTransToolkit import IndicProcessor` fails, use `from IndicTransToolkit.IndicTransToolkit import IndicProcessor`.
- The README example shows the toolkit preserving `9876543210` and an email address through translation.
- For Indic→En the language codes are `src_lang="hin_Deva"` and `tgt_lang="eng_Latn"`. — [IndicTransToolkit on PyPI](https://pypi.org/project/IndicTransToolkit/); [VarunGumma/IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit)

### Inferences
**CPU cost** (back-of-envelope from parameter counts)
- 200M parameters in fp32 is about 0.8 GB RAM.
- 1B in fp32 is about 4–5 GB; fp16 on CPU is slow or unsupported for some ops.
- For six clips (~30 sentences) either model is fine on CPU. Use 200M if the interview laptop is weak.

**Literal vs fluent**
- Literal: preserve Hindi word order where English grammar allows. Keep every disfluency and the self-correction ("five lakh… no no, five crore"). Keep amounts in the speaker's units: "five crore", not "50 million".
- Fluent: idiomatic English. The corrected value only ("five crore rupees"). Numbers may be digits.
- Prompt the LLM with both instructions and a JSON schema: `{"literal": ..., "fluent": ...}`.
- Store both in `translations.jsonl` with fields: `clip`, `sent_id`, `hi_L1`, `en_literal`, `en_fluent`, `system`.

**IndicTrans2 sentence handling**
- Feed sentences, not whole clips; IndicTrans2 was trained at sentence level.
- Code-mixed Latin-script English words in the Hindi input are a risk. Try both the raw L1 text and a version with English words transliterated to Devanagari, and report the difference (see Gaps).
- The checker (Q4) should be run on both systems' output. That gives a natural accuracy comparison to report.

### Gaps
- huggingface.co was unreachable, so the model cards (exact sizes, gating, any `flash_attention` notes) were not read directly.
- No source found on how IndicTrans2 handles Latin-script English words embedded in Hindi input, or whether it converts "पाँच लाख" to "500,000" vs "five lakh". This needs testing on the clips.

---

## Q4. Part D entity checker: Hindi/English number parsing, phone numbers, names, negation, self-correction; accuracy reporting

### Takeaway
Write the checker as plain Python: a Hindi number-word table plus a small Indian-numbering English parser, regexes for digit strings, and fuzzy name matching on normalised romanisations (uroman + rapidfuzz). Score it against hand-labelled gold entity lists per sentence. Do not rely on `word2number` for lakh/crore: it silently returns wrong values.

### Cited Findings
- `word2number` 1.1 is the latest release and dates from 2017. Its source contains an `indian_number_system` dict with `'lakh': 100000` and `'crore': 10000000`, but the dict sits inside a string or docstring; only `american_number_system` is used. Local test results:

  | Input | Returned |
  |---|---|
  | "five lakh" | 5 |
  | "twenty five crore" | 25 |
  | "two point five million" | 2 |
  | "nine eight seven six" | 85 |
  | "fifty thousand" | 50000 (correct) |

  — [word2number on PyPI](https://pypi.org/project/word2number/) (sdist source)
- `text2num` 3.1.0 covers French, Spanish, English, Portuguese, German, Dutch, Danish and Italian. Hindi and Indian numbering are not listed. `indic-numtowords` 1.1.0 goes number→words only, the wrong direction for parsing. — [text2num](https://pypi.org/project/text2num/); [indic-numtowords](https://pypi.org/project/indic-numtowords/)
- `uroman` romanisations of Devanagari names (local test): राहुल → raahul, शर्मा → sharmaa, अग्रवाल → agravaal. Plain `rapidfuzz.fuzz.ratio` against the English spellings gives 91, 92 and 67. After a simple normalisation (lowercase, w→v, sh→s, ph→f, collapse doubled letters) the scores rise to 100, 100 and 86. — [uroman](https://pypi.org/project/uroman/); [rapidfuzz 3.14.6](https://pypi.org/project/rapidfuzz/)
- `indic-transliteration` (2.3.82) does no schwa deletion (राहुल → ITRANS `rAhula`), so uroman is the better romaniser for matching. — [indic-transliteration](https://pypi.org/project/indic-transliteration/) (local test)
- `ai4bharat-transliteration` (IndicXlit), 1.1.3 from Sep 2022, depends on `fairseq`, whose last release is 0.12.2 (Jun 2022). Treat it as a heavy, stale dependency. — [ai4bharat-transliteration](https://pypi.org/project/ai4bharat-transliteration/); [fairseq](https://pypi.org/project/fairseq/)

### Inferences
**Hindi number-word table**
This is linguistic knowledge, not a fetched source; verify the spellings, including nukta and chandrabindu variants.
- 0–99 must be listed one by one, because Hindi 11–99 are irregular: ग्यारह 11, …, उन्नीस 19, बीस 20, इक्कीस 21, …, उनतीस 29, तीस 30, …, निन्यानवे 99.
- Multipliers: सौ 100, हज़ार/हजार 1,000, लाख 100,000, करोड़ 10,000,000, अरब 10^9.
- Fractional prefixes:
  - डेढ़ = 1.5; ढाई = 2.5 (alone they mean 1.5 or 2.5; with a multiplier, e.g. डेढ़ लाख = 150,000);
  - साढ़े X = X + 0.5 (साढ़े तीन लाख = 350,000);
  - सवा X = X + 0.25;
  - पौने X = X − 0.25.
- Also include Latin-script and loanword forms as spoken in code-mix: "lakh/lac", "crore", "thousand", "hundred", "million". Include digits mixed with words: "5 लाख", "2.5 crore".

**Parser**
- Tokenise, then process left to right with `current` and `total`:
  - a unit word (0–99) sets `current`;
  - सौ multiplies `current`;
  - हज़ार, लाख or करोड़ adds `current × mult` to `total`.
- Fractional prefixes modify the next number or multiplier.
- Use the same algorithm for English, with the Indian multipliers (lakh, crore) added to thousand, million and billion. Accept "2.5 crore", "25 lakh", "five hundred thousand" and comma-separated digits ("5,00,000" Indian grouping, and "500,000").
- Comparison should be value-based: Hindi "पाँच करोड़" = 5e7 matches English "five crore", "50 million" or "₹5 crore". A unit-preservation sub-score can separately flag "50 million" if the policy is to keep lakh/crore.

**Phone and account numbers**
- Normalise the spoken digit sequence to a digit string. Hindi digit words (शून्य/ज़ीरो, एक … नौ) and English ones ("nine eight…") are mapped one by one.
- Handle the pair or double patterns: "double five" → 55, "डबल पाँच" → 55, "triple" → three of the next digit. Grouped pairs ("अट्ठानवे सत्तासी" = 98 87) are parsed as 2-digit numbers.
- Compare exact strings. Report the per-digit edit distance as a secondary score.

**Names**
- Romanise the gold Hindi name with uroman, then normalise as above.
- Find the best `rapidfuzz.fuzz.partial_ratio` window over the English output tokens. Accept if ≥ 85 (tune the threshold on the 6 clips and report it).
- Latin-script names in the transcript match directly.

**Negation**
- Hindi cues: नहीं, मत, न, ना, बिल्कुल नहीं, कभी नहीं, नहीं तो (careful: "otherwise"), ना कि.
- Pitfall: न/ना also appear as tag-question particles ("चलोगे ना?"), which are not negations. Handle this with a rule (sentence-final ना with ? is a tag) or with gold labels.
- English cues: not, n't, no, never, neither, nor, none, nothing, without, "instead of" (for ना कि).
- Check per gold negation event: the English sentence must contain a negation cue within the clause aligned to the negated verb. A simple version: the count of negation cues per sentence must be ≥ the gold count.
- Self-correction "नहीं नहीं" is a repair marker, not a propositional negation. Gold labels must tag it separately, or it will inflate negation false positives.

**Self-correction rule**
- The corrected value must appear in the fluent English. The abandoned value must not appear in the fluent version; in the literal version it may appear, but only before the repair marker ("no, no" / "sorry" / "I mean").
- Score two flags: `corrected_present` and `abandoned_leaked`.

**Reporting**
- Hand-label gold entities per sentence: type, Hindi span and normalised value. Hand-label a "checker verdict truth" for each (entity, English output) pair, i.e. whether the translation actually conveys the entity correctly.
- The checker is a classifier of "entity preserved: yes/no". Report its accuracy, precision and recall against the human verdict, per entity type, as a confusion matrix.
- Report separately the translation systems' entity accuracy as measured by the checker, for literal vs fluent and LLM vs IndicTrans2.
- Expected failure cases to look for and report:
  - "fifty lakh" vs "5 million" when value-equivalence is off;
  - fractional words (साढ़े/सवा/पौने) mistranslated, e.g. "quarter to three lakh";
  - phone numbers chunked differently ("98765 43210" vs "nine eight seven…");
  - names translated literally (e.g. a name that is also a common noun: "Kamal" → "lotus");
  - double negation and tag-question ना;
  - negation scope ("I didn't say five lakh" vs "I said not five lakh");
  - English output that hedges or omits the abandoned value.

### Gaps
- No existing open-source Hindi words-to-number parser with lakh/crore and fractions (डेढ़/ढाई/साढ़े/सवा/पौने) was found and verified. Plan to write one (~80 lines) with unit tests.
- No validated threshold for transliteration fuzzy matching on Hindi names was found in a primary source. The 85 threshold above is a proposal from a 5-name local test.

---

## Q5. Part E-ML: simulating wait-k on word timings and plotting entity accuracy vs latency

### Takeaway
Simulate rather than stream. Use the Part D Hindi word end times as the source timeline. For each k, emit English word i once k + i Hindi words have been heard (STACL), on a playback timeline where each word starts at max(ready time, end of the previous word). Compute AL/LAAL in seconds with SimulEval's formulas, and score entity accuracy with the Q4 checker. Run two variants, oracle time-shift and real prefix translation, because only the latter shows anticipation errors on entities.

### Cited Findings
- Under wait-k, English word i is spoken once k + i Hindi words have been heard. Each English word gets a fixed duration or a duration measured from TTS audio. The run is scored with the Part D checker. — `Inter IIT Tech Meet 15.0_ Sarvam.ai Mini PS.pdf` (local, Part E ML getting-started)
- The playback rule: each frame starts at max(arrival time, end of the previous frame), and speech after the speaker stops counts toward latency. — `Sarvam.pdf` §5.1 (local)
- SimulEval's `ALScorer` code:
  - `gamma = target_length / source_length`;
  - sums `d - t_minus_1 / gamma` over target tokens until the first `d >= source_length` (that index is τ), then divides by τ;
  - returns `delays[0]` if the first delay already exceeds the source length;
  - by default `target_length` is the reference length (`use_ref_len=True`).
- `LAALScorer` is the same but with `gamma = max(len(delays), target_length) / source_length`. Speech instances measure delays in ms of source consumed; "computation-aware" variants use elapsed wall time. — [SimulEval latency_scorer.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)
- SimulEval's last PyPI release is 1.1.4 (Nov 2023), so the toolkit is stale. — [simuleval on PyPI](https://pypi.org/project/simuleval/)
- SimulEval also has StartOffset/EndOffset and ATD scorers. ATD assumes 300 ms per source word for speech input. — [SimulEval latency_scorer.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py)

### Inferences
**Simulation** (~60 lines, `wait_k_sim.py`)
- Inputs:
  - `src = [(w_j, start_j, end_j)]` from the aligner, in seconds;
  - `X = valid duration` (the end of the last Hindi word, or the clip duration);
  - `tgt_words` from the literal or fluent translation;
  - per-word durations `dur_i`, either fixed (e.g. 0.35 s, or a per-syllable estimate) or measured from TTS.
- Schedule:
  - `ready_i = end_{min(k+i-1, J)}` (1-indexed: word i needs k + i − 1 Hindi words when i starts at 1, so the first word needs k Hindi words). Once the source is finished, all remaining words are ready at `X`.
  - Add an optional compute delay δ (e.g. 0.2 s): `ready_i += δ`.
  - `start_1 = ready_1`, and `start_i = max(ready_i, start_{i-1} + dur_{i-1})`.
- Latency on the playback timeline:
  - `AL_s` = SimulEval AL with d_i = start_i (seconds), |X| = X, |Y| = reference word count;
  - `LAAL_s` with max(|hyp|, |ref|);
  - also report `start_offset = start_1` and `end_offset = start_N + dur_N − X`. The end offset matters because the PS counts "how soon it finishes after the speaker stops".
- Note: this uses d_i = when the word is heard (playback). Text-mode SimulEval uses d_i = source consumed. State which one is used; the playback version is closer to the official KPI.

**Two translation regimes**
1. **Oracle (shift-only).** Use the full-sentence translation and only re-time it. Entity accuracy is then constant across k, and only latency changes. This isolates the latency side of the trade-off.
2. **Prefix translation (realistic).** For each emission step i, translate the Hindi prefix `w_1 … w_{k+i-1}` with an LLM or IndicTrans2 and commit only English word i (STACL-style prefix-to-prefix; earlier committed words are forced as a prefix):
   - LLM: give the committed English so far and ask for the single next word;
   - IndicTrans2: use HF `generate` with `decoder_input_ids` / `prefix_allowed_tokens_fn`, or a simpler re-translate-and-take-the-i-th-word approach (approximation; describe it honestly).
   This exposes the Hindi SOV problem: verbs and negation (नहीं before the verb, the verb at the end) and amounts with a following unit (पाँच … करोड़) arrive late. Small k commits "five lakh" before the repair, or drops the negation.
   - Cost: about N_tgt × |K| calls per sentence, which is affordable for 6 clips.
   - Cache outputs to JSON so the interview demo runs offline.

**Outputs**
- k ∈ {1, 2, 3, 5, 7, ∞}, where ∞ means full sentence.
- Plot `matplotlib` entity accuracy (y) vs AL_s (x), one point per k, annotated "k = …". Add separate lines per entity type (amount, phone, name, negation) or per regime.
- Also plot a per-clip timeline: Hindi words on one row, English words on another, for the self-correction clip at k = 2 vs k = 5.

**Interpretation the plot should support: where to wait**
- Entity errors should concentrate where the "unit" or "repair" token comes after the number: पाँच [लाख], नहीं नहीं, फ़ोन digits.
- That argues for adaptive waiting: hold before emitting numbers or negated verbs until the multiplier or verb is heard, and stay low-k elsewhere. This connects to the Part A question.

### Gaps
- Whether the official harness's Latency KPI uses AL, LAAL, StartOffset/EndOffset or ATD is not stated in `Sarvam.pdf` §7. The PS only says "How far the English speech trails the speaker, and how soon it finishes after the speaker stops".
- No maintained, ready-made SimulEval agent for Hindi→English speech wait-k on word timings was found. SimulEval itself has not been released since 2023.

---

## Q6. Part E-Dev: WebSocket echo server per §5.1, test client, Dockerfile, and what makes it fall behind

### Takeaway
Use `websockets` 17.x (asyncio implementation, `websockets.asyncio.server.serve`, Python ≥ 3.11), with a `process_request` hook returning `connection.respond(HTTPStatus.OK, "OK\n")` for `/health`. Use separate reader and sender tasks connected by an `asyncio.Queue`, schedule output on `loop.time()` deadlines (not cumulative `sleep`), turn off `permessage-deflate` for raw PCM, and keep CPU work off the event loop. FastAPI + uvicorn is an equally valid alternative.

### Cited Findings
**websockets versions**
- 17.2 was released 3 Oct 2026, and 17.0 (29 Jul 2026) requires Python ≥ 3.11.
- 16.0 (Jan 2026) required ≥ 3.10.
- 14.0 (Nov 2024) made the new asyncio implementation the default and deprecated the legacy one (`websockets.legacy.*`).
- 13.0 (Aug 2024) introduced the new implementation and deprecated the `handler(ws, path)` signature in favour of `handler(ws)` with `ws.request.path`.
- 17.0 also: `process_request` now also runs for non-GET or HTTP/1.0 requests (server replies 405/505 rather than closing); `send(text=...)` and similar are keyword-only; a trio implementation was added. — [websockets changelog](https://github.com/python-websockets/websockets/blob/main/docs/project/changelog.rst); [websockets 17.2 on PyPI](https://pypi.org/project/websockets/17.2/)

**Health check**
- Official FAQ pattern: `def health_check(connection, request): if request.path == "/healthz": return connection.respond(HTTPStatus.OK, "OK\n")`, then `await serve(echo, host, port, process_request=health_check)` and `await server.serve_forever()`.
- websockets says it is "not an HTTP server". It offers "limited support for returning HTTP responses with the `process_request` hook". — [health_check_server.py](https://github.com/python-websockets/websockets/blob/main/example/faq/health_check_server.py); [FAQ server.rst](https://github.com/python-websockets/websockets/blob/main/docs/faq/server.rst)

**`serve()` defaults in 17.2** (wheel source) — [websockets 17.2 on PyPI](https://pypi.org/project/websockets/17.2/)

| Parameter | Default |
|---|---|
| `compression` | `"deflate"` (permessage-deflate on) |
| `open_timeout` | 10 |
| `ping_interval` | 20 |
| `ping_timeout` | 20 |
| `max_size` | 2**20 (1 MiB incoming message cap) |
| `max_queue` | 16 frames (receive high-water mark) |
| `write_limit` | 2**15 bytes (write-buffer high-water mark) |

**TCP_NODELAY**
- CPython asyncio sets `TCP_NODELAY` on TCP transports by default (`base_events._set_nodelay`, called from selector transports). Nagle's algorithm is therefore not normally a problem for asyncio servers. — [CPython asyncio base_events.py](https://github.com/python/cpython/blob/3.13/Lib/asyncio/base_events.py); [selector_events.py](https://github.com/python/cpython/blob/3.13/Lib/asyncio/selector_events.py)

**uvicorn alternative** — [uvicorn 0.54.0 on PyPI](https://pypi.org/project/uvicorn/0.54.0/) (wheel source); [fastapi on PyPI](https://pypi.org/project/fastapi/)
- Versions: uvicorn 0.54.0 (Sep 2026), FastAPI 0.143.1.
- Defaults: `ws_max_size=16 MiB`, `ws_ping_interval=20.0`, `ws_ping_timeout=20.0`, `ws_per_message_deflate=True`.
- `--ws auto` picks the `websockets-sansio` implementation when websockets is installed. The explicit `--ws websockets` implementation still imports `websockets.legacy.server`, which is deprecated.

### Inferences
**Server skeleton** (`server.py`, ~100 lines)
```python
import asyncio, json, time
from http import HTTPStatus
from websockets.asyncio.server import serve

SR, BPS, FRAME = 24000, 2, 3840          # 80 ms @ 24 kHz s16le
DELAY_S = 1.0                             # echo delay

def health(conn, req):
    if req.path == "/health":
        return conn.respond(HTTPStatus.OK, "OK\n")
    if req.path != "/ws":
        return conn.respond(HTTPStatus.NOT_FOUND, "not found\n")

async def handler(ws):
    loop = asyncio.get_running_loop()
    start = json.loads(await ws.recv())            # session_start
    q: asyncio.Queue = asyncio.Queue()             # (due_time, bytes) or None
    t0 = loop.time()
    async def reader():
        n = 0
        async for msg in ws:                       # never blocks on sending
            if isinstance(msg, bytes):
                q.put_nowait((t0 + n*0.08 + DELAY_S, msg)); n += 1
            else:
                m = json.loads(msg)
                if m["type"] == "input_end":
                    valid = m["valid_samples"]*BPS  # trim echo to real samples
                    q.put_nowait(("END", valid))
                    # keep draining silence frames, do not echo them
                    async for _ in ws: pass
                    return
    async def sender():
        sent = 0; limit = None
        while True:
            item = await q.get()
            if item[0] == "END":
                limit = item[1]; break
            due, pcm = item
            await asyncio.sleep(max(0.0, due - loop.time()))   # absolute deadline, no drift
            await ws.send(pcm); sent += len(pcm)                # <=1 s per frame
        await ws.send(json.dumps({"type": "session_complete", "session_id": start["session_id"]}))
    r = asyncio.create_task(reader())
    await sender()                                  # nothing is sent after session_complete
    await r  # or cancel; closing early fails the session, so let the client close
```
- The handler is served with `serve(handler, "0.0.0.0", 8000, process_request=health, compression=None, max_size=2**20)`.
- The skeleton has a known wrinkle for the interview: frames queued before `input_end` that lie past `valid_samples` should be trimmed. Track the byte count and cut the last echoed frame.
- Sketch caveat (not a fetched source): the sender reaches `session_complete` only after reading the END marker, so the reader must not exit before queueing it.

**Test client** (`client.py`)
- Read a 24 kHz s16 WAV and split it into 3,840-byte frames, zero-padding the last.
- Send `session_start`. Send frame k at `t0 + k*0.08` using absolute deadlines. After the last frame, immediately send `input_end` with `valid_samples`. Then send zero frames every 80 ms until `session_complete`.
- A concurrent receive task timestamps each binary frame on arrival and builds the playback timeline: `start = max(arrival, prev_end)`, `prev_end = start + len/2/24000`.
- Report:
  - start offset;
  - measured echo delay distribution (expected ≈ DELAY_S);
  - end offset after `input_end`;
  - gaps (underruns);
  - max lateness of client sends;
  - assertions on the protocol rules: one `session_complete`, nothing after it, server frames ≤ 1 s (≤ 48,000 bytes), completion within 30 s.
- Add a "stress" flag that injects `time.sleep(0.2)` (blocking) into the server handler to demonstrate falling behind.

**Dockerfile**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir websockets==17.2
COPY server.py .
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["python", "server.py"]
```
`python:3.12-slim` satisfies websockets 17's Python ≥ 3.11 requirement. The image has no curl, so the HEALTHCHECK uses Python's urllib.

**What makes the server fall behind, and how to prevent it**
| Cause | Prevention |
|---|---|
| Blocking CPU work (model step, numpy, `time.sleep`) on the event-loop thread stops both reader and sender. The harness then cannot deliver frames within 1 s and the session fails (§5.1). | Use `await loop.run_in_executor(...)` or `asyncio.to_thread`, a separate inference process, or a model step < 80 ms. Measure real-time factor. |
| Sending and reading in one coroutine (await send, then recv): backpressure on send stalls reading. | Use separate reader and sender tasks. |
| Cumulative `sleep(0.08)` drift and scheduling jitter. | Use absolute deadlines from `loop.time()` / `time.monotonic()`. |
| Unbounded queues hide lag until memory grows; latency grows without bound. | Monitor queue depth and lateness. Use a bounded queue with an explicit drop or skip policy for outputs. Log lateness per frame. |
| permessage-deflate (on by default) burns CPU on incompressible PCM. | `compression=None` in `serve()`, `--ws-per-message-deflate false` in uvicorn. |
| Python GC pauses and large allocations. | Pre-allocate buffers, avoid per-frame big objects, optionally `gc.freeze()` after warm-up. |
| `write_limit` and TCP send-buffer backpressure: `await ws.send` blocks when the client reads slowly. | Keep the sender separate. Do not send giant frames; ≤ 1 s per frame is also a §5.1 rule. |
| Model warm-up or weight load on first connection (the 10 s accept limit). | Load weights at start-up, before `/health` returns 200. |
| Ping or keepalive timeouts during long CPU stalls. | Keep the loop free; ping_interval/timeout of 20 s are fine. |
| Container CPU limits or thread oversubscription (torch intra-op threads). | `torch.set_num_threads(n)`, `OMP_NUM_THREADS`. |

### Gaps
- websockets' own docs (readthedocs) were unreachable. The changelog and examples were read from GitHub instead; the asyncio `ServerConnection.respond` signature was taken from the FAQ example, not the API reference.
- I found no official statement of the harness's exact behaviour on TCP backpressure ("cannot be delivered within 1 s … because your server is not reading") beyond the §5.1 rule text.

---

## Q7. Part E-SP: causal front-end on 80 ms frames (resampling, denoising, VAD), telephone and noisy copies, SI-SDR, and when denoising hurts

### Takeaway
A frame loop of 80 ms (1,920 samples at 24 kHz):
- causal resampling with `soxr.ResampleStream` (or `scipy.signal.resample_poly` with carried filter state);
- a classical STFT Wiener or spectral-subtraction denoiser with a running noise estimate, overlap-add with a 20 ms window and 10 ms hop inside the frame;
- Silero VAD on a 16 kHz ring buffer (512-sample chunks);
- DeepFilterNet3 or RNNoise at 48 kHz as the neural comparison.
SI-SDR must be computed after compensating each method's algorithmic delay. Practical traps: DeepFilterNet 0.5.6 is stale and only has binary wheels up to Python 3.11, and its I/O imports a torchaudio module that no longer exists in torchaudio ≥ 2.9.

### Cited Findings
**Resampling**
- `soxr` 1.1.0 (May 2026): `rs = soxr.ResampleStream(in_rate, out_rate, 1, dtype='float32')`, then `rs.resample_chunk(x, last=eof)` per chunk. "Output frame count may not be consistent. This is normal operation." — [soxr on PyPI](https://pypi.org/project/soxr/); [python-soxr](https://github.com/dofuuz/python-soxr)

**Silero VAD**
- `silero-vad` 6.2.3 (Sep 2026), MIT licence, supports 8 kHz and 16 kHz. Processing one 30+ ms chunk takes less than 1 ms on one CPU thread.
- The model call requires exactly 512 samples at 16 kHz or 256 at 8 kHz (32 ms), and keeps 64 or 32 samples of context internally.
- `VADIterator(model, threshold=0.5, sampling_rate=16000, min_silence_duration_ms=100, speech_pad_ms=30)` is the streaming wrapper.
- `get_speech_timestamps` naively decimates (`audio[::step]`) only when the rate is a multiple of 16 kHz. Other rates raise an error, so 24 kHz must be resampled first.
- Install with `pip install silero-vad` plus `onnxruntime` if desired. — [silero-vad on PyPI](https://pypi.org/project/silero-vad/) (README and wheel source); [snakers4/silero-vad](https://github.com/snakers4/silero-vad)

**DeepFilterNet**
- Usage: `pip install deepfilternet`, `model, df_state, _ = init_df()`, `enhance(model, df_state, audio)`. Audio must be 48 kHz. The `deep-filter` Rust binary and a LADSPA plugin exist for real-time use. The `-D/--compensate-delay` option pads to offset the STFT and lookahead delay. Dual MIT/Apache-2.0 licence. — [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- DeepFilterNet3: "48 kHz … 20 ms windows with a hop size of 10 ms … a look-ahead of 2 frames … overall latency of 40 ms". A Core ML streaming port counts 30 ms, so the figure depends on accounting. — [DeepFilterNet3 paper, arXiv 2305.08227](https://arxiv.org/pdf/2305.08227)
- Packaging (inspection on 14 Oct 2026):
  - `deepfilternet` 0.5.6 (31 Aug 2023) is the latest release. It requires `numpy>=1.22,<2.0` and `packaging<24`, which conflicts with IndicTransToolkit's recommended `numpy>=2.1`.
  - Its `DeepFilterLib==0.5.6` dependency has wheels only for cp38–cp311. On Python ≥ 3.12, pip would build from sdist (Rust).
  - `df/io.py` does `from torchaudio.backend.common import AudioMetaData`. In torchaudio 2.8 that import still works through a deprecation shim. The torchaudio 2.11 wheel has no `torchaudio/backend` package at all, so `from df.enhance import …` (which imports `df.io`) will fail there. — [deepfilternet 0.5.6](https://pypi.org/project/deepfilternet/0.5.6/); [DeepFilterLib 0.5.6 files](https://pypi.org/project/DeepFilterLib/0.5.6/#files); [torchaudio 2.8.0](https://pypi.org/project/torchaudio/2.8.0/) (wheel source)
  - Hugging Face Spaces show this class of breakage: a "moved" warning for `torchaudio.backend.common.AudioMetaData`, and `torch._six` errors in older DeepFilterNet versions. — [HF Space hshr/DeepFilterNet](https://huggingface.co/spaces/hshr/DeepFilterNet); [PyTorch forum](https://discuss.pytorch.org/t/no-module-named-torchaudio-backend-common/186937)

**RNNoise**
- `pyrnnoise` 0.4.5 (Sep 2026): `RNNoise(sample_rate=48000)`, `denoise_frame(frame)` on 480-sample (10 ms at 48 kHz) frames, and `denoise_chunk(chunk)` yields `(speech_prob, denoised_frame)`. — [pyrnnoise on PyPI](https://pypi.org/project/pyrnnoise/)

**noisereduce**
- `noisereduce` 3.0.3 (Oct 2024) offers stationary and non-stationary spectral gating. The non-stationary path imports `scipy.signal.filtfilt` (zero-phase, i.e. non-causal) and time-smooths the STFT. It processes `chunk_size` blocks with padding, so it is not a causal per-frame denoiser as shipped. — [noisereduce on PyPI](https://pypi.org/project/noisereduce/) (wheel source); [timsainb/noisereduce](https://github.com/timsainb/noisereduce)

**SI-SDR**
- `torchmetrics` 1.9.0 `scale_invariant_signal_distortion_ratio(preds, target, zero_mean=False)` computes:
  - α = ⟨ŝ, s⟩ / ‖s‖²;
  - SI-SDR = 10·log10(‖αs‖² / ‖αs − ŝ‖²).
  Note that `zero_mean` defaults to False. — [torchmetrics sdr.py](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/audio/sdr.py); [torchmetrics on PyPI](https://pypi.org/project/torchmetrics/)
- The SI-SDR definition comes from Le Roux et al., "SDR – half-baked or well done?" (ICASSP 2019). — [arXiv 1811.02508](https://arxiv.org/abs/1811.02508)

**When denoising hurts**
- Iwamoto et al. (Interspeech 2022) decompose enhancement error into noise and artifact components by orthogonal projection. Artifact errors have a larger impact on ASR degradation than residual noise. "Observation adding" (mixing the noisy input back into the enhanced signal) increases the signal-to-artifact ratio and improves ASR. — [arXiv 2201.06685](https://arxiv.org/abs/2201.06685); [ISCA archive](https://www.isca-archive.org/interspeech_2022/iwamoto22_interspeech.html)
- Araki et al. (Interspeech 2023) find that artifacts also strongly affect human intelligibility. — [ISCA archive](https://www.isca-archive.org/interspeech_2023/araki23_interspeech.html)

**Local ffmpeg**
- ffmpeg 6.1.1 (Ubuntu build) in this sandbox has `pcm_mulaw`, `pcm_alaw`, `libgsm` and `libopus` encoders, but no AMR-NB encoder (the native `amrnb` decoder only; local `ffmpeg -encoders` check).

### Inferences
**Telephone-band and noisy copies** (standard commands; verify `ffmpeg -encoders` on the target machine)
```bash
# G.711 mu-law narrowband (300-3400 Hz band-limit, 8 kHz), then back to 24 kHz for comparison
ffmpeg -i clean.wav -af "highpass=f=300,lowpass=f=3400" -ar 8000 -c:a pcm_mulaw tel.wav
ffmpeg -i tel.wav -ar 24000 -c:a pcm_s16le tel24k.wav
# sox equivalent: sox clean.wav -r 8000 -e mu-law tel.wav sinc 300-3400
# GSM full-rate (libgsm present in Ubuntu ffmpeg)
ffmpeg -i clean.wav -ar 8000 -ac 1 -c:a libgsm tel.gsm && ffmpeg -i tel.gsm -ar 24000 tel_gsm24k.wav
# AMR-NB only if `ffmpeg -encoders | grep amr` shows libopencore_amrnb
# Low-bitrate VoIP: ffmpeg -i clean.wav -c:a libopus -b:a 8k -ar 16000 -application voip voip.opus
```
- Noise mixing in numpy at SNR ∈ {20, 10, 5, 0} dB: `noise *= rms(s)/(rms(noise)*10**(snr/20))`.
- Self-recorded noise (fan, traffic, kitchen, TV babble) has no licensing issues.
- Licences of MUSAN, DEMAND and ESC-50 were not verified here. ESC-50 is generally described as CC BY-NC; verify before redistributing clips in the repo.

**Causal 80 ms frame loop**
- Use a 20 ms window (480 samples at 24 kHz) with a 10 ms hop and a sqrt-Hann analysis/synthesis window, giving perfect reconstruction for 50% overlap.
- Each 80 ms input frame gives 8 hops. Keep a 10 ms carry-over buffer between frames. Output is delayed by one window (≈ 10–20 ms algorithmic latency).
- Noise PSD estimate:
  - initialise from the first 0.3 s, or from frames where VAD says non-speech;
  - update with recursive averaging, λ ≈ 0.98, only on non-speech frames; or use minimum statistics over a ~1.5 s sliding minimum.
- Gains:
  - Wiener: G = max(ξ/(1+ξ), G_min), with decision-directed a-priori SNR ξ (α = 0.98) and G_min ≈ 0.1 (−20 dB floor);
  - spectral subtraction: |Ŝ|² = max(|Y|² − β·N̂, floor·|Y|²).
  - The floor prevents musical noise.
- Sample-rate plan:
  - keep 24 kHz for the denoiser;
  - resample to 16 kHz for Silero (24k → 16k is ratio 2/3; `soxr.ResampleStream(24000, 16000, 1)`). 80 ms is 1,280 samples at 16 kHz, which is 2.5 Silero chunks, so use a ring buffer and emit VAD decisions every 512 samples;
  - resample to 48 kHz for DeepFilterNet/RNNoise (24k → 48k is ×2; 80 ms = 3,840 samples = 8 RNNoise frames or 8 DeepFilterNet hops, which fits neatly).
- Silence trimming must be causal: hold-off and hangover (e.g. `min_silence_duration_ms` 200–300, `speech_pad_ms` 30–100). Never trim inside words. In a streaming system, trimming only adds value as "don't send leading silence"; it cannot remove silence that has not yet been heard.

**DeepFilterNet environment**
- Practical option: a separate Python 3.11 venv with torch/torchaudio ≤ 2.8 and numpy < 2, used only for the DeepFilterNet comparison.
- Alternatives: the pre-built `deep-filter` binary (48 kHz WAV, `-D` delay compensation), or a 3-line shim module that provides `torchaudio.backend.common.AudioMetaData`.
- The model is causal apart from its 2-frame lookahead. Running `enhance()` on the whole file and then accounting for the 40 ms delay should therefore match frame-by-frame processing up to edge effects. I have not verified this numerically; test it by comparing against the binary on one clip.

**SI-SDR protocol**
- References: the clean 24 kHz recording vs (a) the noisy input, (b) Wiener output, (c) DeepFilterNet output, (d) RNNoise output.
- Before scoring, estimate the integer lag by cross-correlation (search ±100 ms) and remove it. Otherwise causal delays of 10–40 ms collapse SI-SDR to near 0 dB or below.
- Use `zero_mean=True`, following Le Roux et al.'s recommendation (torchmetrics defaults to False).
- Report ΔSI-SDR (improvement) per SNR and per condition.
- For the telephone copies, a full-band clean reference penalises the missing band. Use the band-limited clean signal (same filter and codec chain, no noise) as the reference for "denoising on telephone audio", or report both.

**Finding a case where denoising hurts** (try these, one will very likely show it)
1. **Clean or high-SNR input (≥ 20–30 dB):** the denoiser removes breathy or fricative parts (स/श/फ़, aspirated stops), so SI-SDR drops relative to the unprocessed input. Iwamoto's artifact argument predicts ASR harm too.
2. **Telephone μ-law or GSM band-limited input:** codec quantisation noise is speech-correlated. Neural denoisers trained at 48 kHz full band can hallucinate or suppress, and the classical noise estimate locks onto codec noise.
3. **Non-stationary babble or TV speech:** the denoiser keeps the interfering voice or garbles the target, and low-energy word-final sounds and quiet numbers are suppressed. This connects to entity accuracy: a quiet "सवा" or "नहीं" lost.
4. **Musical noise from spectral subtraction** at 0–5 dB SNR without a floor.
- Show it with SI-SDR and, if time allows, a downstream proxy: a Whisper or IndicConformer ASR WER on noisy vs denoised audio. This is optional because ASR is offline analysis, not part of the system.
- Mitigation to report: observation adding (output = enhanced + 0.1–0.3 × noisy) per Iwamoto, an SNR-gated bypass (denoise only when the estimated SNR < 15 dB), and gain floors.

### Gaps
- MUSAN, DEMAND and ESC-50 licence terms were not verified (sites unreachable or not searched within budget).
- No source found confirming that DeepFilterNet's Python package exposes a true frame-by-frame stateful API in 0.5.6. Streaming is documented for the Rust binary and LADSPA plugin only.
- RNNoise's internal algorithmic latency was not confirmed from a primary source. RNNoise is commonly described as 10 ms frames plus a small lookahead.

---

## Q8. Claims that turned out wrong or misleading when checked (for the report's "one wrong claim")

### Takeaway
Four checkable claims failed verification. The best one for the report is that the torchaudio multilingual forced-alignment tutorial's banner says its APIs "will be removed in 2.9", while torchaudio 2.10/2.11 keep `forced_align` and `MMS_FA`. Useful runners-up: `word2number` appears to support lakh/crore but silently returns 5 for "five lakh", and `pip install ctc-forced-aligner` installs a different author's package.

### Cited Findings
- The tutorial source warns: "The APIs described in this tutorial are deprecated in 2.8 and will be removed in 2.9." — [tutorial source](https://github.com/pytorch/audio/blob/main/examples/tutorials/forced_alignment_for_multilingual_data_tutorial.py)
  - The 2.10 release notes say `forced_align` is "preserved and will remain in torchaudio", and the 2.11.0 wheel still exports `functional.forced_align` and `pipelines.MMS_FA`. — [torchaudio releases](https://github.com/pytorch/audio/releases); [torchaudio 2.11.0 wheel](https://pypi.org/project/torchaudio/2.11.0/)
- `word2number`'s source lists `'lakh': 100000, 'crore': 10000000`, but that dict is never used. `w2n.word_to_num("five lakh")` returns 5, and "nine eight seven six" returns 85. — [word2number 1.1](https://pypi.org/project/word2number/)
- The `ctc-forced-aligner` README tells users to install from GitHub. The PyPI name `ctc-forced-aligner` 1.0.2 belongs to Deskpai's ONNX fork with a different API surface. — [GitHub README](https://github.com/MahmoudAshraf97/ctc-forced-aligner); [PyPI 1.0.2](https://pypi.org/project/ctc-forced-aligner/1.0.2/)
- The MMS_FA training-data size is stated inconsistently: the tutorial says "23,000 hours of audio from 1100+ languages", the torchaudio bundle docstring says "31K hours of data in 1,130 languages". — [tutorial source](https://github.com/pytorch/audio/blob/main/examples/tutorials/forced_alignment_for_multilingual_data_tutorial.py); [torchaudio 2.11.0](https://pypi.org/project/torchaudio/2.11.0/)
- The DeepFilterNet README lists PyTorch ">=1.9" as the requirement. In practice the 0.5.6 package fails to import with torchaudio ≥ 2.9 (removed `torchaudio.backend`) and has no Python 3.12+ wheels for its Rust core. — [DeepFilterNet README](https://github.com/Rikorose/DeepFilterNet); [DeepFilterLib files](https://pypi.org/project/DeepFilterLib/0.5.6/#files)

### Inferences
- Pin exact versions in `requirements.txt` and run a smoke test (`python -c "import torchaudio; torchaudio.pipelines.MMS_FA"`) in CI or Docker. Version drift is the biggest risk in the "run on a new clip during the interview" requirement.
- Suggested pins (all verified as current on PyPI, 14 Oct 2026):

  | Group | Pins |
  |---|---|
  | Main environment (Python 3.11 or 3.12) | `torch` (2.11.x to match torchaudio 2.11.0 safely), `torchaudio==2.11.0`, `torchcodec`, `uroman==1.3.1.1`, `rapidfuzz==3.14.6`, `soxr==1.1.0`, `silero-vad==6.2.3`, `onnxruntime`, `torchmetrics==1.9.0`, `soundfile`, `numpy`, `scipy`, `matplotlib` |
  | Dev track | `websockets==17.2` |
  | Translation | `IndicTransToolkit==1.1.1`, `transformers` |
  | DeepFilterNet (separate Python 3.11 env) | `deepfilternet==0.5.6`, `torch/torchaudio<=2.8`, `numpy<2` |

**Suggested repo layout**
```
clips/raw, clips/24k, clips/16k
transcripts/L1/*.txt (+ align_view)
align/align_mms.py
align/hand_labels/*.txt (Audacity)
align/eval_boundaries.py
translate/translate_llm.py, translate/translate_it2.py, translations.jsonl
checker/hindi_numbers.py, checker/en_numbers.py, checker/entities.py, checker/gold/*.json, checker/eval_checker.py
partE_{ml,dev,sp}/…
RESULTS.md (each number + command)
```

### Gaps
- Whether the GitHub tutorial banner has been updated after 2.10 in the rendered docs is unknown, because docs.pytorch.org was unreachable. The claim above is accurate for the tutorial source on `main` as fetched on 14 Oct 2026.
