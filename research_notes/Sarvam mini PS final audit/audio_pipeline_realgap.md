# Audio pipeline validity: channel simulation, sample rate, denoising, synthetic–real gap, audio-quality KPI

Scope: audit of the audio side of "/home/user/interiit_minips/reports/Sarvam mini PS research.md" (§C4, §C5, §C6, §E-SP) against the PS (`/home/user/interiit_minips/Sarvam.pdf`) and four claims from an external critique. As of 2026-10-14.

Verification labels: **[V]** primary source read directly, or package wheel inspected. **[L]** local experiment in this container (ffmpeg 6.1 + numpy; commands given). **[S]** search-engine snippet or secondary summary only. arxiv.org, isca-archive.org, kyutai.org, proceedings.mlr.press and openslr.org were blocked by the proxy, so every paper-level claim below is [S] unless marked otherwise. **[I]** my inference.

PS facts relied on [V, `pdftotext -layout Sarvam.pdf`]:
- "Mono 16-bit PCM at 24 kHz, in both directions. Lower-bandwidth recordings are upsampled."
- "Background noise and telephone audio may occur."
- The hidden test is "Real Hindi recordings as in §4".
- Allowed at inference: "audio pre-processing, such as resampling, gain, denoising and voice-activity detection", and "post-processing of the output audio that does not change its words, such as loudness normalisation or silence trimming".
- "Augmentation applied during training adds no hours."
- "Only your Part A dataset may be used for training."
- "All final KPIs are measured on your system's English speech." Audio quality and naturalness (6 points) is "How clean and natural the English speech sounds, as rated by listeners."
- English targets must be "bulbul:v3, voice shubh, language_code="en-IN", 24 kHz. Pace and temperature are free, but declare them."

## Q1. Telephone-channel simulation and sample-rate handling (critique point 1)

### Takeaway
Critique point 1 is **factually wrong about the report**. §C4 already specifies the chain "24 kHz → 8 kHz, 300–3400 Hz band-pass, G.711 μ-law/A-law, GSM, AMR-NB 4.75–12.2 kbps or Opus 6–12 kbps, 1–5% packet loss, then back to 24 kHz", and §E-SP repeats it as an ffmpeg command.

The real gaps are different:
- there is no **wideband (16 kHz: G.722, AMR-WB, Opus-WB) branch**;
- one fixed 300 Hz low edge is applied to every call path, whereas mobile paths have a ~100 Hz edge;
- the report misses a **bandwidth shortcut**: most real sources are 16 kHz (empty above 8 kHz after upsampling), while TTS sources are full-band.

Measurement shows that the 8 kHz resample already imposes the 3.4–4 kHz upper edge. The low edge is not implied by any codec, so a gentle, randomised 2nd-order high-pass is the right addition, not a brick-wall 300–3400 Hz filter.

### Cited Findings
- **The report already has the chain.** §C4 gives the telephone chain quoted above and cites `phonesim` and a stock ffmpeg with "no AMR-NB encoder". §E-SP gives `ffmpeg -af "highpass=f=300,lowpass=f=3400" -ar 8000 -c:a pcm_mulaw`, then back to 24 kHz, plus GSM and low-bitrate Opus. Neither section has a 16 kHz wideband branch. — [report §C4/§E-SP, local file](file:///home/user/interiit_minips/reports/Sarvam%20mini%20PS%20research.md) [V]
- **phonesim profiles and band edges.** All profiles "start and end at 24 kHz by default". "Each profile resamples to 8 or 16 kHz on entry and to the output rate on exit." The band edges are "a 2nd-order low edge (250–320 Hz on the analogue loop, ~100 Hz on the digital narrowband path, 50–80 Hz on the wideband ones) and a roll-off from the codec's passband to the channel Nyquist". The profiles are:
  - `pstn_narrowband`: G.711, ~300 Hz low edge, roll-off from 3.4 to 4 kHz;
  - `pstn_g726`;
  - `voip_opus_wideband`: roll-off from ~7 kHz, erasures up to 5%;
  - `voip_g722_wideband`;
  - `voip_to_cellular_wideband`: Opus → AMR-WB;
  - `voip_to_cellular_narrowband`: Opus → AMR-NB, `bitrate` 12.2k down to 4.75k, ~100 Hz low edge;
  - `stress_multi_transcode`.
  
  Erasure rates: "up to 1% of frames on a managed trunk or a radio leg… up to 5% on the public Internet"; clock drift "±50 ppm". — [phonesim README](https://github.com/deepmark/phonesim) [V, raw README read]
- **phonesim codec coverage and install.** "A stock distro ffmpeg cannot run AMR". AMR needs a build with libopencore-amr and libvo-amrwbenc, which makes the ffmpeg build LGPLv3 or GPLv3. "EVS has no open encoder and ffmpeg has no G.729 encoder, so neither is offered". Opus erasures need the libopus shared library. Install is `pip install phonesim` (core: numpy, torch, soundfile), and the code is MIT. The latest PyPI version is **0.2.0**. — [phonesim README](https://github.com/deepmark/phonesim) [V]; `pip index versions phonesim` [L]
- **Local ffmpeg 6.1 encoders.** It has pcm_mulaw, pcm_alaw, libgsm and libgsm_ms, g722, g726, libopus/opus, libspeex and libcodec2. It has **no AMR-NB or AMR-WB**. — `ffmpeg -encoders` [L]
- **What the 8 kHz resample and each codec do to a white-noise probe.** Band power is relative to 1–2 kHz, measured after 24 kHz → codec → 24 kHz:

  | chain | 20–100 Hz | 100–300 Hz | 3.4–3.8 kHz | 3.8–4 kHz | 4–6 kHz | > 6.5 kHz |
  |---|---|---|---|---|---|---|
  | G.711 μ-law at 8 kHz | +0.1 dB | +0.1 dB | −1.6 dB | −9.3 dB | −24.5 dB | ≈ −88 dB (digital silence) |
  | GSM-FR | +0.7 dB | +0.8 dB | −2.1 dB | −8.0 dB | −23.0 dB | ≈ −86 dB |
  | Opus 8 kbps voip | −2.1 dB | +0.5 dB | −1.8 dB | −5.0 dB | −15.6 dB | ≈ −83 dB |
  | G.722 at 16 kHz | flat | flat | flat | flat | flat to 7.5 kHz (−1.6 dB) | −24 dB at 8–12 kHz |
  | Report's `highpass=f=300,lowpass=f=3400` + μ-law | −24.7 dB | −6.6 dB | — | — | — | — |

  So the **upper edge comes for free** from the 8 kHz resample, but **no codec removes energy below 300 Hz**. — `ffmpeg -f lavfi anoisesrc …; -ar 8000 -c:a pcm_mulaw|libgsm|libopus -b:a 8k; -ar 24000`, with a Welch-style PSD in numpy [L]
- **torchaudio 2.11 has no codec path left.** `apply_codec` and `torchaudio.io` (`AudioEffector`, `StreamReader`/`StreamWriter`) are absent from the wheel. The wheel still has `functional.resample`, `bandpass_biquad`, `lowpass_biquad`, `mu_law_encoding`, `add_noise` and `speed`. — torchaudio 2.11.0 wheel, grep of `torchaudio/` [V]
- **SpeechBrain's codec augmentation is broken on torchaudio ≥ 2.9.** `CodecAugment` in SpeechBrain 1.1.1 calls `torchaudio.io.AudioEffector` with formats `("wav","pcm_mulaw"), ("mp3",None), ("g722",None)`. That class no longer exists, and SpeechBrain only pins `torchaudio>=2.1.0`. — speechbrain 1.1.1 wheel, `speechbrain/augment/codec.py` [V]
- **audiomentations has no telephone codecs.** Version 0.43.1 has `BandPassFilter`, `HighPassFilter`, `LowPassFilter`, `Resample`, `Mp3Compression`, `BitCrush`, `Aliasing`, `AddBackgroundNoise`, `AddShortNoises`, `ApplyImpulseResponse`, `RoomSimulator`, `Gain`, `Limiter`, `ClippingDistortion` and `TimeStretch`. It has no G.711, AMR or GSM codec. Its own dependencies include `soxr` and `librosa`. — audiomentations 0.43.1 wheel [V]
- **Evidence that codec augmentation helps.** Encoding training audio with AMR-NB and G.711 gave a clear gain on 8 kHz test data for Mandarin E2E ASR, with negligible change on 16 kHz data. — [arXiv 2010.12715](https://arxiv.org/pdf/2010.12715) [S, from the prior research note]
- **The channel gap in Hindi.** Every system's WER on Gramvaani telephone-quality Hindi is 2–3× its clean Kathbath WER (IndicWhisper 26.8 vs 10.3). — [Vistaar](https://github.com/AI4Bharat/vistaar) [V, via the prior note]
- **Source sample rates.** MUCS 2021 Hi-En (SLR104) is "16 kHz, 16-bit". The IndicVoices README tells users to "downsample the audios to 16kHz", but its native rate was not confirmed. FLEURS is 16 kHz (not verified). Common Voice is 48 kHz mp3 (not verified). — [data_sources_licences.md (prior note)](file:///home/user/interiit_minips/research_notes/Sarvam%20mini%20PS%20research/data_sources_licences.md) [V/S as marked there]

### Inferences
- [I] **What the model actually sees.** "Lower-bandwidth recordings are upsampled" means the model receives 24 kHz PCM whose spectrum above 4 kHz (narrowband) or 8 kHz (wideband) is near-digital-silence, with a resampler-specific transition band. Sarvam's resampler is unknown. Augmentation should reproduce exactly this and **randomise the resampler kernel**: soxr HQ/VHQ, ffmpeg swr default, and scipy `resample_poly`. That stops the model from keying on one filter edge.
- [I] **Do not downsample to 8 kHz at inference.** The input is already 24 kHz, so feed it unchanged. No bandwidth extension is needed, and none is allowed as a "cascade" risk. Resampling is needed only for side modules, such as 16 kHz Silero VAD.
- [I] **The bandwidth shortcut (new, not in the report).** Real buckets R1–R3 (IndicVoices, MUCS, FLEURS: about 122 of the 125 real hours) are ≤ 8 kHz bandwidth after upsampling. The synthetic buckets (Parler/IndicF5/Bulbul-hi at 24–44.1 kHz) are full-band to 12 kHz. Band-presence above 8 kHz is therefore a near-perfect "synthetic" label. The model can use it to treat the two pools differently, for example learning entity behaviour only on full-band input.
  - The fix is to low-pass synthetic sources at a random 4–8 kHz with p ≈ 0.7. Alternatively, store all Part A audio band-limited to 8 kHz (a 16 kHz-equivalent bandwidth) before upsampling to 24 kHz.
  - The cheapest version: synthesize, resample to 16 kHz, then upsample to 24 kHz for a share of S1/S2. This is a dataset-level choice, so make it before the mid-term.
- [I] **The 300–3400 Hz band-pass is right for PSTN, too aggressive for mobile.** Use a 2nd-order high-pass with fc ~ U(250, 320) Hz on analogue-PSTN draws, ~100 Hz on mobile-NB draws and 50–80 Hz on wideband draws, matching phonesim. The low-pass is redundant with the 8 kHz resample, so keep it only as a random extra roll-off.
- [I] **Easiest correct implementation:** `phonesim` profiles cover the realistic paths, with the noise injected *before* the encoder, as on a handset. A stock ffmpeg covers G.711, G.722, G.726, Opus and GSM. AMR needs the static ffmpeg build named in the phonesim README. Do not use SpeechBrain `CodecAugment` or `torchaudio.io` with torchaudio 2.11.
- [I] **Timing check.** Word timings are labels, so every channel chain must be delay-free or delay-compensated. phonesim states that its AMR and Opus decode is "delay-compensated". For hand-rolled ffmpeg chains, measure the lag by cross-correlation against the input once per codec and shift. A fixed 10–30 ms codec lag moves every Hindi word time.

### Gaps
- I did not measure codec delay for each ffmpeg chain, or AMR behaviour (no AMR encoder locally).
- No source quantifies the gain from telephone augmentation for Hindi S2ST, or for Mimi-token models.
- The native sample rates of IndicVoices and FLEURS are still unconfirmed.

## Q2. Denoising, "observation adding" and codec-token robustness (critique point 2)

### Takeaway
Critique point 2 is **already in the report**: §E-SP lists "output = enhanced + 0.1–0.3 × noisy" citing Iwamoto et al. Iwamoto et al. do not prescribe a fixed "10–20%". They show that the signal-to-artifact ratio (SAR) rises monotonically with the observation weight, and that WER improves with it. Later work tunes or learns the weight on validation data.

The more important point for this system: the safest default is **no denoising at inference**. Train on noise- and codec-augmented sources with clean targets instead. Keep an SNR-gated denoiser plus observation adding (OA) only if a real, speaker-disjoint, noisy Hindi dev set shows a gain on the final KPIs. No evidence was found on Mimi's robustness to noise.

### Cited Findings
- **Iwamoto et al. 2022.** Kazuma Iwamoto, Tsubasa Ochiai, Marc Delcroix et al., "How Bad Are Artifacts?: Analyzing the Impact of Speech Enhancement Errors on ASR", Interspeech 2022.
  - They decompose enhancement errors into noise and artifact components by orthogonal projection, and find that the artifact component hurts ASR more.
  - OA "can monotonically increase the signal-to-artifact ratio under a mild condition": SDR and SNR fall as the weight rises, SAR rises and WER improves. This holds on both simulated and real recordings.
  
  — [arXiv 2201.06685](https://arxiv.org/abs/2201.06685); [ISCA](https://www.isca-archive.org/interspeech_2022/iwamoto22_interspeech.html) [S: snippets only; the exact weights or SNRs tested and the datasets (likely CHiME-4) were not retrieved]
- **Follow-ups (all [S]).**
  - Ochiai, Iwamoto et al. 2024 propose the AB-SDR training objective, and report that both OA and AB-SDR reduce artifact errors and improve ASR. — [arXiv 2404.14860](https://arxiv.org/html/2404.14860v1)
  - A study of how end-to-end ASR training affects SE artifacts. — [arXiv 2311.11599](https://arxiv.org/pdf/2311.11599)
  - A trained "bridging module" chooses the OA coefficient. It was trained on simulated data, and the authors propose training it on real CHiME-4 noisy speech. — [arXiv 2501.02452](https://arxiv.org/html/2501.02452v1)
  - OA as a convex combination of noisy and enhanced audio, with the weight tuned on validation, used with Whisper back-ends. — [arXiv 2509.21833](https://arxiv.org/pdf/2509.21833)
  - Training-free OA. — [arXiv 2602.20967](https://arxiv.org/abs/2602.20967)
  - SAM-Audio pre-processing *raised* Whisper WER and CER relative to raw noisy input despite cleaner-sounding output, and the harm grew with Whisper size. — [arXiv 2603.04710](https://arxiv.org/html/2603.04710v1)
- **Neural codecs under noise.** Codecs differ notably in how well they preserve phonetic content (ASR-WER), speaker (ASV-EER) and emotion cues under ambient noise, white noise and reverberation; DAC at its highest bitrate degraded least. Mimi is not named in the snippet. — [Tseng & Harwath, arXiv 2505.24248](https://arxiv.org/pdf/2505.24248) [S]
- "Even slight input noise perturbations can cause unintended shifts in quantized codewords" (EnCodec, WavTokenizer). — [arXiv 2509.19025](https://awesomepapers.io/speech-audio/papers/2509.19025) [S]
- Most neural audio codecs are trained on clean or studio speech, and token shift under noise has not been studied systematically. — [arXiv 2608.31037](https://arxiv.org/pdf/2608.31037) [S]
- **DeepFilterNet3 is stale on PyPI.** It is still at `deepfilternet` 0.5.6 (Aug 2023). The report notes its wheels stop at cp311 and that it imports `torchaudio.backend`, which is absent in 2.11. Its paper gives "overall latency of 40 ms" at 48 kHz. — `pip index versions` [L]; [report §E-SP](file:///home/user/interiit_minips/reports/Sarvam%20mini%20PS%20research.md); [arXiv 2305.08227](https://arxiv.org/pdf/2305.08227) [S]

### Inferences
- [I] **Why "no denoise" is the safe default.** The base checkpoint and our LoRA see Mimi-style tokens of the raw input. A denoiser at inference creates a train/test mismatch unless training inputs were denoised the same way. It also adds 10–40 ms of algorithmic delay, which counts against latency. And Iwamoto's result says its artifacts, more than residual noise, are what harm recognition.
- [I] **If denoising is used, apply the same front-end at training time on a share of samples**, or train with OA-processed inputs. Gate it by estimated SNR (for example, denoise only below ~10–15 dB). Use OA in convex form, `y = (1−w)·enh + w·noisy`, with w tuned on the real noisy dev set over {0.1, 0.2, 0.3, 0.5} rather than fixed. The report's additive "+0.1–0.3 × noisy" form changes level, so renormalise the gain afterwards.
- [I] **Source augmentation also helps the audio-quality KPI.** Pairing noisy or telephone *source* audio with clean Bulbul *targets* teaches the model not to carry input noise or channel colouring into its output speech. In a Hibiki-like model the output is generated from tokens conditioned on the input stream, so this is plausible, but no source tests it.
- [I] Neither the 10–20% (critique) nor the 0.1–0.3 (report) figure is a published recommendation, as far as the snippets show. Both should be labelled "tuned on dev".

### Gaps
- Iwamoto's exact OA weights or SNRs and WER numbers were not retrieved (arXiv and ISCA blocked).
- No evidence was found on Mimi specifically under noise or telephone input.
- No study was found on denoising before a speech-LM or S2ST front-end.
- I could not confirm whether Hibiki or Moshi applied source-side noise augmentation; their paper was blocked.

## Q3. Synthetic–real gap for TTS source speech, and what drives the audio-quality KPI (critique point 3)

### Takeaway
Critique point 3 **misattributes the mechanism**.
- The audio-quality KPI is rated on the English output. The output voice is fixed by the Bulbul v3 `shubh` targets, so flat *input* (Hindi TTS) prosody does not directly lower it.
- Flat or synthetic input prosody harms **generalisation to real Hindi**: translation, intent and entity accuracy, and possibly latency behaviour. Intonation and stress also carry questions and corrections, which feeds intent accuracy.
- The report's 62% real share is the main mitigation and is consistent with the literature.

What does drive the audio-quality KPI is mostly on the **target side and at decode time**:
- Bulbul pace and temperature;
- hard cuts from silence insertion;
- English alignment errors;
- codec-token artefacts and sampling temperature;
- stalls or underruns in the stream;
- loudness and clipping.

### Cited Findings
- **Hibiki data and inference.** Hibiki "rel[ies] on synthetic data generation". Its alignment rule ("a word should only appear in the target once it's predictable from the source") "is applied either by inserting silences or by synthesizing targets with a voice controlled, alignment-aware TTS". "A larger [CFG] coefficient will increase voice similarity, but excessive coefficients can lead to worse translations". `--cfg-coef` defaults to 1, and "a typical value to use is 3". — [hibiki README](https://github.com/kyutai-labs/hibiki) [V]
- Hibiki's paper says silence insertion can lead to "hard cuts" when timestamps are inaccurate or there is no pause between words. That motivates alignment-aware TTS re-synthesis. — [arXiv 2502.03382 via search summary](https://pith.science/paper/2502.03382); [ResearchGate](https://www.researchgate.net/publication/388754494_High-Fidelity_Simultaneous_Speech-To-Speech_Translation) [S]
- One secondary summary reports Hibiki short-form Fr→En at 39.2 ASR-BLEU and 0.41 speaker similarity, against Seamless at 37.0 and 0.30. — [pith.science](https://pith.science/paper/2502.03382) [S, unverified]
- Hibiki-Zero (ICML 2026) drops word-level alignment and uses supervised learning followed by RL for latency, reporting gains in naturalness and speaker identity. — [arXiv 2602.11072](https://arxiv.org/html/2602.11072) [S]
- **Synthetic source speech in S2ST.** The Translatotron 2 / CVSS line warns that synthetic source input "may bias the trained model towards synthetic speech input but not generalize well to human speech", and suggests jointly fine-tuning on a mix of real and synthetic data. — [arXiv 2203.13339](https://arxiv.org/pdf/2203.13339) [S]
- A project README for an all-synthetic S2ST model reports a text score falling from ~25 to ~8 chrF++ on real human speech. — [tiny-aya-translate README](https://tiny-aya-translate-readme.static.hf.space/README.md) [S, non-peer-reviewed, indicative only]
- **The ASR literature on the gap.**
  - ASR trained only on synthetic speech has markedly higher error on real speech even when it sounds natural. The cause is missing acoustic and prosodic variability; speaker and prosody conditioning narrows the gap but plateaus. — [Edinburgh thesis](https://era.ed.ac.uk/handle/1842/44684) [S]
  - LLM dialogues voiced by conversational TTS beat classical TTS for Whisper fine-tuning but did not match real in-domain data. — [arXiv 2408.09215](https://arxiv.org/abs/2408.09215) [S, from the report]
  - Speed perturbation 0.9/1.0/1.1 gave a 4.3% relative WER gain. — [Ko et al. 2015](https://www.danielpovey.com/files/2015_interspeech_augmentation.pdf) [S, from the report]
- **What the PS permits on output.** The PS allows output post-processing "that does not change its words, such as loudness normalisation or silence trimming". Server audio frames are at most 48,000 bytes, which is 1 s of 24 kHz s16. — [PS §3, §5](file:///home/user/interiit_minips/Sarvam.pdf) [V]

### Inferences
- [I] **Input prosody versus the KPI.** Flat input prosody affects the audio-quality KPI only *indirectly*: a model confused by real prosody may stall, insert long silences or produce garbled tokens. The report should say "flat TTS prosody widens the synthetic–real gap, hurting translation, intent, entity and latency KPIs", not "degrades audio quality".
- [I] **Mitigations already in the report:** a real majority; many permissioned voices; disfluency injection; noise, RIR and codec augmentation.
- [I] **Mitigations to add on the input side:**
  1. Randomise the TTS speaking rate (0.85–1.15) and temperature *per utterance*, and prefer expressive or conversational TTS settings (Indic Parler-TTS descriptions with varied pitch and pace).
  2. Apply the bandwidth shortcut fix from Q1.
  3. Do not use speed perturbation in its usual form. In S2ST with timed targets, speed perturbation of the source must rescale the Hindi word times *and* the English placement, otherwise the alignment rule is broken. Either apply tempo change before computing timings (offline, at data-build time) or skip it.
- [I] **Audio-quality drivers on the target and decode side, in likely order of impact.** None is quantified for this task.
  1. **Silence-insertion hard cuts.** Insert silence only at English word boundaries where MFA or MMS shows a natural gap of at least ~50–100 ms, or at punctuation. Apply 5–10 ms fade-out and fade-in ramps. Otherwise synthesize the English *chunk-wise* with Bulbul at placement boundaries, which still satisfies the Bulbul-only rule, so that each chunk has natural onsets.
  2. **Bulbul settings.** Pace ≈ 1.0–1.1 and moderate temperature. Declare them and screen targets with UTMOS or DNSMOS plus a round-trip ASR check.
  3. **English alignment accuracy.** A misplaced boundary cuts a phoneme.
  4. **Decoding.** Audio-token temperature and top-k: low temperature gives buzz or repetition, high gives babble.
  5. **Streaming health.** Underruns appear as gaps on the playback timeline; keep inference under 80 ms per step.
  6. **Output post-processing.**
     - Causal fixed or slow gain to roughly −16 to −23 LUFS integrated, with a peak limiter.
     - Trim only leading and trailing silence; do not trim inside the stream, because that changes timing.
     - Do not time-stretch: it adds artefacts and the latency metric sees the shifted timing.
  7. CFG: relevant only if the Sarvam base exposes speaker-similarity conditioning like Hibiki. With a single fixed voice it is probably moot, but test whether a CFG > 1 improves `shubh` consistency.

### Gaps
- I could not retrieve Hibiki's human-evaluation MOS for naturalness, the effect of silence insertion versus alignment-aware TTS on MOS, or the CFG sweep table (paper blocked).
- No measured S2ST gap exists for TTS-only *source* speech in Hindi.
- No source links input-side prosody to output naturalness ratings.

## Q4. Noise and RIR datasets for augmentation: licences, and whether external noise is allowed at all

### Takeaway
Licence-clean options exist: MUSAN (CC BY 4.0 at corpus level), DNS-Challenge noise (only its Freesound-CC0 and AudioSet parts are permissive; DEMAND within it is CC BY-SA 3.0), FSD50K filtered to CC0 and CC-BY clips, the ESC-10 subset (CC BY), and OpenSLR 26/28 RIRs (Apache 2.0). Full ESC-50 is CC BY-NC, and WHAM!'s licence is unverified; avoid both for a model delivered to a company.

The bigger issue is the PS rule "Only your Part A dataset may be used for training". It makes **any external noise corpus a grey area**, babble made of other people's speech (MUSAN speech) most of all. The safest sources are:
- noise harvested from VAD-negative regions of the Part A real recordings themselves;
- self-recorded noise;
- generated noise and simulated RIRs (pyroomacoustics).

Ask the organisers before using MUSAN or DNS.

### Cited Findings
- **DNS Challenge.** The repository's docs are CC BY 4.0 and its code MIT. Its noise is "Audioset… License: CC BY 4.0", "Freesound… Only files with CC0 licenses were selected" and "Demand… License: CC BY-SA 3.0". Its RIRs are "OpenSLR26 and OpenSLR28… License: Apache 2.0". "The datasets are provided under the original terms that Microsoft received such datasets." The `noise_fullband` set is 58 GB. — [DNS-Challenge README and LICENSE](https://github.com/microsoft/DNS-Challenge) [V]
- **ESC-50.** "The dataset as a whole is available under the terms of the Creative Commons Attribution-NonCommercial license (by-nc/3.0)". "The ESC-10 subset is licensed as a Creative Commons Attribution 3.0 Unported", with per-clip Freesound attributions in LICENSE. — [ESC-50 LICENSE](https://github.com/karolpiczak/ESC-50/blob/master/LICENSE) [V]
- **MUSAN.**
  - OpenSLR lists MUSAN (SLR17) as CC BY 4.0. It is about 109 h of music, speech and noise, 16 kHz WAV, 11 GB.
  - The paper says all content is US Public Domain or Creative Commons, with a LICENSE file in each subdirectory, and that it includes only content that allows commercial use.
  
  — [OpenSLR 17](https://openslr.org/17/); [arXiv 1510.08484](https://arxiv.org/pdf/1510.08484) [S]
- **FSD50K.** Clips are individually licensed (CC0, CC-BY, CC-BY-NC, CC Sampling+) with per-clip metadata, and the annotations are CC BY 4.0. One summary reports that ~84.7% of clips are CC0 or CC-BY. — [Zenodo 4060432](https://zenodo.org/records/4060432); [arXiv 2010.00475](https://arxiv.org/abs/2010.00475) [S]
- **WHAM!** No licence statement was retrieved. — [S: none found]

### Inferences
- [I] **Why noise licences matter less here than speech-data licences.** Augmented copies are never redistributed: "augmented copies of a clip… are not part of the dataset". What still matters is commercial-use compatibility, because the trained model goes to Sarvam. So drop NC sources (ESC-50 full, FSD50K NC clips, WHAM! unless verified CC BY), and keep attribution files for CC BY ones.
- [I] **Babble.** Build babble by summing 3–8 *Part A* real utterances from other speakers. That keeps all speech inside Part A. Do not use MUSAN speech or Common Voice for this.
- [I] **Noise harvested from Part A.** Run Silero VAD over the IndicVoices source files, keep non-speech stretches of at least 0.5 s with energy above the digital floor, and pool them. They are real Indian ambient noise: fans, traffic, TV. This is defensible under the "only Part A" rule, and it is free.

### Gaps
- WHAM! licence not found.
- The OpenSLR 26/28 licence was seen only via the DNS README, not the OpenSLR pages, which were blocked.
- No PS clarification exists on whether external noise or RIRs count as "training data".

## Q5. torchaudio 2.11 load, torchcodec and Docker (critique point 4)

### Takeaway
Critique point 4 is **correct but already in the report** (§B, and §E-SP: "`torchaudio.load` now runs on TorchCodec"), and it is **low-impact for the submitted server**:
- The server receives raw `pcm_s16le` frames over WebSocket and needs no file decoding: `np.frombuffer(buf, '<i2').astype(np.float32)/32768`.
- For offline data tools, use `soundfile` 0.14.0, whose wheels bundle libsndfile (training knowledge; not inspected), or install torchcodec plus system FFmpeg in a separate data image.

### Cited Findings
- **torchaudio 2.11 `load` requires TorchCodec but does not declare it.** `torchaudio.load` docstring: "As of TorchAudio 2.9, this function relies on TorchCodec's decoding capabilities under the hood… the parameters `normalize`, `buffer_size`, and `backend` are ignored." `load_with_torchcodec` raises ImportError "TorchCodec is required…" if torchcodec is missing. The torchaudio 2.11.0 wheel METADATA has **no Requires-Dist lines**, so torchcodec is not pulled in automatically. — torchaudio-2.11.0-cp312 wheel, `__init__.py`, `_torchcodec.py`, METADATA [V]
- **torchcodec links system FFmpeg; only the FFmpeg ops are optional.** torchcodec 0.17.0 (latest on PyPI) ships `libtorchcodec_core{4..9}.so`, and `libtorchcodec_core7.so` is dynamically linked against `libavcodec.so.61`, `libavformat.so.61`, `libavutil.so.59`, `libswresample.so.5` and others. At load it tries FFmpeg majors "9, 8, 7, 6, 5, 4" in turn. "FFmpeg is now an optional dependency, since the image decoders… don't need it", but audio decoding does. The inspected Linux wheel also links CUDA 13 libraries (`libcudart.so.13`, `libnvrtc.so.13`). — torchcodec 0.17.0 wheel, `readelf -d`, `_internally_replaced_utils.py`, `_core/ops.py` [V]
- **soundfile 0.14.0** is the latest on PyPI. — `pip index versions` [L]

### Inferences
- [I] **In `python:3.12-slim`, `torchaudio.load` fails unless both torchcodec and FFmpeg shared libraries are installed.** Install FFmpeg with `apt-get install -y ffmpeg`, which pulls libavcodec and adds image size, or with conda.
- [I] **What the report should say in one line:** "The serving container decodes no files; data tools use soundfile; torchaudio is used only for `functional.*`, `MMS_FA` and `forced_align`, which do not need TorchCodec."
- [I] A CUDA-variant torchcodec wheel paired with a CPU torch will fail to load. Install the variant that matches torch.

### Gaps
- I did not build a slim Docker image to measure the size of the FFmpeg layer.

## Q6. Causal streaming front-end on 80 ms frames, and the corrected augmentation recipe (what the report should add)

### Takeaway
The report's E-SP front-end is sound as a Part E-SP *exercise*. For the *submitted system*, the recommended front-end is minimal:
- decode int16;
- DC block;
- a slow causal gain to the training level;
- optionally an SNR-gated denoiser with OA, enabled only if real-dev KPIs improve;
- VAD used only for side decisions (end-of-speech flush, logging), never to drop input frames.

All robustness should come from training-time augmentation of the source stream, with clean targets.

### Cited Findings
- **Frame sizes and component limits.**
  - An 80 ms frame is 1,920 samples at 24 kHz (3,840 bytes s16).
  - `soxr.ResampleStream(24000, 16000, 1, dtype='float32')` gives variable per-chunk output.
  - Silero VAD 6.2.3 needs exactly 512 samples at 16 kHz or 256 at 8 kHz.
  - pyrnnoise 0.4.5 works on 480-sample frames at 48 kHz.
  - DeepFilterNet3 runs at 48 kHz with 40 ms overall latency.
  
  — [report §E-SP](file:///home/user/interiit_minips/reports/Sarvam%20mini%20PS%20research.md); [partD_partE_tooling.md](file:///home/user/interiit_minips/research_notes/Sarvam%20mini%20PS%20research/partD_partE_tooling.md) [V there]. Current versions confirmed: silero-vad 6.2.3, soxr 1.1.0, pyrnnoise 0.4.5, deepfilternet 0.5.6 — `pip index versions` [L]
- **Augmentation must stay at training time.** "Augmentation applied during training adds no hours." — [PS](file:///home/user/interiit_minips/Sarvam.pdf) [V]
- **phonesim models the send side.** It sets the "send-side level + ambient noise" *before* the encoder, as a handset does. — [phonesim README](https://github.com/deepmark/phonesim) [V]

### Inferences
- [I] **Recommended online augmentation (source stream only; targets stay clean Bulbul).** Each training example takes a seeded random draw; log the seed for reproducibility. Steps 1–5 run in this order: RIR → noise → level → channel → final gain.
  0. **Clean** pass-through with p = 0.35.
  1. **Room, p = 0.25.** Real RIRs (OpenSLR 26/28, Apache 2.0) or pyroomacoustics with RT60 ~ U(0.2, 0.8) s. Shift by the direct-path delay so word times stay valid.
  2. **Additive noise, p = 0.45.**
     - SNR ~ U(0, 25) dB, weighted towards 5–20 dB.
     - Noise pool: Part A non-speech harvest, Part A babble (3–8 talkers), self-recorded noise, coloured noise. Use MUSAN noise/music or DNS-CC0 only if the organisers allow it.
     - Use non-stationary noise (TV, traffic, babble) in at least half the noisy draws.
  3. **Level.** Active speech level ~ U(−35, −15) dBFS (P.56-style). Rare clipping with p = 0.02.
  4. **Telephone channel, p = 0.3** of all examples:
     - **Narrowband, 60% of telephone draws:** resample to 8 kHz with a random kernel (soxr HQ/VHQ, ffmpeg swr, scipy polyphase). Apply a 2nd-order high-pass at ~U(250, 320) Hz for PSTN or ~100 Hz for mobile. Codec drawn from G.711 μ-law/A-law, G.726-32, GSM-FR (libgsm), AMR-NB 4.75–12.2 kbps (AMR-enabled ffmpeg) or Opus NB 6–12 kbps. Burst erasures 0–3% of 20 ms frames, with codec PLC. Upsample to 24 kHz with a random kernel.
     - **Wideband, 40% of telephone draws:** resample to 16 kHz. High-pass at 50–80 Hz. Codec drawn from G.722, AMR-WB 6.6–23.85 kbps or Opus WB 12–24 kbps. Erasures 0–5% for Opus. Upsample to 24 kHz.
     - Either way, simplest is `phonesim` profiles `pstn_narrowband`, `pstn_g726`, `voip_to_cellular_narrowband` (bitrate ∈ {4.75k…12.2k}), `voip_to_cellular_wideband`, `voip_g722_wideband` and `voip_opus_wideband`.
  5. **Bandwidth-only, p = 0.3** of the non-telephone *synthetic* draws: low-pass at a random 4–8 kHz and resample through 16 kHz, so synthetic sources match the ≤ 8 kHz bandwidth of the real buckets (Q1 shortcut).
  6. **Speed or tempo:** only at data-build time with re-timing, never online (Q3).
- [I] **Inference front-end (per 80 ms frame).**
  - int16 → float32.
  - 1st-order DC-blocking high-pass at ~20–40 Hz.
  - Causal AGC towards the training median level, with max +20 dB gain, attack ~50 ms, release ~1 s and a limiter.
  - [optional] Denoise only if the running SNR estimate is below ~10–15 dB, with OA weight w tuned on dev; this costs 10–40 ms of latency, so the gain must outweigh it.
  - Feed the model unchanged otherwise. No resampling, since the model takes 24 kHz.
  - Run Silero VAD on a 16 kHz soxr-resampled copy, for flush and end-of-speech heuristics only.
- [I] **Validation the report should add.**
  - On a real, speaker-disjoint Hindi dev set, include genuine phone-call recordings: self-recorded calls made over a real Indian mobile network and saved at the receiver are the most faithful proxy.
  - Compare four inference variants (raw / denoise / denoise+OA / gated) on ASR-BLEU, entity and intent proxies, and on UTMOS or DNSMOS of the *output* English, as a stand-in for the audio-quality KPI.
  - Also report the training ablation with and without telephone augmentation, scored on the telephone dev slice.

### Gaps
- There is no public measurement of Mimi or Hibiki-class S2ST robustness to G.711 or AMR input.
- The best augmentation probabilities above are engineering defaults, not tuned or published values. Tune them on the dev set.
- The Sarvam base checkpoint's input codec and training augmentation are unknown until release; if it was trained with its own channel augmentation, our augmentation should match it.

### Verdict summary (for the report writer)
| Critique point | Verdict | What the report should change |
|---|---|---|
| 1. No handling of 8 kHz G.711 / 16 kHz AMR-WB vs 24 kHz; add 300–3400 Hz band-pass and upsample | **Incorrect as stated**: §C4 and §E-SP already specify 24→8 kHz, 300–3400 Hz, G.711/GSM/AMR-NB/Opus, loss, →24 kHz | Add the **wideband 16 kHz branch** (G.722 / AMR-WB / Opus-WB). Make the low edge 2nd-order and randomised (~100 Hz mobile, 250–320 Hz PSTN), since the codec does not imply it while the 8 kHz resample already gives the upper edge [L]. Randomise the resampler. **Fix the bandwidth shortcut** between real (16 kHz) and full-band synthetic sources. Say "no resampling at inference". Name the AMR-enabled ffmpeg requirement and that SpeechBrain `CodecAugment` and `torchaudio.io` are broken on 2.11. |
| 2. Add observation adding at 10–20% (Iwamoto 2022) | **Already present** (§E-SP: + 0.1–0.3 × noisy). The fixed ratio is not from the paper [S] | Present OA as convex mixing with w tuned on real dev. Make the default "no denoise at inference, robustness via training augmentation". If a denoiser is used, apply it during training too. Account for its 10–40 ms latency. |
| 3. Flat open Indic TTS prosody degrades the audio-quality KPI | **Wrong mechanism**: the KPI is on Bulbul-voiced English output. Flat input prosody widens the synthetic–real gap (translation, intent, entity, latency), not output quality directly | Re-attribute the claim. Add input-side rate and temperature variation. List the real audio-quality drivers: silence-insertion hard cuts (fades, boundary-only insertion or chunk-wise Bulbul), Bulbul pace and temperature, alignment accuracy, sampling temperature, stream underruns, causal loudness and limiter, no time-stretch. Note that noisy-source/clean-target augmentation helps keep output clean. |
| 4. `torchaudio.load` in 2.11 needs torchcodec + system FFmpeg; breaks slim Docker | **Correct** [V], **already noted** in the report, and **low-impact**: the server decodes raw PCM and needs no file I/O | One line: the serving image avoids torchaudio I/O; data tools use soundfile; torchcodec is not auto-installed by torchaudio 2.11 and needs FFmpeg 4–9 shared libraries. |
