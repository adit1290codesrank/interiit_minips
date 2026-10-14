# Inter IIT Tech Meet 15.0 — Sarvam.ai Mini PS

Working repository for the Sarvam.ai mini PS. The task: a system that listens to live Hindi speech, freely mixed with English, and speaks the English translation while the speaker is still talking. The mini PS is the individual selection task: a report plus a small pipeline, defended in an interview where you run your code on a new clip.

## Folder guide

```text
interiit_minips/
├── README.md                                   ← you are here
├── Sarvam.pdf                                  ← main problem statement (the full competition task)
├── Inter IIT Tech Meet 15.0_ Sarvam.ai Mini PS.pdf  ← mini PS (selection task, Parts A–E)
├── reports/
│   └── Sarvam mini PS research.md              ← START HERE: the master report
├── research_notes/
│   └── Sarvam mini PS research/                ← the seven research notes behind the report
│       ├── partA_simultaneous_fundamentals.md
│       ├── data_sources_licences.md
│       ├── targets_and_timing.md
│       ├── hard_content_validation.md
│       ├── transcription_alignment_realgap.md
│       ├── finetune_and_latency.md
│       └── partD_partE_tooling.md
└── prototypes/
    └── hindi_number_parser_prototype.py        ← draft Hindi number-word parser (basis for the Part D checker)
```

Folders still to be created, as laid out in report §D0: `clips/` (your recordings), plus the code and results folders for Parts D and E.

### The two PDFs

| File | What it is |
|---|---|
| `Sarvam.pdf` | The main problem statement. It defines the data to build (Part A dataset, ≤200 h), the fine-tuning phase, the WebSocket streaming protocol (§5.1), the dataset format (§5.2), KPIs, gates and rules. Read it once end to end; most mini PS answers are in it. |
| `Inter IIT Tech Meet 15.0_ Sarvam.ai Mini PS.pdf` | The selection task, with the points for each part: A Understanding (10), B Research (40), C Dataset design (25), D Build and measure (10), E Track task (15). |

### The master report: `reports/Sarvam mini PS research.md`

About 16,000 words, structured to mirror the mini PS:

| Section | Covers |
|---|---|
| Opening summary + **Status and caveats** box | The main argument in one paragraph, the evidence tags, and what is untested |
| **Part A** (A1–A5) | What the system does and where points are won and lost; one session frame by frame; the 10 most misread rules; the "पाँच लाख… नहीं नहीं, पाँच करोड़" answer; the handwritten wait-3 / Average Lagging appendix (AL = 2.625) |
| **Part B** (B1–B7) | Proposals: data sources, training targets, timing the English, hard content (full); fine-tuning blind and decode-time latency (compact); the original idea |
| **Part C** (C1–C7) | Dataset design: sources and the 200 h split, tools, validation thresholds, real-world conditions, failure modes, **the pipeline diagram (C6)**, three open questions |
| **Part D** (D0–D7) | Build plan: repo layout, recording six clips, Level-1 transcription, alignment, the 30-boundary hand check, translation, the entity checker, `RESULTS.md` template, interview runbook |
| **Part E** | ML wait-k track (recommended, reuses Part D), plus compact Dev (WebSocket echo server) and Signal Processing (causal front-end) plans |
| Compressing into 4 pages | What to keep from each part for the submission |
| Appendix | Source table: 36 sources, one line each on what it told us |

**Evidence tags used throughout the report:**
- **[V]** verified against code, a README, a downloaded package or the PS PDFs.
- **[L]** tested locally.
- **[S: re-check]** from a search extract; re-check it against the original paper before quoting.

### The research notes: `research_notes/Sarvam mini PS research/`

Detailed notes, each with a takeaway, cited findings, inferences and gaps. Go to these when the report's summary isn't enough or an interviewer digs into a point.

| Note | Topic | Feeds |
|---|---|---|
| `partA_simultaneous_fundamentals.md` | wait-k, Average Lagging / LAAL / DAL definitions from SimulEval code, the worked AL example, read/write policies, Hibiki/Moshi architecture, self-correction structure | Part A, B6 |
| `data_sources_licences.md` | Hindi and code-mixed speech datasets with licences and hours; TTS and text sources; the 200 h split | B1, C1 |
| `targets_and_timing.md` | How literal the English should be, sentence mapping, Hibiki's contextual alignment, English aligners, the timing ("safe time") recipe | B2, B3 |
| `hard_content_validation.md` | Generating entity-heavy and disfluent data, Hindi number tools, TTS/translation validation thresholds, failure modes | B4, C3, C5 |
| `transcription_alignment_realgap.md` | IndicVoices Level 1, Hindi ASR options, Hindi forced aligners, translation tools, the synthetic-vs-real gap and augmentation | C2, C4, D |
| `finetune_and_latency.md` | Preparing to fine-tune an unknown checkpoint (moshi-finetune recipe, compute budget), decode-time latency choices, likely latency metrics | B5, B6 |
| `partD_partE_tooling.md` | Install commands, APIs, version pins and pitfalls for Part D and all three Part E tracks | D, E |

### The prototype: `prototypes/hindi_number_parser_prototype.py`

A ~50-line draft that converts Hindi number words (including lakh/crore, डेढ़/ढाई/साढ़े/सवा/पौने, and digit-by-digit phone numbers) to values. It is built on NeMo text-processing's Hindi tables. Run it with the path to NeMo's Hindi `data` folder as the first argument. See the caveats below before using it.

## Suggested reading order

1. Skim `Sarvam.pdf` and the mini PS PDF.
2. Read the report's opening summary and **Status and caveats** box.
3. Read **Part A** of the report closely; you must be able to explain it without notes.
4. Read Parts B and C, opening the matching research note for any point you will put in your submission.
5. Use Parts D and E as the build checklist.

## Status and caveats

- **Unchecked paper numbers.** The network proxy blocked arXiv, Hugging Face and Sarvam's docs during research. Most paper figures came from search results and are tagged **[S: re-check]** in the report. Re-check any you put in the submission, since you'll be cross-questioned.
- **Number parser.** `prototypes/hindi_number_parser_prototype.py` is a rough draft with four known bugs:
  - "दो तीन लाख" (two to three lakh) returns 500,000.
  - "साढ़े चार बजे" (4:30) returns 4.
  - "एक एक" becomes the digit string 11, though it can mean "one each".
  - Amounts come back as floats.

  The Part D plan (report §D6) lists the fixes. NeMo's Hindi tables weren't installed in the research environment, so it was tested on stand-in tables.
- **Untested code.** The alignment and other Part D/E scripts in the report are sketches. Torch wasn't installed in the research environment, so they haven't been run. The echo-server sketch (E-Dev) was run against a test client and behaved correctly. The wait-k numbers and simulator were checked by running them.

## Next step

**Part D:** record your 6 clips (20–40 s each, natural code-mixed Hindi with lakh/crore amounts, a phone number, names, a negation and self-corrections; see report §D1) and add them under `clips/raw/`. Once they're in the repo, the alignment, the entity checker and the wait-k simulator can be built on them.
