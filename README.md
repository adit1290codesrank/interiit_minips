# Inter IIT Tech Meet 15.0 — Sarvam.ai Mini PS

Working repository for the Sarvam.ai mini PS: a real-time Hindi (code-mixed) → English speech-to-speech translation problem.

## Contents

| Path | What it is |
|---|---|
| `Sarvam.pdf` | Main problem statement |
| `Inter IIT Tech Meet 15.0_ Sarvam.ai Mini PS.pdf` | Mini PS (selection task) |
| `reports/Sarvam mini PS research.md` | Master report: Part A explanation, Part B proposals, Part C dataset design, build plans for Parts D and E |
| `research_notes/Sarvam mini PS research/` | The seven research notes behind the report |
| `prototypes/hindi_number_parser_prototype.py` | Draft Hindi number-word parser (basis for the Part D entity checker) |

## Status and caveats

- **Unchecked paper numbers.** The network proxy blocked arXiv, Hugging Face and Sarvam's docs during research. Most paper figures came from search results and are tagged **[S: re-check]** in the report. Re-check any you put in the submission, since you'll be cross-questioned.
- **Number parser.** `prototypes/hindi_number_parser_prototype.py` is a rough draft with four known bugs:
  - "दो तीन लाख" (two to three lakh) returns 500,000.
  - "साढ़े चार बजे" (4:30) returns 4.
  - "एक एक" becomes the digit string 11, though it can mean "one each".
  - Amounts come back as floats.

  The Part D plan (report §D6) lists the fixes. It also needs NeMo's Hindi number tables to run. Pass the path to NeMo text-processing's Hindi `data` folder as the first argument. Those tables weren't installed in the research environment, so it was tested on stand-in tables.
- **Untested code.** The alignment and other Part D/E scripts in the report are sketches. Torch wasn't installed in the research environment, so they haven't been run. The echo-server sketch (E-Dev) was run against a test client and behaved correctly. The wait-k numbers and simulator were checked by running them.

## Next step

**Part D:** record your 6 clips (20–40 s each, natural code-mixed Hindi with lakh/crore amounts, a phone number, names, a negation and self-corrections; see report §D1) and add them under `clips/raw/`. Once they're in the repo, the alignment, the entity checker and the wait-k simulator can be built on them.
