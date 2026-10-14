# Verb-final Hindi→English: is "Hold the verb, speak everything else" valid, and how will the output be scored?

Scope: this note checks the core strategy of `reports/Sarvam mini PS research.md` (title; §A4; §B2; §B3) against interpreting-studies evidence, English and Hindi grammar, and how ST evaluation usually works. It then rules on an external critique with two claims: (1) the thesis is a "syntactic fallacy" and should be reframed around passives, topicalisation and fronted adverbials; (2) converting crore→million "violates dataset requirements and risks entity point deduction".

**Evidence tags.** **[V]** means checked against the PS PDF or the report text in this repo. **[S]** means a search-engine extract of a paper or page; the proxy blocked arXiv, the ACL Anthology, ar5iv, cs.umd.edu and hhexiy.ai, so no full paper text was read. Numbers tagged [S] should be re-checked before anyone quotes them. **[K]** means standard grammar or textbook knowledge that this session did not source. These items appear only under Inferences or Gaps and must be checked against Kachru (2006) or Koul (2008) before they are quoted.

**Verdict in one paragraph.** The critique is **half right on claim 1 and mostly wrong on claim 2**.

- *Claim 1, what it gets right.* Read literally, the slogan "speak everything else" is wrong. A canonical English clause cannot voice the object before the verb. The report's own abstract example, "speak 'Yesterday I… my old bank account' immediately", is a broken fragment unless the English is then restructured, for example with left dislocation ("…, I got it closed").
- *Claim 1, what it gets wrong.* The report's body (§B2) already prescribes the critique's toolkit: fronted adverbials, causatives, and topicalisation with resumptive pronouns. The report is also right, and the critique wrong, to reject the passive as the default device. In this task the passive deletes the agent, and the intent KPI explicitly scores "who does what to whom". Human interpreters do use passives more than translators (He et al. 2016), but they are scored on a different objective.
- *Claim 2.* The report never recommends converting crore→million. It keeps lakh/crore in the dataset (PS §3) and in speech. Its checker only *accepts* "50 million" as value-equivalent while flagging the unit change. The critique is right only that a team which *did* convert would break the Part A dataset rule. Whether the final scorer accepts "fifty million" is unknown, so keeping "crore" is the dominant strategy.
- *Fixes needed.* Re-word the thesis around *commitments*, not *words*. Add two blind spots to the polarity rule: clause-final question/tag signals (intonation, "ना", "क्या" placement) and the tense/aspect of "कल".

---

## 1. How do professional simultaneous interpreters handle verb-final source languages into English?

### Takeaway
Interpreters facing SOV→SVO use a small, well-attested toolkit: waiting (a longer lag), segmentation (also called chunking or "salami"), anticipation of the final verb, neutral stalling material, restructuring, passivisation, and omission or generalisation. Verb-final sources measurably lengthen the lag. Professionals typically trail by about 2–4 s, with wide swings. Passivisation is a documented human strategy, but it serves an objective (fluency at low lag) that differs from Sarvam's, which weights intent and entities heavily.

### Cited Findings
- He, Boyd-Graber and Daumé III, NAACL 2016, "Interpretese vs. Translationese: The Uniqueness of Human Strategies in Simultaneous Interpretation", pp. 971–976. They built a parallel corpus of translated and simultaneously interpreted Japanese→English text and compared the two using features tied to the strategies of **segmentation, passivisation, generalisation and summarisation** [S] — [ACL N16-1111](https://aclanthology.org/N16-1111/); later summary in [MoTra 2021](https://preview.aclanthology.org/fix_video/2021.motra-1.9.pdf).
- He et al. note that information the English sentence needs (the verb, the head noun being modified) often arrives late in the Japanese sentence. A secondary summary says interpreters "often use the passive voice" to avoid waiting for the Japanese final verb. It also says passive use and sentence segmentation were "significantly higher" in interpretation than in translation, and that a classifier separated the two with up to 94% accuracy [S, secondary summary; the exact passive-rate numbers were not retrieved] — [ACL N16-1111 PDF](https://aclanthology.org/N16-1111.pdf). The paper's code has a dedicated `--passive` comparison mode (`src/count_statistics.py --passive`) [V, README fetched] — [hhexiy/interpretese](https://github.com/hhexiy/interpretese).
- Grissom II, He, Boyd-Graber, Morgan and Daumé III, EMNLP 2014, "Don't Until the Final Verb Wait". This work applies reinforcement learning to verb-final German→English. The system **predicts the unseen final verb** and learns when a prediction is reliable enough to commit; it reportedly beats both batch and monotone strategies [S] — [EMNLP 2014 PDF](https://mirror.aclweb.org/emnlp2014/papers/pdf/EMNLP2014140.pdf). Follow-ups compare human and machine incremental verb prediction in Japanese and German ([CoNLL 2016, K16-1010](https://preview.aclanthology.org/fix_video/K16-1010.pdf)) [S].
- Lag evidence:
  - Oléron and Nanpon found shorter ear-voice spans for English→French (SVO→SVO) than for German→French (SOV→SVO), and Goldman-Eisler (1972) attributed the difference to German verb position [S] — [Érudit review](https://id.erudit.org/iderudit/1060169ar).
  - Bevilacqua (2009) argues that German's rigid verb-final structure "appears to require more extensive anticipation efforts as well as a longer ear-voice span" than Dutch [S] — same source.
  - Viviani (2021, Geneva) found a longer EVS from German than from French into Italian [S] — [UNIGE](https://archive-ouverte.unige.ch/unige:155800).
- Typical lag magnitudes:
  - Lee (2002) analysed about 800 sentences of English→Korean (SVO→SOV) TV interpreting and found an **average EVS of 3 s**, driven by sentence length, speech rate and pauses [S] — [Meta 47(4) 596–606](https://www.erudit.org/en/revue/meta/2002/v47/n4/008039ar.pdf).
  - A secondary review cites Ono et al. (2008) at **2–4 s** for Japanese interpreters, and a German→French study at **about 3 s** with swings from **−5 to +11 s** [S] — [VNU review](https://jfs.ulis.vnu.edu.vn/index.php/fs/article/download/5233/4523).
- Hodzik and Williams (2017, *Interpreting* 19(1):1–20) measured target-verb latency in German→English simultaneous interpreting (22 advanced students plus 11 trainee or practising interpreters, with a shadowing control) and tested contextual constraint against transitional probability as drivers of prediction [S] — [Benjamins](https://www.benjamins.com/catalog/intp.19.1.01hod). A related Turkish→English (SOV→SVO) study found that **contextual constraint, not transitional probability**, cut latency on the sentence-final verb and produced **more syntactic restructuring** [S] — [UMONS manuscript](https://orbi.umons.ac.be/bitstream/20.500.12907/53841/1/manuscript_Hodzik%2c%20Ozdemir%2c%20Conker%20and%20Bilgin.pdf).
- Practitioner taxonomies for SOV→SVO list waiting, stalling, anticipation and chunking. Stalling is described as using "a neutral word or expression to fill in until they know for sure" [S, practitioner podcast] — [The Complete Interpreter](https://thecompleteinterpreter.buzzsprout.com/2125089/episodes/15556373-approaches-to-interpreting-from-sov-svo-e-g-japanese-english). A strategy-teaching review lists "chunking/segmentation/salami, restructuring/changing order, transcoding, reproduction, linguistic anticipation" [S] — [ResearchGate](https://www.researchgate.net/publication/282102085_Putting_interpreting_strategies_in_their_place_Justifications_for_teaching_strategies_in_interpreter_training).
- On Japanese, interpreters often **keep source order** where the translation could vary, conditioned on factors such as the length of post-modifiers [S] — [J-STAGE ITS vol. 20](https://www.jstage.jst.go.jp/browse/its/20/0/_contents/-char/ja). A Mandarin→English study ties word-order mismatch to **omissions** and reports reorganisation and segmentation as coping strategies [S] — [PolyU](https://research.polyu.edu.hk/en/publications/omission-in-simultaneous-interpreting-word-order-differences/).
- Hindi–English: no empirical study of Hindi/Urdu→English simultaneous interpreting strategies was found. The only first-hand evidence is journalistic. A Lok Sabha interpreter names syntax as the core difficulty: interpreters often have to wait for the end of the sentence, which forces them to anticipate. The service began with English–Hindi simultaneous interpretation in 1964 and aims to convey "at least 90 percent" of exchanges [S, magazine] — [Caravan](https://caravanmagazine.in/lede/parliamentarily-speaking).

### Inferences
- **Two human strategies are off-limits or risky under Sarvam's rules.**
  - *Anticipation* (saying the verb before it is heard) is a premature commitment that cannot be retracted ("No take-backs", PS §5.1 [V]).
  - *Stalling filler* risks the scorer-integrity clause ("filler speech, padding", PS §10 [V]).
  - *Omission and generalisation* trade directly against the entity and intent KPIs.
  - What remains usable is **segmentation, waiting, restructuring (fronting, dislocation) and causative or light-verb frames**. That matches the report's §B2 list, not the critique's passive-first list.
- **Why passives work for human interpreters and not here.** Human Japanese→English interpreters use passives because their implicit objective is fluency at low lag, and dropping an agent is cheap when the agent is recoverable from context. Sarvam's intent KPI scores "who does what to whom" explicitly [V], so the trade-off flips. The passive is acceptable only when the Hindi is itself passive or agentless (for example "खाता बंद कर दिया गया"), or when the agent is re-inserted.
- **Lag budget.** A professional EVS of about 3 s, with wide variance and longer for SOV sources, is a fair human benchmark for the latency KPI. A system that holds only verb groups and entity spans should sit in the same 2–4 s band. One holding whole sentences will exceed it on long sentences.

### Gaps
- Exact passive-voice rates, segmentation rates and examples from He et al. 2016 could not be retrieved (PDF hosts blocked). Re-check before quoting "significantly higher" or 94%.
- No Hindi/Urdu→English interpreting corpus or strategy study was found. Camayd-Freixas, and Gile's Effort Model as applied to SOV sources, were not retrieved. Any statement about Gile's model should be sourced separately.
- No quantitative EVS figure was retrieved for German→English, or for any Hindi pair.

---

## 2. Is "English can't say the object before the verb" accurate? What constructions allow early mention, at what cost? A worked timeline for the PS sentence

### Takeaway
In canonical SVO English the object cannot precede the verb. But spoken English has several grammatical ways to voice the object or topic early: left dislocation with a resumptive pronoun, "As for X / About X", the causative ("have/get X Ved"), nominalisation, and the passive. Each costs something: markedness, extra words, or a lost agent. In practice "hold the verb" means **speak the subject and the fronted adverbials early, then speak the verb and the object as one chunk** after the Hindi verb group. Left dislocation is a targeted extra for long objects. For the PS sentence, the realistic gain from early object mention is under about 1 s of end offset.

### Cited Findings
- **Left dislocation (LD) is real but marked in English, and it is more common in Indian English.**
  - Corpus work reports about **1.42 LD per 1,000 words in Indian English, against 0.16 in British English** (citing Lange 2012), and 5.19 in spoken Ugandan English, attributed to substrate influence [S, conference paper quoting Lange] — via [Goylo 2021 / BSU](https://journals.bsu.by/index.php/philology/en/article/view/3574).
  - Goylo's corpus study of Jamaican, Indian and Canadian English finds LD "more common in informal settings", with frequency shaped by speaker style [S] — [BSU](https://elib.bsu.by/bitstream/123456789/269965/1/39-51.pdf).
  - A 2025 *Linguistics* study measures LD frequency across four registers (informal/formal × spoken/written) for 287 participants [S] — [De Gruyter](https://www.degruyterbrill.com/document/doi/10.1515/ling-2024-0017/html).
  - Switchboard telephone-conversation work argues that LD has a general topic-promoting function, with use conditions about as restrictive as those for topicalisation [S] — [Elsevier/J. Pragmatics](https://researchcollaborations.elsevier.com/en/publications/topicalization-and-left-dislocation-a-functional-opposition-revis/).
- **The Hindi causative maps naturally onto English "have/get X done".**
  - The *-vaana* causatives (करवाना) "denote an action which is not directly performed by the subject but indirectly through some agent" [S, learner blog] — [Transparent Language](https://blogs.transparent.com/hindi/?p=4202).
  - The intermediary is marked with से, as in "माता जी नौकरानी से चाय बनवा रही हैं" = "Mother is having the maid make tea" [S, university grammar notes] — [UMN Hindi-Urdu grammar 7.3](https://open.lib.umn.edu/hindiurdu/chapter/grammar-notes-7-3-the-hindi-causative-construction/).
  - Only the indirect causative licenses the से-marked intermediate agent [S] — [Ramchand, causal chains](https://gillianramchand.blog/wp-content/uploads/2017/08/ramchandcausalchains.pdf).
- The PS's own gloss is "Yesterday I got my old bank account closed" for "मैंने कल अपना पुराना बैंक खाता बंद करवा दिया", with the remark that until "बंद करवा दिया" is heard "the system cannot tell whether the account was opened, closed or only checked" [V] — [PS §2](file:///home/user/interiit_minips/Sarvam.pdf).
- The report's abstract says to "speak 'Yesterday I… my old bank account' immediately, but hold the English verb until 'बंद करवा दिया'" [V], line 3 of the report. Its §B2 prescribes "causative 'got X closed', fronted adverbials, a topicalised object with a resumptive pronoun only when the object is long" and warns that resumptive topicalisation "reads as unnatural English" [V], line 151.
- He et al. 2016 document **passivisation** as an interpreter strategy for the late Japanese verb [S] — [ACL N16-1111](https://aclanthology.org/N16-1111/).

### Inferences
- **On the critique's grammar claim.**
  - It is accurate for canonical clauses: there is no grammatical "I my old bank account closed".
  - It is inaccurate as a general claim, since LD, "as for", the causative, nominalisation and the passive all put the object's words before the main lexical verb.
  - The critique's own remedy (passive, topicalisation with resumptive, fronted adverbials) is the report's §B2 list with one difference: the report rejects the passive for an intent-specific reason. That reason is correct for this KPI mix.
- **What the critique did catch.**
  - The *slogan* "speak everything else" over-promises.
  - The abstract's example "Yesterday I… my old bank account" is a stranded object fragment, which a listener hears as disfluent. It only works if followed by a resumptive ("…, I got it closed") or if "my old bank account" is not voiced until the verb arrives.
  - The thesis and abstract should be re-worded; §B2 needs no structural change.
- **The causative is *not* a pure verb-deferral device.** [K for the Hindi grammar]
  - "Got" commits to *causation*, and that information arrives only at करवा (word 8 of 9). If the speaker had said "बंद कर दिया" (direct: "I closed"), an early "got" would be a premature, if minor, who-did-what error.
  - So "Yesterday I got…" before word 8 is itself a small anticipation.
  - The safe use of the PS gloss is "Yesterday I…" early, then "got my old bank account closed" after the verb group.
- **Ergative case gives an early tense cue.** [K]
  - The ergative मैंने (-ने) marks a perfective transitive clause, so by word 1 the clause is almost certainly past perfective. This makes "Yesterday" (rather than "Tomorrow", since कल means both) safe at word 2.
  - Without ergative marking (for example "मैं कल … करवा दूँगा", "Tomorrow I will get…"), कल stays ambiguous until the final auxiliary.
  - This is a strong argument that the placement rule should key on *the Hindi cue that resolves each English commitment*, not on part of speech. The report's §B3 MT-likelihood alignment partly captures this.
- **Worked timeline for the PS sentence.** The numbers are illustrative only, not measured.
  - Assumptions: Hindi at about 0.35 s per word (word *i* ends at 0.35·*i* s, so "दिया" ends at 3.15 s); English TTS at about 0.4 s per word; safety margin δ = 0.3 s after the cue word; latency measured as the English end time minus the Hindi end time (3.15 s).
  - Hindi words and end times: मैंने 0.35 · कल 0.70 · अपना 1.05 · पुराना 1.40 · बैंक 1.75 · खाता 2.10 · बंद 2.45 · करवा 2.80 · दिया 3.15.

| Option | English as spoken (start time) | Premature commitments | End time | End offset | Notes |
|---|---|---|---|---|---|
| A. Sentence-level wait | "Yesterday I got my old bank account closed" @3.45 (8 words) | none | 6.65 s | 3.5 s | Fluent; worst lag |
| B. Report default: subject + adverbial early, hold verb + object | "Yesterday I…" @1.00–1.80; "…got my old bank account closed" @3.45 (6 words) | none | 5.85 s | 2.7 s | Grammatical; one audible mid-clause pause (~1.6 s) |
| C. Left dislocation | "Yesterday, my old bank account…" @1.00–3.00 (each word no earlier than its Hindi source + δ; "account" 2.60–3.00); "…I got it closed" @3.45 (4 words) | none | 5.05 s | 1.9 s | Saves ~0.8 s over B; 1 extra word ("it"); marked but attested in Indian English |
| D. Agentless passive | "Yesterday my old bank account…" @1.00–3.00; "…was closed" @3.45 | none on polarity, but **drops the agent and the causation** | 4.25 s | 1.1 s | Fastest; intent loss ("who did it") |
| E. PS gloss spoken continuously from 1.00 s | "Yesterday I got my old bank account closed" @1.00–4.20 ("got" at 1.80–2.20) | "got" asserts a positive, causative event at 1.8 s, before बंद (2.45), any नहीं, and करवा (2.80) | 4.20 s | ≈1.05 s | Ties D on lag with no extra wait, but unsafe: for "बंद नहीं करवाया" or "बंद कर दिया" the "got" is already wrong |

  - Reading of the table: under these assumptions, early object mention (C) is worth about 0.8 s of end offset over the report's default (B), at the cost of one marked construction. The passive (D) gains a further 0.8 s by giving up the agent; speaking the PS gloss continuously (E) is no faster than D and commits causation and polarity before the cues arrive. For a 9-word sentence with a 4-word object, the debate is about **sub-second** gains. It matters more for long objects (relative clauses, multi-word entities), which is exactly where the report's rule of "resumptive only when the object is long" applies.
- **Cost model for the constructions.**
  - LD or "as for": +1–2 words, marked, risk of a lower reference match and naturalness rating.
  - Causative: free when the Hindi is causative, wrong when it is not.
  - Nominalisation ("the closure of my old bank account was done…"): heavy and unnatural; avoid.
  - Passive: loses the agent; use only when the source is passive or agentless.

### Gaps
- No acceptability or MOS study of left-dislocated or "as for" English *speech* was found, so the naturalness cost of option C is unquantified.
- Hindi conversational speech rate (words/s) was not sourced. The 0.35 s/word figure is an assumption for illustration and should be replaced by measured Part D clip timings.
- The ergative→perfective cue and the गा/गी future marker are [K]. Verify them in Kachru (2006) or Koul (2008).

---

## 3. Hindi negation, conditionals, questions and causatives: does "hold verb + negator until the Hindi verb group" cover intent?

### Takeaway
The rule covers the main polarity case: preverbal नहीं, न or मत sits inside or just before the verb group, so polarity settles there. It misses three intent signals that arrive **at or after** the verb group's end: polar questions marked only by rising intonation or a final tag ("…ना?", "है ना?"), the tense/aspect auxiliary that resolves "कल" (yesterday or tomorrow), and conditional or contrastive markers ("तो", "नहीं तो", "वरना"). The hold point should be "the end of the Hindi clause, including its final tag and intonation", not "the Hindi verb group".

### Cited Findings
- **Three negators with distinct uses:** मत (non-honorific imperative); न (subjunctive, honorific imperative, conditional, participial and gerundive phrases); नहीं (elsewhere), per Bhatia (1995) [S] — [LINGUIST List 16.1627](https://linguistlist.org/issues/16/1627/). Nahi expresses mainly negative epistemic necessity and mat exclusively negative deontic necessity [S] — [HAL-INALCO](https://hal-inalco.archives-ouvertes.fr/hal-01389401).
- **Position:** नहीं comes before the main verb ("Ram nahi gaya", "Ram did not go") [S] — [arXiv 2607.16765](https://arxiv.org/pdf/2607.16765). It usually sits right before the main verb, or before the auxiliary when there is no main verb [S, learner resource] — [HindiPod101](https://www.hindipod101.com/blog/2021/08/10/hindi-negation/). In formal syntax the sentential negator heads its own NegP immediately below TP [S] — [Kumar monograph, Routledge](https://www.routledge.com/products/9780203959763). Sentential negation also makes some word orders that are fine in affirmatives ungrammatical [S] — [Vasishth 2004](https://www.ling.uni-potsdam.de/~vasishth/pdfs/VasishthRLC04.pdf).
- **Polar questions:** Hindi-Urdu yes/no questions use rising intonation on the verbal complex with no verb inversion, and *optionally* the particle क्या. Its unmarked position is clause-initial; clause-medial क्या signals contrastive focus. One small study found that **only 6 of 50 yes/no questions had the particle, while 47 of 50 had final rising intonation** [S] — [Syed & Dash, GLOW-in-Asia 2017](https://glowlinguistics.org/asia11/wp-content/uploads/sites/3/2017/11/v1-syed-dash.pdf); [CLE Urdu questions](https://www.cle.org.pk/clt16/Presentations/Questions%20in%20Urdu%20Word%20Order%20and%20Intonation.pdf).
- **Causatives and "who does what to whom":** -वाना/-वाया causatives encode indirect action through an intermediary marked by से [S] — [UMN grammar notes 7.3](https://open.lib.umn.edu/hindiurdu/chapter/grammar-notes-7-3-the-hindi-causative-construction/); [Ramchand](https://gillianramchand.blog/wp-content/uploads/2017/08/ramchandcausalchains.pdf).
- **The KPI wording:** intent covers "what is asked, refused, confirmed or corrected, negations, conditions, and who does what to whom" [V] — [PS §7](file:///home/user/interiit_minips/Sarvam.pdf).
- **The report's rule:** "place the English verb, its auxiliary and any 'not' after the Hindi verb group and its negator" (§A4 step 3), and "negators and auxiliaries go to the Hindi verb group" (§B3 step 3) [V].

### Inferences
- **"Asked" is the uncovered intent class.** If 47 of 50 polar questions carry their force mainly in final intonation, the first English word ("Did…" vs "I…") depends on information that arrives *last*. A monotonic declarative start ("Yesterday I got…") would turn a question ("मैंने कल खाता बंद करवा दिया?", "Did I…?" or "Has it been closed?") into a statement. The same applies to confirmation-seeking tags ("…करवा दिया ना?", "I got it closed, right?").
  - The **English rendering of a tag is itself monotonic-friendly**: a declarative plus a tag ("…, right?") lets the system start early and append the question force at the end. A full inverted question ("Did you…") does not.
  - Recommended target convention: render Hindi intonation-only questions as **declarative + "right?" / "is that right?"** where the Hindi has no क्या. Use an inverted English question only when a clause-initial क्या or a wh-word is heard early.
- **Conditionals.** [K] अगर/यदि at the start of the clause is an *early* cue and can be voiced immediately ("If…"). But अगर is often dropped, leaving the condition marked only by the subjunctive or by "तो" at the start of the consequent. "नहीं तो" / "वरना" ("otherwise") reverse polarity across clauses. The placement rule needs a "conditional-or-not" commitment, resolved at the first of अगर, the protasis verb form, or तो.
- **Negation positions the rule must handle.** [K]
  - The standard case is preverbal नहीं between a conjunct noun and the light verb ("बंद नहीं करवाया"). Under negation the vector verb (दिया) usually drops, so "दिया" will not appear.
  - Colloquial post-verbal or emphatic नहीं also occurs ("करवाया ही नहीं", "गया नहीं") and comes *after* the verb.
  - Negation can also scope only over a constituent ("पाँच लाख नहीं, पाँच करोड़"), which the report already treats as repair versus negated predicate (§A4) [V].
  - So "verb group + negator" should be implemented as **"verb group plus one token of look-ahead, or the clause boundary"**.
- **Causatives.** The report correctly says the passive would delete the causer. It should also say that the English must keep the **से-intermediary** where one is present ("I had the bank close it", not "I closed it"), because that is the "whom" in "who does what to whom".

### Gaps
- No Hindi-specific corpus statistic was found for the share of negations that are post-verbal in conversation, or for how often अगर is dropped.
- The 6/50 vs 47/50 figure comes from a small study (likely Urdu or Hindi read speech); it may not hold for telephone speech.
- Kachru (2006) and Koul (2008) were not retrieved. All [K] items need checking against one of them.

---

## 4. How are intent and entity accuracy likely scored? Is "5 crore" → "50 million" safe?

### Takeaway
The PS scores everything "on your system's English speech", almost certainly through an ASR transcript, with entity accuracy as "exactly right" plus a stated equivalence for written number forms. Entity metrics in the literature are often **exact string match**, while LLM judges would likely treat 5 crore = 50 million as equivalent. Because the references "keep the speaker's units" and the equivalence rule is only illustrated with *same-unit* forms (25,000 / 25k), **keeping "crore/lakh" in both the dataset and the output weakly dominates conversion**: it can never score worse, and conversion adds an arithmetic error channel. The critique is right that converting in the *dataset* breaks PS §3. It is wrong that the report proposes conversion.

### Cited Findings
- PS §3 (Part A translations): "Write numbers in words. Digits are also accepted. **Amounts keep the speaker's units (lakh, crore).** A sentence may carry several English translations" [V] — [PS §3](file:///home/user/interiit_minips/Sarvam.pdf).
- PS §11 FAQ: "Equivalent written forms of the same number, such as 25,000 and 25k, both count as correct." Also: "The references encode the corrected meaning. Speaking the abandoned value as fact is wrong" [V] — [PS §11](file:///home/user/interiit_minips/Sarvam.pdf).
- PS §7: "All final KPIs are measured on your system's English speech"; entity accuracy means names, places, organisations, amounts, numbers, dates and products "come out exactly right"; KPIs are "scored relative to the other teams and the unchanged base checkpoint" [V] — [PS §7, §9.1](file:///home/user/interiit_minips/Sarvam.pdf).
- PS §10 scorer integrity: "reading out alternatives the speaker did not offer" can lead to disqualification [V]. Saying "five crore, that is fifty million" is therefore risky.
- The report does not convert. §A4: "Keeping 'crore' is safe either way". Its §D6 checker compares amounts by value ("five crore", "50 million" and "₹5 crore" all match 5×10⁷) "with a separate unit-preservation flag" [V], report lines 73 and 468.
- **Entity metrics in recent speech-LLM ST work use exact string match.** A 2025 study scores named entities and terms with the NEuroParl-ST script as case-sensitive exact matches, and finds that entity accuracy and xCOMET can diverge sharply (high xCOMET with very low entity accuracy) [S] — [arXiv 2512.16378](https://arxiv.org/pdf/2512.16378).
- **LLM judges are increasingly used next to ASR-BLEU** because ASR-BLEU is biased by the ASR backend. One streaming S2ST paper pairs automatic metrics with listening tests and an LLM-as-judge adequacy protocol [S] — [SimulS2ST-Omni, arXiv 2607.19810](https://arxiv.org/pdf/2607.19810). A SimulMT paper uses a WMT25-style LLM-judge prompt with Gemini-3.1-Pro alongside BLEU, COMET and MetricX [S] — [arXiv 2604.21045](https://arxiv.org/pdf/2604.21045). IWSLT simultaneous S2ST reports ASR-BLEU (Whisper) with ATD, StartOffset and EndOffset [S] — [NAIST IWSLT 2024, arXiv 2407.00826](https://arxiv.org/pdf/2407.00826).
- **Large-unit numbers are a known failure point for translation systems.** For Chinese↔English, numbers with large units ("million", "billion", 亿) reach error rates "as high as 20%" even for Llama-3.1-8B [S] — [arXiv 2501.04927](https://arxiv.org/html/2501.04927v1). Commercial NMT fails CheckList-style number tests on separators and units (example: 100.01 million rendered as 1.001亿 = 100.1 million) [S] — [Findings ACL 2021.415](https://preview.aclanthology.org/fix_video/2021.findings-acl.415.pdf).

### Inferences
- **Likely scorer architecture** (inference, not stated by Sarvam): ASR of the English output, then (a) a reference-based quality metric (BLEU/chrF/COMET or an LLM judge) against the human references; (b) an entity check, either a normaliser plus exact match on entity spans or an LLM judge given the reference entities; (c) an LLM judge or rubric for intent categories; (d) time-aligned lag metrics.
- **Adjudicating "5 crore" → "50 million".** The outcome depends on the entity checker:

| Scorer type | "five crore" | "fifty million" |
|---|---|---|
| Exact string match on the reference entity | correct | wrong |
| Value normaliser (parse to 5×10⁷) | correct | correct |
| LLM judge | correct | probably correct (unverified) |
| ASR on "crore" | Whisper-class ASR may mis-hear "crore" as "core" or "crow", a risk for "crore" | n/a |

  - "Crore" is never worse on the first three. The FAQ's example (25,000 ↔ 25k) shows equivalence *within* a unit system and says nothing about cross-system conversion.
  - The one real risk for "crore" is ASR mis-recognition of a word that is rare in Western ASR training data. That argues for clear Bulbul pronunciation and for testing Whisper-class ASR on "lakh"/"crore", not for conversion.
- **The critique's claim 2, decomposed.**
  - "Violates dataset requirements": **true for the dataset** (PS §3) and could cost mid-term audit points and G3, but **not applicable** to the report, which keeps units.
  - "Risks entity point deduction": **plausible** under an exact-match scorer and unknown otherwise. It is a reason to keep units, which the report already does.
  - The critique attacks a position the report does not hold.
- **Safe recommendation.**
  - Targets and speech say "five crore", "two and a half lakh" or "twenty-five lakh" in the speaker's unit.
  - Never append a conversion: it is redundant, adds latency, and risks the "alternatives" clause.
  - Keep digits optional in text (PS: "digits are also accepted"), but the *spoken* form is what is scored.
  - Use the value-normalising checker internally (as §D6 does) and also report the unit-preservation flag.

### Gaps
- Sarvam has not published the scorer: no ASR model, metric, judge prompt, or rule on unit conversion. Ask at the mid-term or check the format validator when it is released.
- No study measures how LLM judges treat lakh/crore against million-based equivalents. No ASR WER figure for "lakh/crore" in en-IN TTS output was found.

---

## 5. What do human references usually look like, and what do monotonic targets cost on BLEU/COMET? Is the dual-target plan sound?

### Takeaway
Human references for ST test sets are almost always offline, fluent translations. Several NAIST/IWSLT and COMET-variant studies show that offline references **under-score** monotonic or interpretation-style output, and that standard COMET prefers long-distance reordering. Monotonic training targets still improve the latency–quality trade-off for distant pairs. So the report's dual-target plan (a chunk-monotonic `en_text` for training plus a fluent `en_alt`, with the cost measured) is sound, provided the monotonic target stays grammatical and its cost against the fluent version is measured before committing.

### Cited Findings
- Doi, Ko, Makinae, Sudoh and Nakamura (IWSLT 2024), using the NAIST En→Ja chunk-wise monotonic translation (CMT) evaluation set: an existing **SI-based test set underestimates** models, and CMT references **score SimulST models higher than offline models**. Offline-style output had almost no omissions, while human SI omits content [S] — [ACL 2024.iwslt-1.30](https://aclanthology.org/2024.iwslt-1.30.pdf); [arXiv 2406.08940](https://arxiv.org/pdf/2406.08940).
- Simul-COMET (Findings ACL 2026): standard COMET "tends to favor offline-style translations that reorder long-distance material". Simul-COMET is trained on SI-style data derived from COMET's training data with LLMs and evaluated on En–Ja [S] — [ACL 2026.findings-acl.2110](https://aclanthology.org/2026.findings-acl.2110/).
- Simul-MuST-C (EMNLP 2024): offline-style references can bias evaluation towards offline outputs, and the benefit of segment-based monotonicity "depends on the grammatical distance" [S] — [ACL 2024.emnlp-main.1238](https://aclanthology.org/2024.emnlp-main.1238.pdf).
- Zhao et al. 2021, "It is Not as Good as You Think! Evaluating Simultaneous Machine Translation on Interpretation Data", reports a large BLEU gap between interpretation and translation references; earlier notes record **13.83 BLEU** [S, number not re-verified this session] — [arXiv 2110.05213](https://arxiv.org/pdf/2110.05213); see also `research_notes/Sarvam mini PS research/targets_and_timing.md`.
- An automatic SI quality metric paper notes that interpreters "maintain the word order of the source language to minimize latency, whereas translation reorders to keep fluency", and that MT metrics correlate only moderately with human SI quality judgements [S] — [arXiv 2407.06650](https://arxiv.org/pdf/2407.06650); [Findings EACL 2024.15](https://preview.aclanthology.org/fix_video/2024.findings-eacl.15.pdf).
- On the training side, LLM-SI-Corpus (CWMT guideline) "reduces latency while achieving better quality" for En→Ja [S] — [arXiv 2404.12299](https://arxiv.org/abs/2404.12299). Wait-k pseudo-references give up to +2.7 BLEU on Zh→En and Ja→En [S] — [arXiv 2010.11247](https://arxiv.org/abs/2010.11247). Both are cited in the report's §B2 [V].
- The PS allows "several English translations" per sentence, and G2 manifests must list "the target text used" for each training pair [V] — [PS §3, §8](file:///home/user/interiit_minips/Sarvam.pdf).

### Inferences
- **Sarvam's references are probably offline-style**, since they are "human reference translations" written to "encode the corrected meaning". That means fluent and disfluency-free. Expect a quality-KPI penalty (12 points) for heavy restructuring. Most of the cited studies are En→Ja (head-initial to head-final), the mirror image of Hi→En; the direction of the effect should carry over, but the size is unverified for Hindi.
- **The dual-target plan is sound** with three refinements:
  1. Make the monotonic target **"fluent-first, monotonic where cheap"**. Fronted adverbials and verb-final chunking cost little against a fluent reference (option B above keeps the *same words* as the fluent version, only timed differently). Left dislocation and resumptives change the words and should be used only when the object is long.
  2. The cheapest monotonicity is **temporal, not lexical**. With a Hibiki-style model, much of the latency gain comes from *when* the fluent words are placed, so a fluent target with good placement may capture most of the benefit. Measure this first.
  3. Keep the report's decision rule: if monotonic loses more than about 10 BLEU against fluent while saving less than about 0.5 s of mean lag, move toward fluency.
- **Scorer relativity matters.** KPIs are relative to other teams and to the base checkpoint [V], so a small quality loss that buys a large intent or latency gain is worth taking only if competitors are not getting both from fluent-with-placement.

### Gaps
- No Hi→En study of monotonic versus fluent targets was found. The reference style and quality metric Sarvam uses are unknown.
- The 13.83 BLEU figure and the Simul-COMET effect sizes were not re-read from the PDFs.

---

## 6. Do S2ST models trained on interpretation-style or monotonic targets sound less natural?

### Takeaway
No direct study was found that rates the naturalness (MOS) of S2ST output trained on monotonic or interpretation-style *text* targets. The available evidence concerns *timing-induced* unnaturalness (silence insertion, choppy pauses) and the gap between system and human naturalness in Hibiki-class models. The naturalness KPI (6 points) is more exposed to pause structure than to syntax.

### Cited Findings
- Hibiki's human ratings, as recorded in earlier research notes: about 3.73/5 naturalness against 4.18/4.12 for ground truth (Fr→En), via a secondary source [S] — [Pith summary of arXiv 2502.03382](https://pith.science/paper/2502.03382) (see `targets_and_timing.md`).
- NaturalFlow (arXiv 2606.13121) studies chunk-by-chunk pauses that listeners dislike in simultaneous speech output [S, from earlier notes] — cited in report §B3.
- A streaming S2ST paper pairs ASR-BLEU with human listening studies because ASR-BLEU misses perceptual quality [S] — [arXiv 2607.19810](https://arxiv.org/pdf/2607.19810).
- He et al. 2016 show that interpretese differs systematically from translationese: more passives, more segmentation, more generalisation and summarisation [S] — [ACL N16-1111](https://aclanthology.org/N16-1111/). That paper concerns text style, not listener ratings.

### Inferences
- **Option B (subject + adverbial, pause, verb + object) creates exactly the mid-clause pause** that listeners penalise ("Yesterday I… [1.6 s] …got my old bank account closed"). Option C (left dislocation) fills that gap with content. So the naturalness trade-off between B and C is closer than the "LD reads as unnatural" argument suggests: B has a natural syntax with an unnatural pause, and C has a marked syntax with continuous speech.
  - Cheapest test: a small AB listening test on 20 Bulbul renderings of each.
- **Indian English tolerates LD better.** Indian English uses LD about 9× more often than British English (1.42 vs 0.16 per 1,000 words [S]). With an en-IN Bulbul voice, LD may sound less marked than the report assumes. This argues for allowing LD somewhat more often than "rare", but the evidence is indirect.

### Gaps
- No MOS or acceptability data exists for monotonic versus fluent English *speech* in S2ST. No study measures listener tolerance for mid-clause pauses against left dislocation.

---

## Recommendations for thesis wording and target style (synthesis; all inferences from the findings above)

1. **Retitle around commitments.** Suggested: **"Hold the commitment, not the sentence"**, or "Speak what Hindi has settled; hold what it hasn't". The abstract should say: *speak subjects, time and place adverbials and settled noun phrases at once; hold the English verb, polarity, question force and entity values until the Hindi cue that settles each one.* Replace the abstract example with a well-formed one:
   - "Yesterday I… got my old bank account closed" (default), or
   - "Yesterday, my old bank account — I got it closed" (long-object variant).
2. **Default target style, in priority order:**
   - (a) Fluent English that keeps Hindi *chunk* order where English allows: fronted time and place adverbials, subject early.
   - (b) Verb plus object spoken as one chunk after the Hindi verb group.
   - (c) Left dislocation with a resumptive only when the object is long (for example, more than about 4 words or about 1.5 s of Hindi), and allowed more freely than "rare" for an en-IN voice, pending an AB test.
   - (d) The causative "had/got X Ved" *only when the Hindi is causative* (-वा-), keeping the से-intermediary.
   - (e) The passive only when the Hindi is passive or agentless.
   - (f) Never nominalisation chains.
3. **Widen the polarity rule to an intent rule.** Hold the English verb, auxiliary and "not" until the Hindi verb group **plus one token or the clause boundary**, covering post-verbal or emphatic नहीं and dropped vector verbs. Treat question force and tense as commitments too:
   - render intonation-only questions and tags as declarative + "right?";
   - fix "yesterday/tomorrow" for कल only after the ergative -ने or the tense auxiliary.
4. **Units.** Keep lakh/crore in the dataset and in speech; never convert or append conversions. Keep the value-normalising checker plus the unit flag. State in the report that the critique's claim 2 does not apply to this design, and why (PS §3; FAQ shows only same-unit equivalence; exact-match entity metrics exist).
5. **Dual targets stay**, with one added experiment: "fluent words + safe placement" against "chunk-monotonic words + safe placement". This tests whether temporal monotonicity alone captures most of the lag benefit at no cost against the reference.
