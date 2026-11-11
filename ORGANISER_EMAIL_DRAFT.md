# Draft email to the organisers (optional — team decision)

Send to: hackathon.talentarena@mobileworldcapital.com · Fill in the [brackets] · All teammates should agree first.

---

**Subject:** [TEAM NAME] — questions on the 3 November rules clarification

Hello,

We are [TEAM NAME]. Thank you for the 3 November clarification. We want to follow it strictly. Short yes/no answers to these questions would let us finish:

1. **Corruption rate in external training data.** We train on external matches (DFL, SkillCorner) that we degrade synthetically to look like automatic tracking. May that ball-dropout rate be set from DATA.md's published figure ("ball missing on 41% of graded frames")? No ground truth is involved.
2. **Label-free fitting.** May any parameter be fitted on the provided tracking without labels (self-training, test-time adaptation)? We will not do this unless you say yes.
3. **Hand-set Smart Tagging constants.** No external dataset contains Metrica Smart Tagging. May rules that read it use constants set by hand from the documentation and the Laws of the Game, never tuned on competition data?
4. **Earlier designs.** May our final model keep feature and decoder *designs* we first explored before your clarification, if every model and parameter is re-fitted and re-selected on external data only?
5. **Metrica sample data.** The metrica-sports/sample-data repository has no licence file. May we use it, with attribution, to calibrate or validate?
6. **Research repository.** May our earlier research stay in a separate, archived repository, available to you on request, while the submission repository contains only the final pipeline?

*[Optional disclosure paragraph — include only if the whole team agrees:]*
For transparency: before your clarification, some of our research trained and tuned models on the public 5-minute window, and on 24 October we ran a short hand-annotation pilot on the provided game. That work was dropped. Our submission is trained, tuned and selected on external data only and does not use any of it. Earlier, we also used hosted coding assistants in a folder that contained the match data. We now keep the match data out of any AI tool's reach.

If anything here affects our eligibility, please tell us and we will comply.

Best regards,
[NAME], on behalf of [TEAM NAME]

---

## What we do if they don't answer (internal, don't send)

| # | Default |
|---|---|
| 1 | Keep the current 60% visibility in the shipped models and disclose it in the tech doc; new models use a wide, pre-declared range |
| 2 | Don't do it |
| 3 | Don't use Smart Tagging |
| 4 | Keep designs, re-select inherited constants on external data, disclose lineage |
| 5 | Don't use Metrica sample data |
| 6 | Keep the archived research repository separate |
