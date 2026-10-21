# LumiCycle Team Study Pack

Release update (21 October 2026): first read [current architecture](docs/CURRENT_MODEL_ARCHITECTURE.md)
and [changes](docs/CHANGES.md). The public feature is Day → Night only. New distilled
student options use explicitly external Turbo supervision; older bidirectional
descriptions refer to training history rather than supported demo behavior.

Use this order so the team learns the implemented project before reading abandoned experiments.

## Essential reading

1. `docs/PROJECT_MASTER_GUIDE.md` — complete source-grounded explanation of the current best model,
   training, losses, failures, limitations, and professor-style questions.
2. `docs/CURRENT_MODEL_ARCHITECTURE.md` — concise architecture and experiment history.
3. `docs/VIVA.md` — short answers for common viva questions.
4. `PROJECT_PROCESS.md` — chronological build and training diary.
5. `docs/ARCHITECTURE.md` — architecture and data-flow diagrams.
6. `docs/TEAM_VIDEO_WATCHLIST.md` — short beginner-friendly videos and post-video questions.

## Presentation and demonstration

7. `docs/PRESENTATION.md` — presentation outline and speaking material.
8. `docs/DEMO_CHECKLIST.md` — live-demo preparation and fallback checklist.
9. `docs/REPORT.md` — college-report draft.
10. `README.md` — installation, training, evaluation, and demo commands.
11. `remote-demo/README.md` — how the Vercel interface and home-PC GPU worker communicate.

## Failed experiments and future work

12. `V2_1_IMPLEMENTATION.md` — V2.1 detail-refinement experiment; retained as a failed ablation.
13. `docs/LUMIRENDER_FUNDAMENTALS.md` — physics-guided relighting fundamentals.
14. `docs/LUMIRENDER_LITERATURE_REVIEW.md` — literature notes behind LumiRender.
15. `docs/LUMIRENDER_PRESENTATION.md` — presentation material for that experiment.
16. `docs/LUMIRENDER_VIVA.md` — questions about Gaussian lights, reflections, and rendering.
17. `V3_GENERAL_OUTDOOR_PLAN.md` — unimplemented plan for broader outdoor generalization.
18. `DEEP_RESEARCH_HANDOFF.md` — background supplied for additional research.
19. `docs/RESULTS_TEMPLATE.md` — template for recording formal evaluation results.

## Facts every member must know

- The main method is a modified CycleGAN for unpaired image-to-image translation.
- The current showcase model is LumiCycle V2 checkpoint `step_00004500.pt`.
- V2 was warm-started from V1 step 13,000; it was not trained entirely from scratch.
- V2 step 12,000 is the final checkpoint, but step 4,500 has the best internal validation score.
- The 5,500-step checkpoint belongs to failed V2.1, not V2.
- LumiRender is an honest failed experiment that mostly learned global darkening.
- Established components must be attributed. The contribution is the implementation, integration,
  V2 modifications, training system, experiments, failure analysis, and deployment.
- Do not claim state-of-the-art performance or physically correct relighting.
