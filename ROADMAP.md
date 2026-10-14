# PrepAI roadmap

## Done
- Layer 1: PYQ analysis (hybrid semantic mapping 94%, importance score, recurring patterns)
- Layer 2: difficulty, monotone PFA mastery model, adaptive quiz, time-constrained planner
- Offline question generation (cloze MCQs), Streamlit dashboard
- Multi-agent system: Orchestrator, Exam Analyst, Question Generator, Critic, Answer Evaluator,
  Learner Model, Tutor, Study Planner, Coverage Auditor, Resource Recommender
- AI grading of written answers (offline; LLM via Ollama when installed)
- LLM Coordinator agent (dynamic agent selection with logged reasoning; rule fallback) + LLM Critic check via Ollama
- YouTube video recommendations per topic (search links; ranked videos with free YouTube API key)

## Next
- Trained topic-importance model with leave-last-year-out backtesting (needs 6+ years of papers)
- Forgetting curve in mastery (time since last practice) + spaced repetition
- Auto-graded numerical generators (Banker's, CPU scheduling, page replacement, disk scheduling)
- Multi-day plan to exam date + .ics calendar export
- Mock Exam agent, Rubric agent, Mapping Reviewer agent, Progress Coach agent
- OCR for scanned papers; real-user evaluation with classmates
