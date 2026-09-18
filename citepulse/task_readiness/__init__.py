"""AI Assistant Task Readiness (#48 Task Completion Success Rate, #58
Interaction Readiness): a live browser-driving agent loop that attempts
per-site, LLM-generated tasks against the audited site -- distinct from
the static crawl (#46) and text-prompt-only AI-visibility testing (#22/
#24). See harness.py's module docstring for the independent-success-
verification design this module is built around.

Deliberately scoped down: no task_library.py (every task comes from
task_generator.py, never a hand-authored file) and no multi-engine
runner.py orchestration (CitePulse has exactly one engine, local Ollama).
"""
