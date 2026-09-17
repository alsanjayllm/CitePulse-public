# CitePulse

[![Tests](https://github.com/alsanjayllm/CitePulse-public/actions/workflows/test.yml/badge.svg)](https://github.com/alsanjayllm/CitePulse-public/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Is your site cited by AI answers — and can an AI agent actually *use* it,
not just read about it?

CitePulse is a small, local-first, open-source AEO (Answer Engine
Optimization) audit tool. It runs entirely on your own machine — no
account, no API key, no data sent to a third party — using a local Ollama
model instead of a paid LLM API by default.

**Status: v1 complete, v2 in progress.** Six KPIs are wired end to end
today: llms.txt Readiness, AI Crawl Accessibility, Citation Rate, AI
Share of Voice, Task Completion Success Rate, and Interaction Readiness —
plus per-run/multi-model comparison (local Ollama or an optional
OpenRouter cloud model), a KPI subset picker, an N-run trend view, and a
"v1 core" zip-and-run Windows build for a locked-down corporate machine.
Every KPI follows the same design principle: **never fabricate a score**.
When something can't be measured (a blocked fetch, a rate-limited
request, an ambiguous LLM answer), CitePulse reports "not determined,"
never a guessed number. See [`docs/PACKAGING.md`](docs/PACKAGING.md) for
the Windows build, and [CONTRIBUTING.md](CONTRIBUTING.md) if you want to
add a KPI or open a PR.

## Why CitePulse

Most AEO/GEO visibility tools are cloud SaaS: you send them your brand and
competitor data, and pay per query against a commercial AI API. CitePulse
is the opposite on both counts:

- **100% local and private** — nothing about your site or your prompts
  leaves your machine.
- **Zero recurring cost** — a local Ollama model by default; bring your own
  OpenAI/Anthropic/Perplexity key only if you want to test against a real
  commercial engine instead. Web search for citation checks works the same
  way: free DuckDuckGo/Google News out of the box, with an optional
  Serper/Tavily/Bing key only if you want to extend past DuckDuckGo's
  free-tier limits.
- **Open source and auditable** — read exactly what it measures and how.
- **Honest about what it's measuring** — CitePulse's citation-rate/
  share-of-voice metrics reflect *this run's own model* synthesizing
  answers over live web search results — that's a useful, repeatable
  proxy, but it is explicitly not a direct measurement of what ChatGPT,
  Perplexity, or Google's AI Overviews would actually say. Every report
  says so.
- **AI Assistant Task Readiness** — beyond "are you cited," CitePulse tests
  whether an autonomous AI agent can actually complete a task on your site
  (find pricing, fill a form). Almost no other tool, free or paid, tests
  this today.

## How CitePulse compares

CitePulse isn't trying to out-feature the big AEO/GEO SaaS platforms — it's
solving a different problem (local, free, auditable) for a different buyer
(a solo developer, a privacy-conscious team, or a locked-down corporate
machine that can't send brand data to a third party). For context, here's
how it stacks up against the three platforms most commonly named as
category leaders in 2026 buyer comparisons: [Profound](https://www.tryprofound.com/),
[AthenaHQ](https://athenahq.ai/), and [Scrunch AI](https://scrunch.com/).

| | **CitePulse** | Profound | AthenaHQ | Scrunch AI |
|---|---|---|---|---|
| Deployment | Local, self-hosted (your machine) | Cloud SaaS | Cloud SaaS | Cloud SaaS |
| Starting price | **$0** (local Ollama) | $99/mo (ChatGPT-only tier) | $295/mo | $250/mo |
| Data leaves your machine? | No — nothing sent to a third party by default | Yes | Yes | Yes |
| Open source | Yes (MIT) | No | No | No |
| AI engines tracked | ChatGPT, Perplexity, Gemini, or any OpenRouter model, plus a free local model | ChatGPT, Perplexity, Gemini, Google AI Overviews | ChatGPT, Perplexity, Gemini, Google AI Overviews | ChatGPT, Perplexity, Gemini, Google AI Overviews |
| Citation rate / AI share of voice | Yes | Yes | Yes | Yes |
| llms.txt / AI crawl accessibility | Yes | Partial | Partial | Yes (AXP) |
| AI agent task-completion testing (can an agent actually *use* your site, not just cite it) | Yes | No | No | No |
| Multi-model / cross-engine comparison in one run | Yes | Yes | Yes | Yes |
| Enterprise compliance (SOC 2, etc.) | N/A — no data ever leaves your machine | Not published | Not published | SOC 2 Type II |
| Best fit | Developers, indie sites, privacy-first teams, corporate-LAN environments | Enterprise marketing/PR/SEO teams wanting prompt-volume depth | Mid-market AEO/SEO teams wanting guided reporting | Enterprise teams wanting monitoring + optimization + AI content delivery |

Pricing and feature details for Profound, AthenaHQ, and Scrunch AI are
publicly listed and change frequently — see each vendor's site for current
numbers. CitePulse's own numbers above are the only ones we control, and
they're pulled straight from this repo.

## System requirements

- Python 3.11+
- 8GB RAM minimum, 16GB recommended (for running a local LLM comfortably)
- ~5-10GB free disk (model weights + browser download)
- No GPU required — CPU-only works, just slower. An NVIDIA/Apple/AMD GPU is
  used automatically by Ollama if present.

## Quick start

```bash
git clone https://github.com/alsanjayllm/CitePulse-public.git
cd CitePulse-public
pip install -e ".[dev]"
citepulse setup
citepulse audit https://example.com
```

## Portable Windows download (no admin, no Python required)

No corporate machine, no problem: a standalone Windows build needs
nothing installed — no Python, no pip, no admin rights, no installer.

1. Build it (or grab a build someone else produced): `build_dist.bat`
   produces `dist\citepulse-win-x64-<version>.zip`.
2. Unzip it anywhere you can write — Desktop, a USB drive, wherever.
   Keep all the files in the unzipped folder together; don't move just
   `citepulse.exe` on its own, it needs the rest of the folder alongside
   it.
3. Run it from inside that folder:

```bash
citepulse.exe setup
citepulse.exe audit https://example.com
```

Windows will show two unsigned-app warnings the first time you run it —
both expected for an open-source tool without a paid code-signing
certificate, neither a sign anything is wrong:

1. Double-clicking `Start CitePulse.bat` shows an **"Open File – Security
   Warning"** dialog ("Unknown Publisher") — click **Run**.
2. Once the app starts listening locally, Windows Defender Firewall
   shows **"has blocked some features of this app"** for `citepulse.exe`
   — click **Allow access**. CitePulse only listens on `127.0.0.1`
   (your own machine), never on the network, regardless of what you
   click here.

## Example output

Real output from `citepulse audit`, not a mockup (run IDs/timestamps
trimmed for readability):

A site with no `llms.txt`:

```
# CitePulse Report — https://example.com
Run: ... | Status: completed | Completed: ...

## KPI #46 — llms.txt Readiness
Value: 0.0 score_0_to_3 | Band: critical
**Remediation** (high severity): No llms.txt file was found at
https://example.com/llms.txt (checked https://example.com/llms.txt,
https://example.com/.well-known/llms.txt). Publish a plain-text or
Markdown file at /llms.txt listing your site's key pages for AI agents
and crawlers, starting with a top-level title (H1) and organized under
H2 section headers with linked pages underneath each.
_Business KPI context (illustrative): Technical debt index (score) (spacelift)_
```

A site that already publishes one well:

```
# CitePulse Report — https://github.com
Run: ... | Status: completed | Completed: ...

## KPI #46 — llms.txt Readiness
Value: 3.0 score_0_to_3 | Band: best_in_class
No gap detected — nothing to remediate.
_Business KPI context (illustrative): Technical debt index (score) (spacelift)_
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md) for what's in scope and how to report a
concern.

## License

MIT — see [LICENSE](LICENSE).
