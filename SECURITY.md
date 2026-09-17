# Security Policy

CitePulse's core pitch is that it runs entirely on your own machine and
sends nothing to a third party by default. That claim is only worth
anything if it's easy to check, and easy to challenge if it's ever wrong
— read the source, it's the whole point.

## Reporting a vulnerability

Please open a [GitHub issue](https://github.com/alsanjayllm/CitePulse-public/issues)
describing the concern. This is a small, early-stage, single-maintainer
project with no other users' data at stake (everything runs locally, in
your own SQLite database), so there's no separate private disclosure
channel yet. If a report turns out to involve something more sensitive,
say so in the issue and we'll figure out a private channel from there.

## What's in scope

- Anything that would make CitePulse send data somewhere the user didn't
  ask it to
- Anything that would let a hostile *audited* website (the URL you point
  CitePulse at) compromise the tool itself or the machine it runs on
- Secrets or credentials handled unsafely

## What's not

- A heuristic secret scanner missing an unusual pattern — defense in
  depth, not a guarantee
- Issues that only affect the operator's own machine via their own
  deliberate input (e.g. auditing a URL they typed themselves) — not a
  meaningful attack surface for a single-user local CLI tool
