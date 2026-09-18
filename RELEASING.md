# Releasing CitePulse

A small, single-maintainer release process. No PyPI publishing today —
CitePulse is installed from source (`pip install -e .`) or the Windows
zip build.

1. Bump `version` in [`pyproject.toml`](pyproject.toml).
2. Tag and push:
   ```bash
   git tag vX.Y.Z
   git push --tags
   ```
3. Build the Windows distributable:
   ```bash
   build_dist.bat
   ```
   This produces `dist\citepulse-win-x64-<version>.zip`.
4. Create a [GitHub Release](https://github.com/alsanjayllm/CitePulse-public/releases/new)
   for the tag, and attach the zip from step 3.
5. Use the commit log since the last tag as release notes — there's no
   separate CHANGELOG file to keep in sync.
