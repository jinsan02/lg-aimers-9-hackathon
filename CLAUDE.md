# CLAUDE.md → [AGENTS.md](AGENTS.md)

**This project keeps one entry point. Read [AGENTS.md](AGENTS.md) now — it holds the
competition spec, the machines, the non-negotiables, and the handoff procedure.**

This file exists only because Claude Code auto-loads `CLAUDE.md`. It should be a
symlink to `AGENTS.md`, but a real symlink needs `core.symlinks=true` plus Windows
Developer Mode, and `mklink` is currently denied on this laptop. Git would then
check the symlink out as a plain text file containing the literal string
`AGENTS.md` — silently replacing the rules with one word.

To convert it once Developer Mode is on
(Settings → Privacy & security → For developers):

```bash
git config core.symlinks true && rm CLAUDE.md && ln -s AGENTS.md CLAUDE.md && git add -A CLAUDE.md
```

Do not add content here. Everything belongs in `AGENTS.md`.
