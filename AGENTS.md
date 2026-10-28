# Repository rules

## Security

- Never commit `.env` files, credentials, API keys, access tokens, passwords, private keys, or provider secrets.
- Never commit OpenAI, Groq, GitHub, AWS, Google, Azure, or other provider credentials, regardless of the filename containing them.
- Store credentials in environment variables or ignored local configuration files.
- Before every commit or push, inspect the staged files and scan them for potential secrets.
- Do not print secret values in logs, command output, documentation, tests, examples, or chat responses.
- If a secret may have been exposed, stop before pushing, notify the user, and recommend rotating the credential.
- An `.env.example` file may contain variable names and safe placeholders only; it must never contain real credentials.

## Git

- Do not force-push, rewrite published history, or bypass repository protections unless the user explicitly requests it.
- Do not add ignored files with `git add --force`.
