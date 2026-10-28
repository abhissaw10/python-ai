# Python AI Projects

A growing collection of small Python applications demonstrating practical AI
concepts with Groq, OpenAI-compatible APIs, LangChain, Beautiful Soup, and
Gradio.

This repository currently includes:

- A news website summarizer built directly with an OpenAI-compatible client.
- A LangChain version using a prompt template, `ChatGroq`, and
  `StrOutputParser`.
- A conversational chatbot with session-based, in-memory recall.
- Small OpenAI and Groq API examples.

## Projects

### News website summarizer

[`websitesummarizer.py`](websitesummarizer.py) fetches a news page, extracts its
readable text with Beautiful Soup, and asks a Groq-hosted model for the five
leading news highlights. It supports both a terminal command and a Gradio UI.

```bash
python websitesummarizer.py "https://example.com/news"
python websitesummarizer.py
```

### LangChain news summarizer

[`LangChainSummarizer.py`](LangChainSummarizer.py) implements the same workflow
as a LangChain Expression Language pipeline:

```text
ChatPromptTemplate -> ChatGroq -> StrOutputParser
```

Run it from the terminal or open its Gradio UI:

```bash
python LangChainSummarizer.py "https://example.com/news"
python LangChainSummarizer.py
```

### In-memory chatbot

[`simple_chatbot.py`](simple_chatbot.py) demonstrates short-lived conversational
memory. Statements are stored for the current Gradio session, and questions are
answered only from that saved conversation.

```bash
python simple_chatbot.py
```

The memory is intentionally temporary and is cleared when the session or
application is restarted.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your own provider keys to the local `.env` file:

```dotenv
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

Most applications use `GROQ_API_KEY`. `OPENAI_API_KEY` is needed only for the
direct OpenAI example. Provider usage may incur charges or consume account
quota.

## Security

- Real credentials are never included in this repository.
- `.env`, private keys, virtual environments, and common local editor files are
  excluded by [`.gitignore`](.gitignore).
- Each person who clones the repository must supply their own API credentials.
- Do not put a real credential in source code, examples, documentation, commits,
  issues, or screenshots.
- The optional `--share` flag creates a public Gradio link. The example apps do
  not include authentication or rate limiting, so use that flag only when you
  understand that visitors can consume the configured provider quota.
- The summarizers fetch user-provided URLs. Keep the UI local and submit only
  trusted public URLs unless additional network protections are added.

If a credential is ever committed, revoke it at the provider immediately and
remove it from the complete Git history; deleting it in a later commit is not
enough.

## License

This project is available under the [MIT License](LICENSE). You may use, modify,
and distribute it as described in the license, while retaining the copyright and
license notice.
