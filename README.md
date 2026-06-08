# socratOT

Socratic tutor for OT students learning anatomy and neuroscience. Uses GPT-4o and RAG to ask guiding questions instead of giving direct answers.

## Setup

```bash
git clone https://github.com/bahodir4/multimodal-socratic-tutor.git
cd multimodal-socratic-tutor
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add your OPENAI_API_KEY to .env
python scripts/ingest_corpus.py --sample
streamlit run src/app/main.py
```

## Usage

- **Chat** — ask anatomy questions, the tutor guides you with hints
- **Image analysis** — upload any anatomy diagram, get Socratic questions
- **Dashboard** — see your mastery scores and weak topics
- **Settings** — set your name, OT level, toggle TTS/STT

## Running tests

```bash
pytest tests/ -v
```

## Evaluation

```bash
python src/evaluation/run_evaluation.py --quick
python src/evaluation/run_evaluation.py
```

Results go to `src/evaluation/results/`.

## Docker

```bash
docker compose up --build -d
```

## Project structure

```
src/
  app/          Streamlit pages (chat, dashboard, image analysis, settings)
  core/         Socratic engine, RAG pipeline, student memory, assessment
  models/       LLM and embedding wrappers
  schemas/      Pydantic models
  config/       Settings and prompts
  evaluation/   RAGAS and compliance evaluators
  utils/        Logging and helpers
scripts/        Corpus ingestion scripts
tests/          Unit and integration tests
```

## License

MIT
