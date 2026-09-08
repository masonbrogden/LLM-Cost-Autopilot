# Project plan

This repository is meant to be a small LLM cost autopilot service.

## Scaffolded structure

- app/
  - __init__.py
  - registry.py
  - config.py
  - classifier.py
  - router.py
  - evaluator.py
  - schemas.py
  - logging_db.py
  - main.py
  - providers/
    - __init__.py
    - base.py
    - ollama.py
    - paid.py
  - routes/
    - __init__.py
    - proxy.py
- dashboard/
  - __init__.py
  - app.py
- data/
  - .gitkeep
- scripts/
  - __init__.py
  - smoke_providers.py
  - run_benchmark.py
  - seed_prompts.py
- tests/
  - __init__.py
  - test_registry.py
  - test_providers.py
  - test_router.py
- .env.example
- .gitignore
- requirements.txt
- README.md
- Dockerfile
- docker-compose.yml

## Notes

- No application logic is implemented yet.
- This scaffold is intentionally minimal and empty, with package markers and project layout only.
- The real work is to be completed in the sequence described in claude-code-prompts.md.
