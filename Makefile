check:
	uv run pytest -q tests > logs/pytest.log 2>&1; uv run python check.py
