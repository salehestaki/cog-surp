FROM python:3.12.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1

RUN python -m pip install --no-cache-dir uv==0.11.32

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --extra all --no-dev

ENTRYPOINT ["/workspace/.venv/bin/cog-surp"]
CMD ["doctor"]
