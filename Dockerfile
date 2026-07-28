FROM python:3.12.13-slim-bookworm

ARG UV_VERSION=0.11.32
LABEL org.opencontainers.image.title="Cog-Surp" \
      org.opencontainers.image.description="Reproducible surprisal-N400 research workbench" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://github.com/salehestaki/cog-surp" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    PATH="/workspace/.venv/bin:${PATH}"

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}" \
    && groupadd --system --gid 10001 cogsurp \
    && useradd --system --uid 10001 --gid cogsurp \
       --create-home --home-dir /home/cogsurp cogsurp

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md LICENSE NOTICE CITATION.cff ./
COPY src ./src
COPY configs ./configs
COPY demo ./demo
COPY tests ./tests

# The default image is intentionally CPU/demo focused. Dataframe, Parquet, and
# dashboard extras are installed from the lock; model/EEG research stacks and
# CUDA packages are not part of this lightweight public-demo image. Dev tools
# remain available for the documented offline manifest smoke test.
RUN uv sync --locked --extra data --extra dashboard \
    && mkdir -p /workspace/artifacts \
    && chown -R cogsurp:cogsurp /workspace /home/cogsurp

USER cogsurp
EXPOSE 8501

ENTRYPOINT ["cog-surp"]
CMD ["doctor"]
