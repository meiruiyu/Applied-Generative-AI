FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /code

COPY pyproject.toml uv.lock README.md /code/
COPY src /code/src
COPY app /code/app
COPY helper_lib /code/helper_lib
COPY cnn_api.py /code/cnn_api.py
COPY models /code/models

RUN uv sync --frozen

EXPOSE 80
CMD ["uv", "run", "--frozen", "fastapi", "run", "app/main.py", "--port", "80"]