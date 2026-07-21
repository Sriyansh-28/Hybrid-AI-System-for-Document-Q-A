# Serving image for the merged, domain-adapted model.
# Build the merged weights first (scripts/merge_and_export.py), then mount or
# copy them in and point MODEL_PATH at them.

FROM python:3.10-slim

WORKDIR /app

# Install deps first so Docker can cache this layer across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/models/merged

EXPOSE 8000

# Mount your merged weights at /models/merged at runtime, e.g.
#   docker run -p 8000:8000 -v $(pwd)/outputs/merged:/models/merged llm-post-training
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
