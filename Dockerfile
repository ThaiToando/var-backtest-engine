FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (separate layer) so Docker can cache this step
# and only reinstall packages when requirements.txt actually changes --
# much faster rebuilds during development than copying all code first.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project
COPY . .

# Default command: run the fast core pipeline. Override with
# `docker run <image> python -m src.backtest` to run the full backtest instead.
CMD ["python", "run_pipeline.py"]