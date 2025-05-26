FROM ghcr.io/mlflow/mlflow:v2.22.0

WORKDIR /mlflow

EXPOSE 8080

ENV MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlruns
ENV PORT=8080

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "./mlruns"]
