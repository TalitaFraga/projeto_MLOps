# Projeto MLOps - Gen Z Burnout Prediction

Este projeto implementa um fluxo completo de MLOps para predição de risco de burnout em jovens da Geração Z. A solução contempla extração de dados diretamente do Kaggle, processamento ETL, treinamento de modelo de Machine Learning, rastreamento de experimentos com MLflow, versionamento de dados/modelos com DVC e DagsHub, além de uma API com frontend simples para consumo do modelo.

## Objetivo

O objetivo do projeto é construir um pipeline de MLOps capaz de:

* baixar automaticamente um dataset do Kaggle;
* executar etapas de ETL;
* treinar um modelo de Machine Learning;
* aplicar balanceamento de classes com SMOTE;
* registrar métricas, parâmetros e modelo no MLflow;
* versionar dados, métricas e modelo com DVC;
* enviar os artefatos versionados para o DagsHub;
* disponibilizar uma API de predição;
* permitir uso da API por meio de um frontend simples;
* executar a solução via Docker Compose.

## Tecnologias utilizadas

* Python
* Pandas
* Scikit-learn
* Imbalanced-learn
* SMOTE
* MLflow
* DVC
* DagsHub
* KaggleHub
* FastAPI
* Uvicorn
* Docker
* Docker Compose
* HTML, CSS e JavaScript

## Estrutura do projeto

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── metrics/
│   └── RF_results.json
├── models/
│   └── random_forest_model.pkl
├── src/
│   ├── Extract/
│   │   └── extract.py
│   ├── Load/
│   │   └── load.py
│   ├── Transform/
│   │   └── transform.py
│   ├── Train/
│   │   ├── preprocess.py
│   │   └── train.py
│   ├── api.py
│   ├── config.py
│   └── etl.py
├── .dvc/
│   └── config
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yaml
├── dvc.yaml
├── dvc.lock
├── main.py
├── params.yaml
├── README.md
└── requirements.txt
```

## Dataset

O dataset utilizado é baixado diretamente do Kaggle por meio da biblioteca `kagglehub`.

Dataset:

```text
hammadansari7/gen-z-mental-wellness-and-digital-lifestyle-patterns
```

A configuração do dataset está no arquivo `params.yaml`:

```yaml
dataset:
  kaggle_dataset_id: hammadansari7/gen-z-mental-wellness-and-digital-lifestyle-patterns
```

## Configuração do ambiente

Antes de executar o projeto, crie um arquivo `.env` a partir do `.env.example`.

No Windows PowerShell:

```powershell
copy .env.example .env
```

No Linux/macOS:

```bash
cp .env.example .env
```

Depois, preencha o `.env` com suas credenciais reais.

Exemplo de estrutura:

```env
# Kaggle
KAGGLE_USERNAME=seu_usuario_kaggle
KAGGLE_KEY=sua_chave_kaggle

# DagsHub / DVC
DAGSHUB_USERNAME=seu_usuario_dagshub
DAGSHUB_TOKEN=seu_token_dagshub
DAGSHUB_REPO_URL=https://dagshub.com/seu_usuario/seu_repositorio

# MLflow
MLFLOW_TRACKING_URI=https://dagshub.com/seu_usuario/seu_repositorio.mlflow
MLFLOW_TRACKING_USERNAME=seu_usuario_dagshub
MLFLOW_TRACKING_PASSWORD=seu_token_dagshub
MLFLOW_EXPERIMENT_NAME=genz_burnout_prediction

# Modelo usado pela API
MLFLOW_MODEL_URI=models:/champion-model/latest
```

## Atenção sobre credenciais

O arquivo `.env` não deve ser enviado para o GitHub ou DagsHub, pois contém credenciais privadas.

Garanta que o `.gitignore` contenha:

```gitignore
.env
.dvc/config.local
.venv/
__pycache__/
data/
models/
metrics/
.dvc/cache/
```

## Pipeline MLOps

O pipeline é controlado pelo DVC e possui duas etapas principais:

### 1. ETL

A etapa de ETL executa:

* download do dataset no Kaggle;
* leitura dos dados brutos;
* transformação das colunas categóricas com `get_dummies`;
* remoção de colunas configuradas;
* salvamento do dataset processado.

Comando definido no `dvc.yaml`:

```yaml
etl:
  cmd: python -m src.etl
```

### 2. Treinamento

A etapa de treinamento executa:

* leitura do dataset processado;
* separação entre features e target;
* divisão treino/teste;
* aplicação do SMOTE no conjunto de treino;
* treinamento do modelo Random Forest;
* cálculo de métricas;
* salvamento do modelo;
* registro de métricas, parâmetros e artefatos no MLflow;
* registro do modelo como `champion-model`.

Comando definido no `dvc.yaml`:

```yaml
train:
  cmd: python -m src.Train.train
```

## Modelo utilizado

O modelo final utilizado no projeto é:

```text
RandomForestClassifier
```

A Decision Tree foi removida do pipeline, mantendo somente o Random Forest.

O modelo é registrado no MLflow com o nome:

```text
champion-model
```

A API carrega o modelo a partir do MLflow usando:

```env
MLFLOW_MODEL_URI=models:/champion-model/latest
```

## Execução com Docker Compose

O projeto pode ser executado com Docker Compose.

Para construir e subir os serviços:

```bash
docker compose up --build
```

Esse comando executa os serviços definidos no `docker-compose.yaml`.

O serviço `pipeline` executa:

```text
dvc repro
dvc push
```

Ou seja, ele roda o pipeline e envia os artefatos para o DagsHub via DVC.

O serviço `api` sobe a aplicação FastAPI na porta `8000`.

Após a inicialização, acesse:

```text
http://localhost:8000
```

## Serviços do Docker Compose

O projeto possui dois serviços principais:

### pipeline

Responsável por:

* configurar o remote DVC do DagsHub;
* autenticar o DVC com usuário e token;
* executar `dvc repro`;
* executar `dvc push`.

Esse container é de execução única. Após finalizar o pipeline, ele pode aparecer como encerrado no Docker Desktop, o que é esperado.

### api

Responsável por:

* subir a API FastAPI;
* servir o frontend;
* disponibilizar o endpoint `/predict`;
* carregar o modelo pelo MLflow.

Esse container permanece rodando.

## Execução separada

Também é possível executar cada parte separadamente.

Para rodar apenas o pipeline:

```bash
docker compose run --rm pipeline
```

Para subir apenas a API:

```bash
docker compose up api
```

## Execução local sem Docker

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o pipeline com DVC:

```bash
dvc repro
```

Envie os artefatos para o DagsHub:

```bash
dvc push
```

Execute a API localmente:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Depois acesse:

```text
http://localhost:8000
```

## API

A API foi implementada com FastAPI.

Endpoints principais:

### GET /

Retorna o frontend da aplicação.

```text
http://localhost:8000
```

### GET /health

Verifica se a API está rodando.

```text
http://localhost:8000/health
```

### POST /predict

Realiza a predição usando o modelo carregado pelo MLflow.

```text
http://localhost:8000/predict
```

A documentação automática da API está disponível em:

```text
http://localhost:8000/docs
```

## Frontend

O projeto possui um frontend simples feito em HTML, CSS e JavaScript.

O frontend permite:

* preencher os dados de entrada do modelo;
* selecionar variáveis categóricas por dropdown;
* converter automaticamente os dropdowns em colunas no formato `get_dummies`;
* visualizar o JSON enviado para a API;
* enviar os dados para o endpoint `/predict`;
* visualizar o resultado da predição.

As variáveis categóricas são enviadas no formato booleano esperado pelo modelo.

Exemplo para gênero:

```json
{
  "Gender_Female": false,
  "Gender_Male": true,
  "Gender_Non-binary": false
}
```

## Exemplo de JSON enviado para predição

```json
{
  "features": {
    "Age": 24,
    "Daily_Social_Media_Hours": 4.81,
    "Screen_Time_Hours": 6.93,
    "Night_Scrolling_Frequency": 2.61,
    "Online_Gaming_Hours": 2.07,
    "Exercise_Frequency_per_Week": 5.41,
    "Daily_Sleep_Hours": 6.84,
    "Caffeine_Intake_Cups": 1.52,
    "Study_Work_Hours_per_Day": 11.42,
    "Overthinking_Score": 4.95,
    "Anxiety_Score": 4.13,
    "Mood_Stability_Score": 5.74,
    "Social_Comparison_Index": 4.67,
    "Sleep_Quality_Score": 6.27,
    "Motivation_Level": 6.13,
    "Emotional_Fatigue_Score": 6.45,
    "Wellbeing_Index": 4.28,
    "Gender_Female": false,
    "Gender_Male": true,
    "Gender_Non-binary": false,
    "Student_Working_Status_Both": false,
    "Student_Working_Status_Student": false,
    "Student_Working_Status_Working": true,
    "Content_Type_Preference_Educational": false,
    "Content_Type_Preference_Entertainment": false,
    "Content_Type_Preference_Gaming": false,
    "Content_Type_Preference_Lifestyle": false,
    "Content_Type_Preference_News": true
  }
}
```

## MLflow

O projeto utiliza o MLflow para registrar:

* parâmetros do treinamento;
* métricas do modelo;
* artefatos;
* modelo treinado;
* modelo registrado como `champion-model`.

O tracking server utilizado é o MLflow remoto do DagsHub:

```text
https://dagshub.com/seu_usuario/seu_repositorio.mlflow
```

A API acessa o modelo pelo MLflow, e não diretamente pela pasta local `models/`.

## DVC e DagsHub

O DVC é utilizado para versionar:

* dados brutos;
* dados processados;
* modelo treinado;
* métricas.

O remote do DVC aponta para o DagsHub:

```text
https://dagshub.com/seu_usuario/seu_repositorio.dvc
```

Para baixar artefatos versionados em outra máquina:

```bash
dvc pull
```

Para enviar artefatos atualizados:

```bash
dvc push
```

No Docker Compose, o `dvc push` já é executado automaticamente pelo serviço `pipeline`.

## Arquivos importantes

### `params.yaml`

Define parâmetros do dataset, caminhos, transformação, treinamento, modelo e artefatos.

### `dvc.yaml`

Define as etapas do pipeline MLOps.

### `dvc.lock`

Registra o estado atual dos artefatos e dependências do pipeline.

### `src/Train/train.py`

Contém a lógica de treinamento do Random Forest, registro no MLflow e salvamento de métricas/modelo.

### `src/api.py`

Contém a API FastAPI e o endpoint `/predict`.

### `frontend/`

Contém a interface web utilizada para preencher dados e enviar predições.

## Evidências esperadas

Para apresentação ou entrega, recomenda-se capturar evidências de:

1. Docker Desktop com o container da API rodando;
2. execução do `docker compose up --build`;
3. frontend disponível em `http://localhost:8000`;
4. predição funcionando no frontend;
5. endpoint `/docs` da FastAPI;
6. experimento registrado no MLflow/DagsHub;
7. modelo registrado como `champion-model`;
8. artefatos versionados no DagsHub via DVC.

## Comandos principais

Construir e subir tudo:

```bash
docker compose up --build
```

Rodar somente o pipeline:

```bash
docker compose run --rm pipeline
```

Subir somente a API:

```bash
docker compose up api
```

Rodar o pipeline localmente:

```bash
dvc repro
```

Forçar reexecução do pipeline:

```bash
dvc repro -f
```

Enviar artefatos para o DagsHub:

```bash
dvc push
```

Baixar artefatos do DagsHub:

```bash
dvc pull
```

Acessar frontend:

```text
http://localhost:8000
```

Acessar documentação da API:

```text
http://localhost:8000/docs
```

## Status do projeto

O projeto atende aos principais requisitos de MLOps:

* pipeline versionado com DVC;
* dados baixados automaticamente do Kaggle;
* ETL automatizado;
* treinamento com Random Forest;
* SMOTE aplicado no conjunto de treino;
* métricas e artefatos registrados no MLflow;
* modelo registrado como `champion-model`;
* integração com DagsHub;
* API containerizada com FastAPI;
* frontend para consumo do modelo;
* execução via Docker Compose.
