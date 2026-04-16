# Como executar o projeto

1. Clone o Repositório e entre na Pasta Raiz do Projeto
```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DA_PASTA>
```
2. Crie a Ative uma Variável de Ambiente

No Windows PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

No Linux / macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

3. Gerar um Token do Kaggle
- Duplique a pasta `.env.example`
- Renomei a pasta copiada para `.env`
- No Kaggle, acesse sua conta e vá para a sessão de configurações
  - Procure por API Tokens e gere um token  
  - Copie esse token no arquivo `.env`

4. Rode a Pipeline

```bash
python main.py
```

ou 

```bash
dvc repro
```
