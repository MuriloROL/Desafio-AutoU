# Classificador de Emails (Flask + Transformers)

Uma app simples que classifica o conteúdo de emails (texto ou PDF/TXT) em categorias financeiras usando Zero-Shot Classification da Hugging Face.

## Como rodar

1. Crie e ative um ambiente virtual (opcional, recomendado)
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Instale dependências (primeira vez pode demorar por causa de `torch`):
   ```bash
   pip install -r requirements.txt
   ```
3. Rode a aplicação
   ```bash
   python app.py
   ```
4. Acesse no navegador:
   - http://localhost:5000

## Categorias

- Por padrão, o app usa as categorias:
  `fatura, comprovante, cobrança, oferta, spam, suporte, relatório, alerta de segurança`.
- Para customizar, exporte a variável de ambiente `CATEGORIES` (separadas por vírgula):
  ```bash
  export CATEGORIES="fatura, cobrança, nota fiscal, reunião, spam"
  ```

## Modelo

- Modelo padrão: `joeddav/xlm-roberta-large-xnli` (multilíngue, bom para PT-BR).
- Para trocar, defina `HF_MODEL`:
  ```bash
  export HF_MODEL="facebook/bart-large-mnli"
  ```

Na primeira execução, o modelo será baixado automaticamente (pode demorar).

## Notas

- PDFs: extração de texto via `pypdf`. PDFs somente imagem não terão texto detectado.
- Segurança: app usa `dev-secret` se `FLASK_SECRET_KEY` não for definido. Troque em produção.
- Execução em produção: use algo como `gunicorn -w 2 -b 0.0.0.0:5000 app:create_app()`.
