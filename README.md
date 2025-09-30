# 📧 Classificador de Emails

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![ML](https://img.shields.io/badge/ML-Transformers-orange.svg)](https://huggingface.co/transformers/)

Uma aplicação web que classifica automaticamente o conteúdo de emails (texto ou arquivos PDF/TXT) em categorias personalizáveis usando classificação zero-shot da Hugging Face.

## 🚀 Como rodar

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes do Python)
- Git (opcional, para clonar o repositório)

### Instalação

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

## 📋 Funcionalidades

- Upload de arquivos (PDF/TXT) ou digitação direta do conteúdo
- Interface intuitiva e responsiva
- Suporte a temas claro/escuro
- Classificação em tempo real
- Visualização de resultados detalhados

## 🏷️ Categorias

- Por padrão, o app usa as categorias:
  `fatura, comprovante, cobrança, oferta, spam, suporte, relatório, alerta de segurança`.
- Para customizar, exporte a variável de ambiente `CATEGORIES` (separadas por vírgula):
  ```bash
  export CATEGORIES="fatura, cobrança, nota fiscal, reunião, spam"
  ```

## 🤖 Modelo

- Modelo padrão: `joeddav/xlm-roberta-large-xnli` (multilíngue, bom para PT-BR).
- Para trocar, defina `HF_MODEL`:
  ```bash
  export HF_MODEL="facebook/bart-large-mnli"
  ```

Na primeira execução, o modelo será baixado automaticamente (pode demorar).

## 📝 Notas Técnicas

- **Extração de PDF**: Utiliza `pypdf` para extração de texto. PDFs que são apenas imagem não terão o texto detectado.
- **Segurança**: 
  - Em desenvolvimento, usa `dev-secret` como chave secreta.
  - Para produção, defina a variável de ambiente `FLASK_SECRET_KEY` com uma chave segura.
- **Performance**:
  - A primeira classificação pode demorar enquanto o modelo é carregado.
  - Para produção, considere usar um serviço de cache como Redis.
- **Deploy em Produção**:
  ```bash
  # Usando Gunicorn
  pip install gunicorn
  gunicorn -w 4 -b 0.0.0.0:5000 app:create_app()
  
  # Ou usando Waitress (Windows)
  pip install waitress
  waitress-serve --port=5000 app:create_app()
  ```

## 📄 Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e enviar pull requests.

## 📧 Contato

Seu Nome - [@seu_twitter](https://twitter.com/seu_twitter) - seu.email@exemplo.com

Link do Projeto: [https://github.com/seu-usuario/classificador-emails](https://github.com/seu-usuario/classificador-emails)
