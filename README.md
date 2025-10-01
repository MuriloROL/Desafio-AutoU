# 📧 AutoU - Classificador Inteligente de Emails

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![ML](https://img.shields.io/badge/ML-Transformers-orange.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Este app é uma aplicação web que utiliza inteligência artificial para classificar automaticamente o conteúdo de emails em categorias personalizáveis, utilizando técnicas avançadas de processamento de linguagem natural (NLP) através da biblioteca Transformers da Hugging Face.

## 🎯 Visão Geral

O app foi desenvolvido para ajudar usuários e empresas a organizarem automaticamente seus emails em categorias lógicas, melhorando a produtividade e a gestão de caixa de entrada. A aplicação utiliza um modelo de classificação zero-shot, o que significa que não requer treinamento prévio e pode classificar textos em categorias personalizadas em tempo real.

## ✨ Funcionalidades Principais

- **Classificação Inteligente**: Identifica automaticamente a categoria mais adequada para cada email
- **Análise de Produtividade**: Classifica emails como produtivos ou improdutivos
- **Suporte a Múltiplos Formatos**:
  - Texto direto
  - Arquivos TXT
  - Arquivos PDF (com extração de texto)
- **Interface Moderna e Responsiva**:
  - Design limpo e intuitivo
  - Tema claro/escuro
  - Totalmente responsivo para dispositivos móveis
- **Resultados Detalhados**:
  - Categoria principal
  - Nível de confiança
  - Ranking completo de categorias
  - Resposta sugerida personalizável

## 🚀 Como Executar Localmente

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes do Python)
- Git (opcional, apenas para clonar o repositório)
- Conexão com a internet (para baixar os modelos na primeira execução)

### Instalação Passo a Passo

1. **Clone o repositório** (opcional):
   ```bash
   git clone https://github.com/seu-usuario/autou-classificador.git
   cd autou-classificador
   ```

2. **Crie e ative um ambiente virtual** (recomendado):
   ```bash
   # Linux/MacOS
   python -m venv .venv
   source .venv/bin/activate

   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```
   
   > 💡 **Nota**: A primeira instalação pode demorar alguns minutos devido ao download dos pacotes de machine learning.

4. **Configure as variáveis de ambiente** (opcional):
   ```bash
   # Categorias personalizadas (separadas por vírgula)
   export CATEGORIES="fatura, cobrança, nota fiscal, reunião, spam"
   
   # Porta do servidor (opcional, padrão: 5000)
   export PORT=8080
   ```

5. **Inicie o servidor**:
   ```bash
   python app.py
   ```

6. **Acesse a aplicação**:
   Abra seu navegador e acesse:
   ```
   http://localhost:5000
   ```
   
   > 🔄 Se a porta 5000 estiver em uso, a aplicação tentará usar a porta 8080 automaticamente.

## 🛠️ Como Usar

1. **Página Inicial**:
   - Cole o texto do email na caixa de texto
   - OU faça upload de um arquivo (TXT ou PDF)
   - Clique em "Classificar"

2. **Página de Resultados**:
   - Visualize a categoria principal
   - Confira o nível de confiança da classificação
   - Veja o ranking completo de categorias
   - Copie a resposta sugerida com um clique
   - Alternar entre temas claro/escuro

3. **Personalização**:
   - As categorias podem ser personalizadas através da variável de ambiente `CATEGORIES`
   - O modelo de classificação pode ser alterado no código-fonte (consulte a documentação do Hugging Face)

## 📊 Categorias Padrão

O app vem pré-configurado com as seguintes categorias:

- `Fatura`: Faturas e contas a pagar
- `Comprovante`: Comprovantes de pagamento
- `Cobrança`: Mensagens de cobrança
- `Oferta`: Promoções e ofertas
- `Spam`: Mensagens não solicitadas
- `Suporte`: Solicitações de suporte técnico
- `Relatório`: Relatórios e atualizações
- `Alerta de Segurança`: Alertas de segurança e acesso
- `Demissão`: Questões de demissão


## 🔧 Personalização

### Alterando as Categorias

Para usar suas próprias categorias, defina a variável de ambiente `CATEGORIES` antes de iniciar o aplicativo:

```bash
export CATEGORIES="fatura, cobrança, nota fiscal, reunião, spam, orçamento, contrato"
python app.py
```

### Usando um Modelo Diferente

Por padrão, o AutoU usa o modelo `facebook/bart-large-mnli`. Para usar um modelo diferente:

1. Altere a linha no arquivo `app.py`:
   ```python
   classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
   ```
   
2. Consulte a [documentação do Hugging Face](https://huggingface.co/models?pipeline_tag=zero-shot-classification) para outros modelos disponíveis.

## 📦 Estrutura do Projeto

```
.
├── app.py                 # Aplicação principal (Flask)
├── requirements.txt       # Dependências do projeto
├── static/               # Arquivos estáticos (CSS, imagens)
│   ├── logo-autou.png
│   └── style.css
├── templates/            # Templates HTML
│   ├── index.html        # Página inicial
│   └── resultados.html   # Página de resultados
└── utils/
    └── extract.py        # Utilitários para extração de texto
```




## 📞 Contato

Murilo Rodrigues Lima- [https://www.linkedin.com/in/devmurilolima/](https://www.linkedin.com/in/devmurilolima/) - murilo.rl.dev@gmail.com

Link do Projeto: [https://github.com/MuriloROL/Desafio-AutoU](https://github.com/MuriloROL/Desafio-AutoU)


## 🤖 Modelo

- Modelo padrão: `joeddav/xlm-roberta-large-xnli` (multilíngue, bom para PT-BR).

Na primeira execução, o modelo será baixado automaticamente (pode demorar).

## 📝 Notas Técnicas

- **Extração de PDF**: Utiliza `pypdf` para extração de texto. PDFs que são apenas imagem não terão o texto detectado.
- **Segurança**: 
  - Em desenvolvimento, usa `dev-secret` como chave secreta.
- **Performance**:
  - A primeira classificação pode demorar enquanto o modelo é carregado.

