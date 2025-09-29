import os
from typing import List, Tuple, Optional

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# NLP imports
from transformers import pipeline, AutoTokenizer
import re
import nltk
from nltk.corpus import stopwords

# Local utils
from utils.extract import extract_text_from_pdf

ALLOWED_EXTENSIONS = {"txt", "pdf"}

# Default categories for finance email classification (can be overridden via env VAR CATEGORIES)
DEFAULT_CATEGORIES = [
    "fatura",
    "comprovante",
    "cobrança",
    "oferta",
    "spam",
    "suporte",
    "relatório",
    "alerta de segurança",
    "demissão",
]

# Deterministic mapping from category -> produtividade
CATEGORY_TO_PRODUCTIVITY = {
    "demissão": "Produtivo",
    "comprovante": "Produtivo",
    "cobrança": "Produtivo",
    "relatório": "Produtivo",
    "oferta": "Improdutivo",
    "suporte": "Produtivo",
    "alerta de segurança": "Improdutivo",
    "spam": "Improdutivo",
    "fatura": "Produtivo",
}


def get_categories() -> List[str]:
    cats = os.getenv("CATEGORIES")
    if not cats:
        return DEFAULT_CATEGORIES
    # allow comma-separated labels
    return [c.strip() for c in cats.split(",") if c.strip()]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


classifier = None


def get_classifier():
    global classifier
    if classifier is None:
        # Multilingual NLI model suitable for Portuguese
        model_name = os.getenv("HF_MODEL", "joeddav/xlm-roberta-large-xnli")
        # Force slow tokenizer to avoid protobuf dependency for fast tokenizers
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            tokenizer=tokenizer,
        )
    return classifier


def classify_text(text: str) -> Tuple[str, List[Tuple[str, float]]]:
    clf = get_classifier()
    labels = get_categories()
    result = clf(text, candidate_labels=labels, hypothesis_template="Este email é sobre {}.")
    # result: {labels: [...], scores: [...]} sorted by score desc
    top_label = result["labels"][0]
    pairs = list(zip(result["labels"], result["scores"]))
    return top_label, pairs


# Ensure NLTK stopwords are available
try:
    _ = stopwords.words("portuguese")
except LookupError:
    try:
        nltk.download("stopwords")
    except Exception:
        pass


def preprocess_text(text: str) -> str:
    """Basic cleaning: lowercase, remove punctuation, Portuguese stopwords."""
    try:
        sw = set(stopwords.words("portuguese"))
    except Exception:
        sw = set()
    text = re.sub(r"[^\w\s]", "", text.lower())
    tokens = [w for w in text.split() if w and w not in sw]
    return " ".join(tokens)


def classify_productivity(text: str) -> Tuple[str, List[Tuple[str, float]]]:
    """Classify into Produtivo vs Improdutivo using zero-shot with PT hypothesis."""
    clf = get_classifier()
    cleaned = preprocess_text(text)
    labels = ["Produtivo", "Improdutivo"]
    result = clf(
        cleaned[:2000],
        candidate_labels=labels,
        hypothesis_template="Este email é {}.",
    )
    top_label = result["labels"][0]
    pairs = list(zip(result["labels"], result["scores"]))
    return top_label, pairs


def generate_response_for_category(text: str, category: str) -> str:
    """Heuristic, template-based reply generator by category, enriched with info
    extracted from the email body (e.g., valores e datas).

    This avoids external LLMs and keeps things deterministic.
    """
    # Extract simple entities
    # Money (e.g., R$ 123,45)
    money_match = re.search(r"(?:R\$\s*)?[\d\.]{1,3}(?:\.[\d]{3})*(?:,[\d]{2})|R\$\s*[\d,\.]+", text, flags=re.IGNORECASE)
    valor = money_match.group(0) if money_match else None
    # Dates in DD/MM/YYYY or DD/MM/YY
    date_match = re.search(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", text)
    data_ref = date_match.group(1) if date_match else None

    # Small helper to assemble optional parts
    def opt(parts: List[str]) -> str:
        return " ".join([p for p in parts if p])

    # Normalize category for comparisons
    cat = (category or "").strip().lower()

    if cat == "fatura":
        return opt([
            "Olá,",
            f"Recebemos sua fatura{f' no valor de {valor}' if valor else ''}{f' com referência em {data_ref}' if data_ref else ''}.",
            "Estamos conferindo os detalhes e enviaremos a confirmação/posicionamento em breve.",
            "Caso precise de algo adicional, fique à vontade para responder este email.",
            "Atenciosamente,",
            "Equipe de Suporte",
        ])
    if cat == "comprovante":
        return opt([
            "Olá,",
            f"Obrigado por enviar o comprovante{f' de {valor}' if valor else ''}{f' referente a {data_ref}' if data_ref else ''}.",
            "Vamos validar o recebimento e atualizaremos o status assim que possível.",
            "Se houver qualquer divergência, retornaremos com orientações.",
            "Atenciosamente,",
            "Equipe Financeira",
        ])
    if cat == "cobrança":
        return opt([
            "Olá,",
            f"Identificamos sua mensagem sobre cobrança{f' de {valor}' if valor else ''}{f' com vencimento/competência em {data_ref}' if data_ref else ''}.",
            "Estamos analisando o caso e retornaremos com os próximos passos.",
            "Se possível, confirme dados do pagamento ou anexe documentos complementares.",
            "Atenciosamente,",
            "Equipe Financeira",
        ])
    if cat == "oferta":
        return opt([
            "Olá,",
            "Agradecemos o contato e a proposta compartilhada.",
            "Encaminhamos internamente para avaliação e retornaremos caso seja de interesse.",
            "Obrigado pela compreensão.",
            "Atenciosamente,",
            "Equipe de Parcerias",
        ])
    if cat == "spam":
        return opt([
            "Olá,",
            "Agradecemos a mensagem. No momento, não temos interesse.",
            "Por gentileza, remova este email de listas de distribuição, se aplicável.",
            "Atenciosamente,",
            "Equipe",
        ])
    if cat == "suporte":
        return opt([
            "Olá,",
            "Obrigado por entrar em contato com o suporte.",
            "Registramos seu chamado e nossa equipe está investigando o ocorrido.",
            "Assim que tivermos uma atualização, retornaremos com instruções.",
            "Atenciosamente,",
            "Suporte Técnico",
        ])
    if cat == "relatório":
        return opt([
            "Olá,",
            f"Recebemos o relatório{f' referente a {data_ref}' if data_ref else ''}.",
            "Vamos revisar as informações e enviaremos um parecer em seguida.",
            "Obrigado pelo envio.",
            "Atenciosamente,",
            "Equipe",
        ])
    if cat == "alerta de segurança":
        return opt([
            "Olá,",
            "Agradecemos o alerta. Levamos segurança muito a sério.",
            "Nossa equipe já está verificando o incidente reportado e tomará as medidas cabíveis.",
            "Se possível, mantenha o canal aberto para trocarmos mais detalhes.",
            "Atenciosamente,",
            "Segurança da Informação",
        ])

    # Demissão / Carta de demissão
    if "demiss" in cat:
        return opt([
            "Olá,",
            f"Acusamos o recebimento do seu pedido de demissão{f' datado de {data_ref}' if data_ref else ''}.",
            "Agradecemos pela colaboração prestada até aqui.",
            "Nossa equipe de RH dará sequência aos procedimentos de desligamento e retornará com as orientações sobre documentos, prazos e eventuais acertos (ex.: aviso prévio, devolução de equipamentos).",
            "Se precisar de alguma declaração ou tiver dúvidas, por favor, responda este email.",
            "Desejamos sucesso em seus próximos passos.",
            "Atenciosamente,",
            "Recursos Humanos",
        ])

    # Fallback genérico
    return opt([
        "Olá,",
        "Obrigado pela sua mensagem.",
        "Estamos analisando o conteúdo e retornaremos em breve com os próximos passos.",
        "Atenciosamente,",
        "Equipe",
    ])


def read_uploaded_text() -> Optional[str]:
    # Priority: direct text > file upload
    raw_text = request.form.get("email-text", "").strip()
    if raw_text:
        return raw_text

    if "email-file" not in request.files:
        return None
    file = request.files["email-file"]
    if file and file.filename and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        ext = filename.rsplit(".", 1)[1].lower()
        if ext == "txt":
            # Read as UTF-8 with fallback
            data = file.read()
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    return data.decode("latin-1")
                except Exception:
                    return data.decode("utf-8", errors="ignore")
        elif ext == "pdf":
            return extract_text_from_pdf(file)
    return None


def create_app():
    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/processar", methods=["POST"])
    def processar():
        text = read_uploaded_text()
        if not text:
            flash("Nenhum conteúdo de email fornecido. Cole o texto ou envie um .txt/.pdf.")
            return redirect(url_for("index"))

        categoria, ranking = classify_text(text)
        resposta = generate_response_for_category(text, categoria)
        # Determinar produtividade a partir da categoria (fallback para modelo se desconhecida)
        cat_key = (categoria or "").strip().lower()
        mapped = CATEGORY_TO_PRODUCTIVITY.get(cat_key)
        if mapped:
            prod_label = mapped
            prod_ranking = [("Produtivo", 1.0), ("Improdutivo", 0.0)] if mapped == "Produtivo" else [("Improdutivo", 1.0), ("Produtivo", 0.0)]
        else:
            prod_label, prod_ranking = classify_productivity(text)
        return render_template(
            "resultados.html",
            texto=text,
            categoria=categoria,
            ranking=ranking,
            resposta=resposta,
            produtividade=prod_label,
            produtividade_ranking=prod_ranking,
        )

    @app.route("/api/processar", methods=["POST"])
    def api_processar():
        # Accept both form and JSON
        text = request.form.get("email-text")
        if not text and "email-file" in request.files:
            text = read_uploaded_text()
        if not text:
            payload = request.get_json(silent=True) or {}
            text = (payload.get("texto") or payload.get("text") or "").strip()
        if not text:
            return {"error": "Nenhum conteúdo fornecido."}, 400

        # Alinhar API com a classificação por categorias e resposta sugerida, e produtividade
        categoria, ranking = classify_text(text)
        resposta = generate_response_for_category(text, categoria)
        cat_key = (categoria or "").strip().lower()
        mapped = CATEGORY_TO_PRODUCTIVITY.get(cat_key)
        if mapped:
            prod_label = mapped
            prod_ranking = [("Produtivo", 1.0), ("Improdutivo", 0.0)] if mapped == "Produtivo" else [("Improdutivo", 1.0), ("Produtivo", 0.0)]
        else:
            prod_label, prod_ranking = classify_productivity(text)
        return {
            "categoria": categoria,
            "resposta": resposta,
            "ranking": [(l, float(s)) for l, s in ranking],
            "produtividade": prod_label,
            "produtividade_ranking": [(l, float(s)) for l, s in prod_ranking],
        }

    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    app = create_app()
    app.run(host=host, port=port, debug=True)
