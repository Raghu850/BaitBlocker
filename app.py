from flask import Flask, render_template, redirect, url_for,jsonify, session, request, flash, make_response,get_flashed_messages
from flask_dance.contrib.google import make_google_blueprint, google
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import easyocr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import joblib
import ipaddress
import numpy as np
from urllib.parse import urlparse,urlunparse
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import timedelta
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


from functools import wraps
import os,time
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any
from markupsafe import escape
from PIL import Image
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

# Allow OAuth over HTTP (dev only)
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# Replace with your real SendGrid API key
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
VERIFIED_SENDER = os.getenv("VERIFIED_SENDER")

# Serializer for generating tokens
s = URLSafeTimedSerializer(app.secret_key)
 

# Database Config
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# Upload folder for SMS images
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), "static/uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# --- Cache Prevention Decorator ---
def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return no_cache

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=0)

# --- Model ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True, nullable=False)  # <-- required
    password = db.Column(db.String(200))
    phone = db.Column(db.String(20), unique=True, nullable=True)    # <-- optional
    profile_icon = db.Column(db.String(255), default='default_icon.png')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    type = db.Column(db.String(50))          # keep this one
    input_data = db.Column(db.Text)          # keep this one
    result = db.Column(db.Text)
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('detections', lazy=True))


    # Relationship to User
    user = db.relationship('User', backref=db.backref('detections', lazy=True))


with app.app_context():
    db.create_all()

# --- Google OAuth ---
google_bp = make_google_blueprint(
    client_id=os.getenv("GOOGLE_OAUTH_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_OAUTH_CLIENT_SECRET"),
    scope=[
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile"
    ],
    redirect_url="/google_login/callback"
)
app.register_blueprint(google_bp, url_prefix="/google_login")

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Load ML model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "phishing_model.pkl")
ml_model = joblib.load(os.path.abspath(MODEL_PATH))

# Gemini model
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "models/gemini-1.5-pro-latest")
gemini_model = genai.GenerativeModel(GEMINI_MODEL_ID)

# Known brands & TLDs
KNOWN_BRANDS = {
    "instagram": "instagram.com",
    "facebook": "facebook.com",
    "meta": "meta.com",
    "google": "google.com",
    "gmail": "mail.google.com",
    "apple": "apple.com",
    "microsoft": "microsoft.com",
    "office": "office.com",
    "outlook": "outlook.com",
    "paypal": "paypal.com",
    "netflix": "netflix.com",
    "amazon": "amazon.com",
    "whatsapp": "whatsapp.com",
    "linkedin": "linkedin.com",
    "x": "x.com",
    "twitter": "twitter.com",
    "github": "github.com",
}

SUSPICIOUS_TLDS = {"zip", "country", "kim", "loan", "mom", "men", "work", "click", "xyz", "link", "gq", "cf", "tk"}
URL_SHORTENERS = {"bit.ly", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "is.gd", "buff.ly", "adf.ly", "rebrand.ly", "cutt.ly"}

# -----------------------------
# Utility functions
# -----------------------------
def is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False

def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    dp = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            dp[j], prev = min(dp[j]+1, dp[j-1]+1, prev+cost), dp[j]
    return dp[-1]

def domain_parts(host: str):
    parts = host.split(".")
    if len(parts) < 2:
        return host, "", ""
    sld, tld = parts[-2], parts[-1]
    sub = ".".join(parts[:-2]) if len(parts) > 2 else ""
    return sub, sld, tld

def local_signal_checks(url: str):
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    scheme = parsed.scheme.lower()
    path = parsed.path or "/"
    query = parsed.query or ""
    sub, sld, tld = domain_parts(host)
    reasons = []

    signals = {
        "scheme": scheme,
        "host": host,
        "path": path,
        "query": query,
        "is_https": scheme == "https",
        "has_at_symbol": "@" in url,
        "has_ip_host": is_ip(host),
        "subdomain_count": len(sub.split(".")) if sub else 0,
        "has_many_subdomains": False,
        "has_hyphen_in_domain": "-" in sld,
        "has_digits_in_domain": bool(re.search(r"\d", sld)),
        "tld": tld,
        "suspicious_tld": tld in SUSPICIOUS_TLDS,
        "is_shortener": host in URL_SHORTENERS,
        "length": len(url),
        "brand_lookalike": None,
        "brand_distance": None,
        "brand_target": None,
    }
    signals["has_many_subdomains"] = signals["subdomain_count"] >= 3

    # Brand lookalike detection
    best_brand, best_dist = None, None
    for brand, legit in KNOWN_BRANDS.items():
        legit_sld = domain_parts(legit.lower())[1]
        dist = levenshtein(sld, legit_sld)
        if best_dist is None or dist < best_dist:
            best_brand, best_dist = (brand, legit.lower()), dist
    if best_brand:
        signals["brand_target"] = best_brand[0]
        signals["brand_lookalike"] = best_dist <= 3 and sld != domain_parts(best_brand[1])[1]
        signals["brand_distance"] = best_dist

    # Human-readable reasons
    if not signals["is_https"]:
        reasons.append("The URL uses HTTP instead of HTTPS.")
    if signals["has_at_symbol"]:
        reasons.append("The URL contains an '@' symbol, which can hide the real destination.")
    if signals["has_ip_host"]:
        reasons.append("The host is an IP address instead of a domain name.")
    if signals["has_many_subdomains"]:
        reasons.append("The URL has many subdomains.")
    if signals["has_hyphen_in_domain"]:
        reasons.append("The domain contains hyphens, often used in deceptive domains.")
    if signals["has_digits_in_domain"]:
        reasons.append("The domain contains digits, which may indicate impersonation.")
    if signals["suspicious_tld"]:
        reasons.append(f"The top-level domain '.{tld}' is commonly abused.")
    if signals["is_shortener"]:
        reasons.append("The domain is a known URL shortener.")
    if signals["brand_lookalike"]:
        reasons.append(f"Possible typosquatting: '{sld}' looks like '{KNOWN_BRANDS[signals['brand_target']]}' (edit distance {signals['brand_distance']}).")

    # Risk score
    weight = (
        2 * (not signals["is_https"]) +
        2 * signals["brand_lookalike"] +
        1 * signals["has_ip_host"] +
        1 * signals["has_many_subdomains"] +
        1 * signals["suspicious_tld"] +
        1 * signals["is_shortener"] +
        1 * signals["has_hyphen_in_domain"] +
        1 * signals["has_digits_in_domain"]
    )
    local_risk = min(1.0, weight / 7.0)

    line_by_line = {
        "scheme": scheme,
        "domain": host,
        "subdomains": sub or "(none)",
        "sld": sld or "(n/a)",
        "tld": tld or "(n/a)",
        "path": path,
        "query": query or "(none)"
    }

    return signals, reasons, local_risk, line_by_line

# -----------------------------
# Gemini classifier
# -----------------------------
def gemini_classifier(url, signals, reasons, line_by_line):
    """
    Enhanced Gemini AI classifier for phishing detection with better suggested domains.
    """
    import json, re

    try:
        # Prompt Gemini to also suggest the correct legitimate domain if it's phishing
        prompt = (
            "You are a cybersecurity expert. Analyze the URL for phishing, typosquatting, HTTPS issues.\n"
            "Return ONLY JSON with keys:\n"
            "- verdict: 'phishing', 'suspicious', or 'safe'\n"
            "- confidence: 0-1\n"
            "- reasons: human-friendly explanations\n"
            "- user_message: concise message\n"
            "- line_by_line_explanation\n"
            "- original_legit_domain: the legitimate domain this URL is mimicking (if any)\n\n"
            f"URL: {url}\nSignals: {json.dumps(signals)}"
        )

        response = gemini_model.generate_content(prompt)
        raw = re.sub(r"^```json|```", "", response.text.strip())
        data = json.loads(raw)

        verdict = data.get("verdict", "unknown").lower()
        confidence = float(data.get("confidence", 0))
        confidence = max(0.0, min(1.0, confidence))

        # Smart fallback: if AI returns no original domain, attempt brand mapping
        original_domain = data.get("original_legit_domain", "")
        if not original_domain and signals.get("brand_target"):
            # Map brand_target intelligently (e.g., "mystore" → "microsoft.com")
            brand_target = signals["brand_target"].lower()
            brand_map = {
                "mystore": "microsoft.com",
                "insta": "instagram.com",
                "paypal": "paypal.com",
                "fb": "facebook.com",
            }
            original_domain = brand_map.get(brand_target, "")

        # Ensure clickable https://
        if original_domain and not original_domain.startswith(("http://", "https://")):
            original_domain = "https://" + original_domain

        return {
            "label": verdict,
            "confidence": confidence,
            "reasons": data.get("reasons", reasons or ["Local heuristics applied."]),
            "user_message": data.get("user_message", "⚠️ Automated analysis result from AI."),
            "line_by_line_explanation": data.get("line_by_line_explanation", line_by_line),
            "original_legit_domain": original_domain,
            "raw_json": data
        }

    except Exception:
        # Fallback
        fallback_label = "phishing" if signals.get("brand_lookalike") or not signals.get("is_https") else "safe"
        fallback_confidence = 0.8 if fallback_label == "phishing" else 0.7
        fallback_reasons = []
        if not signals.get("is_https"):
            fallback_reasons.append("🔒 The URL uses HTTP instead of HTTPS.")
        if signals.get("brand_lookalike"):
            fallback_reasons.append(f"👀 The URL looks like {signals.get('brand_target')}")
        return {
            "label": fallback_label,
            "confidence": fallback_confidence,
            "reasons": fallback_reasons,
            "user_message": "❌ High-risk indicators detected. Likely phishing.",
            "line_by_line_explanation": line_by_line,
            "original_legit_domain": KNOWN_BRANDS.get(signals.get("brand_target"), ""),
            "raw_json": {}
        }




# -----------------------------
# ML classifier
# -----------------------------
def ml_classifier(features_array):
    try:
        pred = ml_model.predict(features_array)[0]
        label = "phishing" if pred == -1 else "safe"
        return {"label": label, "raw_pred": int(pred)}
    except Exception as e:
        return {"label": "error", "error": str(e)}

# -----------------------------
# Predict URL
# -----------------------------

def transform_url(url):
    """Minimal placeholder feature extractor"""
    return [len(url), url.count("-"), url.count("@")]

def predict_url(url: str) -> dict:
    """
    Predicts if a URL is phishing, suspicious, or safe.
    Combines:
    - Local heuristic checks
    - ML classifier
    - Gemini AI classifier
    Returns a structured dictionary for frontend display.
    """
    from urllib.parse import urlparse, urlunparse

    # --- Normalize URL ---
    def normalize_url(u: str) -> str:
        """Ensure URL has a protocol and clean trailing slashes."""
        if not u.startswith(("http://", "https://")):
            u = "https://" + u
        parsed = urlparse(u)
        return urlunparse(parsed._replace(path=parsed.path.rstrip("/")))

    url_norm = normalize_url(url)

    # --- Local heuristic checks ---
    signals, local_reasons, local_risk, line_by_line = local_signal_checks(url_norm)

    # --- ML prediction ---
    features_array = np.array(transform_url(url_norm)).reshape(1, -1)
    ml_result = ml_classifier(features_array)
    ml_label = ml_result.get("label", "error")

    # --- Gemini AI analysis ---
    gemini_result = gemini_classifier(url_norm, signals, local_reasons, line_by_line)
    gemini_label = gemini_result.get("label", "unknown")
    gemini_conf = float(gemini_result.get("confidence", 0.0))

    # --- Suggested Legit Domain ---
    original_domain = gemini_result.get("original_legit_domain", "")
    if not original_domain and signals.get("brand_target"):
        original_domain = KNOWN_BRANDS.get(signals["brand_target"], "")
    if original_domain and not original_domain.startswith(("http://", "https://")):
        original_domain = "https://" + original_domain

    # --- Final Verdict Decision ---
    if gemini_label in {"phishing", "safe"} and gemini_conf >= 0.7:
        final_verdict = gemini_label
        reasons = gemini_result.get("reasons", local_reasons)
        user_message = gemini_result.get("user_message", "⚠️ Automated analysis result from AI.")
    elif ml_label == "phishing" or signals.get("brand_lookalike") or local_risk >= 0.5:
        final_verdict = "phishing"
        reasons = local_reasons
        user_message = "❌ High-risk indicators detected. Likely phishing."
    elif 0.3 <= local_risk < 0.7:
        final_verdict = "suspicious"
        reasons = local_reasons
        user_message = "⚠️ Suspicious signs detected. Check carefully."
    else:
        final_verdict = "safe"
        reasons = local_reasons
        user_message = "✅ URL appears safe."

    # --- User-Friendly Reasons ---
    friendly_reasons = []
    for reason in reasons:
        r = reason.lower()
        if "looks like" in r:
            parts = reason.split("looks like")
            friendly_reasons.append(
                f"👀 The URL <code>{parts[0].strip()}</code> looks similar to <code>{parts[1].strip()}</code>. Attackers may try to trick you!"
            )
        elif "http" in r and "https" not in r:
            friendly_reasons.append("🔒 The site is not using HTTPS. Connection may not be secure.")
        elif "@" in r:
            friendly_reasons.append("⚠️ The URL contains '@', which may hide the real destination.")
        else:
            friendly_reasons.append(f"⚠️ {reason}")

    # --- Advice ---
    advice = []
    if final_verdict == "phishing":
        advice.append("❌ Avoid entering any personal or sensitive information on this site.")
    elif final_verdict == "suspicious":
        advice.append("⚠️ URL shows suspicious signs. Double-check before proceeding.")
    else:
        advice.append("✅ URL appears safe, but always verify before logging in.")

    if original_domain:
        advice.append(f"💡 Suggested Legit Domain: {original_domain}")

    # --- Final Structured Response ---
    return {
        "url": url_norm,
        "final_verdict": final_verdict.capitalize(),
        "friendly_reasons": friendly_reasons,
        "user_message": user_message,
        "advice": advice,
        "original_legit_domain": original_domain,
        "gemini_verdict": gemini_label.capitalize(),
        "confidence": gemini_conf,
        "reasons": reasons,
        "line_by_line": line_by_line,
    }
# -----------------------------
# Helpers for Mail Phishing
# -----------------------------
def clean_text(text: str) -> str:
    """Lowercase, remove symbols, filter stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    filtered = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(filtered)


def gemini_email_analysis(text: str) -> dict:
    """
    Call Gemini API to classify email text.
    Returns dictionary with keys: verdict, confidence, reasons, line_by_line_explanation, user_message
    """
    try:
        prompt = f"""
        You are a cybersecurity analyst. Analyze the following email text and classify it as 'phishing' or 'safe'.
        Provide output ONLY as JSON with these keys:
        {{
          "verdict": "phishing" | "safe",
          "confidence": 0.0-1.0,
          "reasons": ["reason1", "reason2"],
          "line_by_line_explanation": {{"greeting": "...", "links": "...", "language": "...", "urgency": "...", "sender": "..."}},
          "user_message": "final user-friendly summary"
        }}

        Email Text:
        {text}
        """
        response = gemini_model.generate_content(prompt)
        cleaned = response.text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").replace("json", "").strip()
        data = json.loads(cleaned) if cleaned.startswith("{") else {}
        data.setdefault("verdict", "unknown")
        data.setdefault("confidence", 0.0)
        data.setdefault("reasons", [])
        data.setdefault("line_by_line_explanation", {})
        data.setdefault("user_message", "")
        return data
    except Exception as e:
        return {
            "verdict": "unknown",
            "confidence": 0.0,
            "reasons": [f"Gemini analysis failed: {str(e)}"],
            "line_by_line_explanation": {},
            "user_message": "We could not analyze this email with Gemini."
        }



# -----------------------------
# Gemini SMS Data Structure
# -----------------------------
@dataclass
class GeminiSMSResult:
    verdict: str
    confidence: float
    reasons: List[str]
    risky_phrases: List[str]
    urls: List[Dict[str, Any]]
    user_message: str

OFFICIAL_DOMAINS = [
    "airtel.in", "icicibank.com", "hdfcbank.com", "sbi.co.in",
    "axisbank.com", "paytm.com", "phonepe.com", "google.com", "amazon.in"
]

def highlight_phrases(text: str, phrases: List[str]) -> str:
    escaped = escape(text or "")
    for p in sorted(set(phrases), key=len, reverse=True):
        pattern = re.compile(rf"(?i)\b({re.escape(p)})\b")
        escaped = pattern.sub(r"<mark>\\1</mark>", escaped)
    return escaped

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-1.5-pro-latest")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL_ID)

def call_gemini_sms_image(image_path: str) -> GeminiSMSResult:
    prompt = """
You are a cybersecurity SMS phishing detector.
Analyze the text inside this SMS screenshot image.

Guidelines:
- safe → if legitimate, no suspicious links, no sensitive info requests
- suspicious → unusual wording/links, but not clearly phishing
- phishing → malicious intent, fake links, urgent CTA, sensitive info

Respond ONLY in valid JSON with this format:
{
  "verdict": "safe" | "suspicious" | "phishing",
  "confidence": 0.0-1.0,
  "reasons": ["short reason 1", "short reason 2"],
  "risky_phrases": ["word1","word2"],
  "urls": [{"url": "https://...", "status": "safe|suspicious|phishing"}],
  "user_message": "1-2 sentence explanation"
}
"""
    try:
        img = Image.open(image_path)
        resp = gemini_model.generate_content(
            [prompt, img],
            generation_config={"response_mime_type": "application/json"}
        )
        data = json.loads(resp.text)
    except Exception as e:
        data = {
            "verdict": "unknown",
            "confidence": 0.0,
            "reasons": [f"Gemini error: {str(e)}"],
            "risky_phrases": [],
            "urls": [],
            "user_message": "Automatic analysis failed."
        }

    verdict = str(data.get("verdict", "unknown")).lower()
    if verdict not in {"safe", "suspicious", "phishing"}:
        verdict = "unknown"

    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    reasons = [str(r) for r in data.get("reasons", [])][:6]
    risky_phrases = list(set(data.get("risky_phrases", [])))
    urls_list = [{"url": str(item.get("url","")).strip(),
                  "status": item.get("status","suspicious")} for item in data.get("urls", [])]
    user_message = data.get("user_message", "No explanation provided.")

    return GeminiSMSResult(verdict, confidence, reasons, risky_phrases, urls_list, user_message)

# --- Routes ---
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
@nocache
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            flash("Please enter both email and password", "warning")
            return redirect(url_for("login"))

        # Fetch user from DB
        user = User.query.filter_by(email=email).first()

        # Check if user exists and password is set
        if user is None or not user.password:
            flash("Invalid email or password", "danger")
            return redirect(url_for("login"))

        # Check password
        if check_password_hash(user.password, password):
            session["user"] = {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "profile_icon": user.profile_icon
                }

            flash("Logged in successfully!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
@nocache
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if User.query.filter_by(email=email).first():
            flash("Email already registered!", "danger")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return render_template("register.html")

        hashed_password = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_password)

        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/dashboard")
@nocache
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    user = User.query.get(session["user"]["id"])
    if not user:
        flash("User not found, please login again.", "danger")
        session.clear()
        return redirect(url_for("login"))

    # Get user detection history
    history = Detection.query.filter_by(user_id=user.id).order_by(Detection.timestamp.desc()).all()

    # Build stats dictionary
    stats = {
        "total_scans": len(history),
        "phishing": sum(1 for h in history if h.result.lower() == "phishing"),
        "safe": sum(1 for h in history if h.result.lower() == "safe"),
        "suspicious": sum(1 for h in history if h.result.lower() == "suspicious"),
    }

    # --- Custom phishing verdict output for dashboard ---
    # If a URL is submitted, show detailed phishing analysis
    url_input = request.args.get("url")
    phishing_output = None
    if url_input:
        result = predict_url(url_input)
        url_display = result.get("url", url_input)
        verdict = result.get("final_verdict", "Unknown")
        user_message = result.get("user_message", "")
        reasons = result.get("friendly_reasons", result.get("reasons", []))
        advice = result.get("advice", [])
        original_legit_domain = result.get("original_legit_domain", "")
        # Compose output similar to your example
        phishing_output = f"""
The URL <b>{url_display}</b> is highly suspicious and likely a phishing attempt. Phishing websites often mimic legitimate domains to deceive users into entering sensitive information.<br><br>
<b>Verdict:</b> {"❌ Phishing" if verdict.lower() == "phishing" else "✅ Safe" if verdict.lower() == "safe" else "⚠️ Suspicious"}<br>
<b>Message:</b> {user_message}<br><br>
<b>Why this URL may be unsafe:</b>
<ul>
""" + "".join(f"<li>{r}</li>" for r in reasons) + "</ul>"

        if original_legit_domain:
            phishing_output += f"""
<b>Suggested Correct URL:</b> If you intended to visit this site, the correct URL is:<br>
Correct URL: <b>https://{original_legit_domain}/login</b><br>
"""

        phishing_output += "<br><b>Advice:</b><ul>"
        for a in advice:
            phishing_output += f"<li>{a}</li>"
        phishing_output += "</ul>"

        phishing_output += """
<br><b>What to Do If You Clicked on a Phishing Link:</b>
<ul>
<li>Disconnect from the Internet: Immediately disconnect your device to prevent further data transmission.</li>
<li>Run a Security Scan: Use reputable antivirus software to scan your device for malware.</li>
<li>Change Passwords: If you entered any credentials, change your passwords on the affected accounts.</li>
<li>Monitor Accounts: Keep an eye on your financial and personal accounts for any unauthorized activity.</li>
<li>Report the Incident: Report the phishing attempt to organizations like the Better Business Bureau or the Federal Trade Commission.</li>
</ul>
<b>Additional Resources:</b>
<ul>
<li><a href="https://us.norton.com/blog/emerging-threats/what-to-do-if-you-click-on-a-phishing-link" target="_blank">Norton’s Guide on Phishing Links</a></li>
<li><a href="https://www.keepersecurity.com/blog/2022/10/13/what-to-do-if-you-click-on-a-phishing-link/" target="_blank">Keeper Security’s Advice</a></li>
</ul>
"""

    return render_template(
        "dashboard.html",
        user=user,
        stats=stats,
        user_history=history,
        phishing_output=phishing_output
    )

@app.route("/logout")
@nocache
def logout():
    session.clear()
    flash("Logged out successfully", "info")
    return redirect(url_for("login"))

# --- Google OAuth Callback ---
@app.route("/google_login/callback")
@nocache
def google_login_callback():
    if not google.authorized:
        return redirect(url_for("google.login"))

    try:
        resp = google.get("/oauth2/v2/userinfo")
        if not resp.ok:
            flash("Failed to fetch user info from Google", "danger")
            return redirect(url_for("login"))

        user_info = resp.json()   # <-- this was missing
        email = user_info.get("email")
        name = user_info.get("name")
        picture = user_info.get("picture")

        if not email:
            flash("Google login failed: Email not found", "danger")
            return redirect(url_for("login"))

        # check if user exists
        user = User.query.filter_by(email=email).first()
        if not user:
            # new user
            user = User(
                name=name,
                email=email,
                profile_icon=picture,
                phone=None
            )
            db.session.add(user)
            db.session.commit()
        else:
            # update existing
            user.name = name
            user.profile_icon = picture
            db.session.commit()

        # save session
        session["user"] = {
    "id": user.id,
    "email": user.email,
    "name": user.name,
    "phone": user.phone,
    "profile_icon": user.profile_icon or "default_icon.png"
}

        flash("Logged in with Google successfully!", "success")
        return redirect(url_for("dashboard"))

    except Exception as e:
        return f"Google login failed: {str(e)}", 500





# --- Firebase Phone Login Page ---
@app.route("/firebase_login")
@nocache
def firebase_login():
    return render_template(
        "firebase_login.html",
        firebase_api_key=os.getenv("FIREBASE_API_KEY"),
        firebase_auth_domain=os.getenv("FIREBASE_AUTH_DOMAIN"),
        firebase_project_id=os.getenv("FIREBASE_PROJECT_ID"),
    )

# --- Firebase Verification Callback ---
@app.route("/firebase_login/verify", methods=["POST"])
@nocache
def firebase_verify():
    phone = request.form.get("phone")
    if not phone:
        flash("Invalid phone verification.", "danger")
        return redirect(url_for("firebase_login"))

    user = User.query.filter_by(phone=phone).first()

    if user:
        # Existing phone → login
        session["user"] = {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "phone": user.phone,
            "profile_icon": user.profile_icon or "default_icon.png"
        }
        return redirect(url_for("dashboard"))
    else:
        # New phone → force set email
        session["pending_phone"] = phone
        return redirect(url_for("set_email"))

@app.route("/set_email", methods=["GET", "POST"])
def set_email():
    phone = session.get("pending_phone")
    if not phone:
        return redirect(url_for("login"))

    if request.method == "POST":
        email = request.form.get("email").strip()
        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            existing_user.phone = phone
            db.session.commit()
            session.pop("pending_phone", None)
            session["user"] = {"id": existing_user.id, "email": existing_user.email, "phone": existing_user.phone,"profile_icon": existing_user.profile_icon}
            return redirect(url_for("dashboard"))
        else:
            new_user = User(name=email.split("@")[0], email=email, phone=phone)
            db.session.add(new_user)
            db.session.commit()
            session.pop("pending_phone", None)
            session["user"] = {"id": new_user.id, "name":new_user.name,"email": new_user.email, "phone": new_user.phone}
            return redirect(url_for("dashboard"))

    return render_template("set_email.html", phone=phone)

@app.route("/edit_profile", methods=["POST"])
def edit_profile():
    if "user" not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401

    user = User.query.get(session["user"]["id"])
    if not user:
        return jsonify({'success': False, 'error': 'User not found'}), 404

    # Update name/email if provided
    if "name" in request.form:
        user.name = request.form["name"]
        session["user"]["name"] = user.name  

    if "email" in request.form:
        user.email = request.form["email"]
        session["user"]["email"] = user.email  

    if "phone" in request.form:
        user.phone = request.form["phone"]
        session["user"]["phone"] = user.phone  

    # Handle profile icon upload
    if "profile_icon" in request.files:
        file = request.files["profile_icon"]
        if file.filename != "":
            filename = f"user_{user.id}_{int(datetime.utcnow().timestamp())}{os.path.splitext(file.filename)[1]}"
            filepath = os.path.join("static/profile_icons", filename)
            file.save(filepath)

            user.profile_icon = f"profile_icons/{filename}"
            session["user"]["profile_icon"] = user.profile_icon
            session["user"]["profile_icon_timestamp"] = int(datetime.utcnow().timestamp())

    db.session.commit()

    # ✅ Return updated info for frontend
    return jsonify({
        'success': True,
        'message': 'Profile updated successfully',
        'name': user.name,
        'phone': user.phone,
        'email': user.email,
        'profile_icon_url': url_for('static', filename=user.profile_icon or "profile_icons/default_icon.png")
    })

@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect(url_for("login"))

    u = User.query.get(session['user']['id'])
    if not u:
        flash("User not found. Please login again.", "danger")
        session.clear()
        return redirect(url_for("login"))

    # Ensure profile icon exists
    if not u.profile_icon:
        u.profile_icon = "profile_icons/default_icon.png"
    
    # Ensure created_at exists and is a datetime object
    if not u.created_at:
        u.created_at = datetime.utcnow()

    # Commit changes if any were made
    db.session.commit()

    return render_template("profile.html", user=u)


@app.template_filter('format_datetime')
def format_datetime(value, format='%B %d, %Y'):
    if not value:
        return "Unknown"
    # Convert string to datetime if needed
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    return value.strftime(format)


@app.route("/delete_account", methods=["POST"])
def delete_account():
    if "user" not in session:
        return jsonify({'success': False, 'error': 'Login required.'})

    user = User.query.get(session['user']['id'])
    if not user:
        return jsonify({'success': False, 'error': 'User not found.'})

    # Delete local profile icon if not default and not a remote URL
    if user.profile_icon and not user.profile_icon.startswith("http"):
        rel_path = user.profile_icon
        if not rel_path.startswith("profile_icons/"):
            rel_path = f"profile_icons/{rel_path}"

        if rel_path != "profile_icons/default_icon.png":
            icon_path = os.path.join(app.root_path, 'static', rel_path)
            if os.path.exists(icon_path):
                os.remove(icon_path)

    db.session.delete(user)
    db.session.commit()
    session.clear()
    
    flash("Your account has been deleted successfully!", "success")
    return jsonify({'success': True, 'redirect': url_for('login')})

@app.route('/phishing-detector', methods=['GET', 'POST'])
def phishing_detector():
    result = None
    if request.method == 'POST':
        url = request.form.get('url')
        if "login" in url or "verify" in url:
            result = "Phishing"
        else:
            result = "Safe"
    return render_template("phishing_detector.html", result=result)

# ✅ Context processor ensures "user" is always available in templates
@app.context_processor
def inject_user():
    if "user" in session:
        user = User.query.get(session["user"]["id"])
        return {"current_user": user}
    return {"current_user": None}

# =========================
# SMS Phishing Detector
# =========================
@app.route("/sms-phishing", methods=["GET", "POST"])
@nocache
def sms_phishing_detector():
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

            base = secure_filename(file.filename)
            uniq = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            filename = f"{uniq}_{base}"
            uploaded_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(uploaded_path)

            image_url = url_for("static", filename=f"uploads/{filename}", _=uniq)

            # Run analysis
            gem = call_gemini_sms_image(uploaded_path)

            badge_map = {
                "safe": "✅ Safe",
                "suspicious": "⚠ Suspicious",
                "phishing": "❌ Phishing",
            }
            badge = badge_map.get(gem.verdict, "❔ Unknown")

            highlighted_text = highlight_phrases(
                "(OCR skipped – Gemini vision used)", gem.risky_phrases
            )

            analysis = {
                "badge": badge,
                "verdict": gem.verdict,
                "confidence": f"{gem.confidence:.2f}",
                "reasons": gem.reasons,
                "risky_phrases": gem.risky_phrases,
                "urls": gem.urls,
                "user_message": gem.user_message,
                "recommendations": [
                    "Don’t click links in suspicious messages.",
                    "Never share passwords or OTP over SMS.",
                    "Contact the organization via official channels.",
                    "Report phishing SMS to your carrier or cybercrime authority.",
                    "Delete the SMS if phishing is confirmed.",
                ],
            }

            # Save detection if logged in
            if session.get("user"):
                user_obj = None
                if session["user"].get("email"):
                    user_obj = User.query.filter_by(email=session["user"]["email"]).first()
                elif session["user"].get("phone"):
                    user_obj = User.query.filter_by(phone=session["user"]["phone"]).first()

                if user_obj:
                    new_detection = Detection(
                        user_id=user_obj.id,
                        type="sms",
                        input_data=f"Image: {filename}",
                        result=gem.verdict,
                        confidence=float(gem.confidence)
                    )
                    db.session.add(new_detection)
                    db.session.commit()

            session["sms_once"] = {
                "image_path": image_url,
                "highlighted_text": highlighted_text,
                "analysis": analysis,
            }

        return redirect(url_for("sms_phishing_detector"))

    payload = session.pop("sms_once", None)
    return render_template("sms_phishing.html", **payload) if payload else render_template("sms_phishing.html")


# =========================
# Mail Phishing Detector
# =========================
@app.route("/mail-phishing", methods=["GET", "POST"])
@nocache
def mail_phishing_detector():
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

            base = secure_filename(file.filename)
            uniq = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            filename = f"{uniq}_{base}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            image_url = url_for("static", filename=f"uploads/{filename}", _=uniq)

            text_lines = reader.readtext(filepath, detail=0)
            extracted_text = "\n".join(text_lines)

            if extracted_text.strip():
                cleaned = clean_text(extracted_text)
                from joblib import load
                model = load("model.pkl")
                vectorizer = load("vectorizer.pkl")
                vec = vectorizer.transform([cleaned])
                pred = model.predict(vec)[0]
                ml_result = "phishing" if pred == 1 else "safe"

                gemini_result = gemini_email_analysis(extracted_text)
                badge = "🚨 Fraudulent Email" if gemini_result.get("verdict", ml_result) == "phishing" else "✅ Safe Email"

                analysis = {
                    "badge": badge,
                    "verdict": gemini_result.get("verdict", ml_result),
                    "confidence": f"{gemini_result.get('confidence',0.0):.2f}",
                    "reasons": gemini_result.get("reasons", []),
                    "line_by_line_explanation": gemini_result.get("line_by_line_explanation", {}),
                    "user_message": gemini_result.get("user_message", "No summary available."),
                    "risky_phrases": re.findall(
                        r'\b(?:urgent|verify|password|login|click|update|account|bank|verify)\b',
                        extracted_text, flags=re.IGNORECASE
                    ),
                    "urls": [{"url": u, "status": "detected"} for u in re.findall(r'https?://\S+', extracted_text)],
                }
            else:
                extracted_text = ""
                analysis = {"user_message": "❗ Could not extract text from image."}

            if "user" in session:
                user_obj = None
                if session["user"].get("email"):
                    user_obj = User.query.filter_by(email=session["user"]["email"]).first()
                elif session["user"].get("phone"):
                    user_obj = User.query.filter_by(phone=session["user"]["phone"]).first()

                if user_obj:
                    new_detection = Detection(
                        user_id=user_obj.id,
                        type="email",
                        input_data=extracted_text[:500],
                        result=analysis["verdict"],
                        confidence=float(gemini_result.get("confidence", 0.0))
                    )
                    db.session.add(new_detection)
                    db.session.commit()

            session["mail_once"] = {
                "image_path": image_url,
                "email_text": extracted_text,
                "highlighted_text": extracted_text,
                "analysis": analysis,
            }

        return redirect(url_for("mail_phishing_detector"))

    payload = session.pop("mail_once", None)
    return render_template("mail_phishing.html", **payload) if payload else render_template("mail_phishing.html")


# =========================
# URL Phishing Detector
# =========================
@app.route("/url_phishing_detector", methods=["GET", "POST"])
def url_phishing_detector():
    prediction_result = None
    entered_url = None

    if request.method == "POST":
        entered_url = request.form.get("url", "").strip()
        if entered_url:
            try:
                prediction_result = predict_url(entered_url)

                # Save to DB if user is logged in
                if "user" in session and prediction_result and "final_verdict" in prediction_result:
                    user_obj = None
                    if session["user"].get("email"):
                        user_obj = User.query.filter_by(email=session["user"]["email"]).first()
                    elif session["user"].get("phone"):
                        user_obj = User.query.filter_by(phone=session["user"]["phone"]).first()

                    if user_obj:
                        new_detection = Detection(
                            user_id=user_obj.id,
                            type="url",
                            input_data=entered_url,
                            result=prediction_result["final_verdict"],
                            confidence=float(prediction_result.get("confidence", 0.0))
                        )
                        db.session.add(new_detection)
                        db.session.commit()

            except Exception as e:
                prediction_result = {"error": f"❗ Error analyzing URL: {str(e)}"}
        else:
            prediction_result = {"error": "❗ Please enter a URL to analyze."}

    return render_template("url_phishing_detector.html", prediction=prediction_result, url=entered_url)

# --- Apply no-cache headers globally ---
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/history", methods=["GET"])
@nocache
def history():
    if "user" not in session:
        return redirect(url_for("login"))

    user = User.query.get(session["user"]["id"])
    if not user:
        flash("User not found, please login again.", "danger")
        session.clear()
        return redirect(url_for("login"))

    result_filter = request.args.get("result")

    query = Detection.query.filter_by(user_id=user.id)
    if result_filter:
        query = query.filter(Detection.result.ilike(result_filter))

    history = query.order_by(Detection.timestamp.desc()).all()

    # Get flash messages explicitly for history delete
    history_deleted_flash = None
    for category, msg in get_flashed_messages(with_categories=True):
        if category == "history_deleted":
            history_deleted_flash = msg

    return render_template("history.html", user_history=history, history_deleted_flash=history_deleted_flash)


@app.route("/clear_history", methods=["POST"])
@nocache
def clear_history():
    if "user" not in session:
        return redirect(url_for("login"))

    user = User.query.get(session["user"]["id"])
    if not user:
        flash("User not found, please login again.", "danger")
        session.clear()
        return redirect(url_for("login"))

    Detection.query.filter_by(user_id=user.id).delete()
    db.session.commit()

    flash("History cleared successfully!", "history_deleted")
    return redirect(url_for("history"))


# Forgot Password Page
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")
        user = User.query.filter_by(email=email).first()

        if user:
            token = s.dumps(email, salt="password-reset-salt")
            reset_link = url_for("reset_with_token", token=token, _external=True)

            try:
                message = Mail(
                    from_email=VERIFIED_SENDER,
                    to_emails=email,
                    subject="Reset Your Password",
                    html_content = f"""
<html>
  <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <div style="max-width: 600px; margin: auto; padding: 20px; border: 1px solid #eaeaea; border-radius: 10px;">
      <h2 style="color: #1a73e8;">BaitBlocker Password Reset</h2>
      <p>Hello,</p>
      <p>We received a request to reset your password for your BaitBlocker account associated with <strong>{email}</strong>.</p>
      <p>Click the button below to reset your password:</p>
      <p style="text-align: center; margin: 30px 0;">
        <a href="{reset_link}" style="
          background-color: #1a73e8;
          color: white;
          padding: 12px 25px;
          text-decoration: none;
          border-radius: 5px;
          font-weight: bold;
        ">Reset Password</a>
      </p>
      <p>If you did not request a password reset, please ignore this email. Your account remains secure.</p>
      <p>For security reasons, this link will expire in 1 hour.</p>
      <br>
      <p>Thanks,<br><strong>The BaitBlocker Team</strong></p>
    </div>
  </body>
</html>
"""

                )
                sg = SendGridAPIClient(SENDGRID_API_KEY)
                sg.send(message)

                # Redirect to separate email_sent page
                return redirect(url_for("email_sent"))
            except Exception as e:
                flash(f"Error sending email: {str(e)}", "danger")
        else:
            flash("No account found with this email", "danger")

    return render_template("forgot_password.html")


# Email Sent Confirmation Page
@app.route("/email_sent")
def email_sent():
    return render_template("email_sent.html")


# Reset Password Page
@app.route("/reset/<token>", methods=["GET", "POST"])
def reset_with_token(token):
    try:
        email = s.loads(token, salt="password-reset-salt", max_age=3600)
    except:
        flash("The reset link is invalid or expired", "danger")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        new_password = request.form.get("password")
        hashed = generate_password_hash(new_password)
        user = User.query.filter_by(email=email).first()
        if user:
            user.password = hashed
            db.session.commit()
            flash("Password reset successfully ✔️", "success")
            return redirect(url_for("login"))

    return render_template("reset.html")
@app.route('/change-password', methods=['POST'])
def change_password():
    user_id = session.get('user_id') or session.get('user', {}).get('id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Please login first.'})

    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    if not new_password or not confirm_password:
        return jsonify({'success': False, 'error': 'All fields are required.'})

    if new_password != confirm_password:
        return jsonify({'success': False, 'error': 'Passwords do not match.'})

    user = User.query.get(user_id)
    if not user:
        return jsonify({'success': False, 'error': 'User not found.'})

    user.password = generate_password_hash(new_password)
    db.session.commit()

    return jsonify({'success': True, 'message': 'Password updated successfully!'})


if __name__ == "__main__":
    app.run(debug=True)