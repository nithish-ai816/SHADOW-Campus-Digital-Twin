import base64
import json
import smtplib
import ssl
import uuid
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
ASSETS       = BASE / "assets"
MAP_IMAGE    = ASSETS / "campus_map.jpg"
LOGO_IMAGE   = ASSETS / "shadow_mark.png"
AVATAR_IMAGE = ASSETS / "admin_avatar.png"
DATA_FILE    = BASE / "shadow_data.json"
USERS_FILE   = BASE / "shadow_users.json"

st.set_page_config(
    page_title="SHADOW | Campus Energy Digital Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# USER STORE  (local JSON, hashed passwords)
# ─────────────────────────────────────────────────────────
import hashlib, re as _re

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def _load_users() -> dict:
    if not USERS_FILE.exists():
        # seed one default admin so the app is never locked out
        default = {"admin": {"name":"Admin","email":"admin@shadow.local",
                              "role":"Admin","pw":_hash("Shadow@2024")}}
        USERS_FILE.write_text(json.dumps(default, indent=2), encoding="utf-8")
        return default
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except:
        return {}

def _save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")

def _valid_email(e: str) -> bool:
    return bool(_re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", e))

def _pw_strong(pw: str) -> str:
    """Return empty string if OK, else error message."""
    if len(pw) < 8:            return "Password must be at least 8 characters."
    if not _re.search(r"[A-Z]", pw): return "Password must contain an uppercase letter."
    if not _re.search(r"[0-9]", pw): return "Password must contain a number."
    return ""

# ─────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────
DEFAULT_BLOCKS = ["A", "B", "C"]
DEFAULT_ROOMS  = [
    {"room":"A-101","block":"A","status":"Normal",          "power":1.8,"voltage":230.0,"capacity":40},
    {"room":"A-102","block":"A","status":"Normal",          "power":2.0,"voltage":231.0,"capacity":40},
    {"room":"A-103","block":"A","status":"Warning",         "power":4.2,"voltage":230.0,"capacity":40},
    {"room":"A-104","block":"A","status":"High Consumption","power":6.1,"voltage":230.0,"capacity":40},
    {"room":"B-101","block":"B","status":"Normal",          "power":2.1,"voltage":229.0,"capacity":35},
    {"room":"B-102","block":"B","status":"Normal",          "power":1.9,"voltage":230.0,"capacity":35},
    {"room":"B-103","block":"B","status":"Warning",         "power":4.5,"voltage":230.0,"capacity":35},
    {"room":"B-104","block":"B","status":"Normal",          "power":2.3,"voltage":231.0,"capacity":35},
    {"room":"C-101","block":"C","status":"Normal",          "power":2.0,"voltage":230.0,"capacity":30},
    {"room":"C-102","block":"C","status":"Warning",         "power":4.1,"voltage":230.0,"capacity":30},
    {"room":"C-103","block":"C","status":"High Consumption","power":5.7,"voltage":229.0,"capacity":30},
    {"room":"C-104","block":"C","status":"Offline",         "power":0.0,"voltage":0.0,  "capacity":30},
]
DEFAULT_SETTINGS = {"power_warning_kw":4.0,"power_high_kw":5.5,"voltage_min":210.0,"voltage_max":250.0}

# ─────────────────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────────────────
def save_data():
    DATA_FILE.write_text(json.dumps({
        "blocks":       st.session_state.blocks,
        "rooms":        st.session_state.rooms,
        "settings":     st.session_state.settings,
        "camera_creds": st.session_state.camera_creds,
        "footage":      st.session_state.footage,
    }, indent=2), encoding="utf-8")

def load_data():
    if not DATA_FILE.exists():
        return {"blocks":DEFAULT_BLOCKS.copy(),"rooms":[r.copy() for r in DEFAULT_ROOMS],
                "settings":DEFAULT_SETTINGS.copy(),"camera_creds":{},"footage":[]}
    try:
        d = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        blocks = d.get("blocks", DEFAULT_BLOCKS)
        rooms  = d.get("rooms",  DEFAULT_ROOMS)
        sett   = d.get("settings", DEFAULT_SETTINGS)
        if not isinstance(blocks,list) or not isinstance(rooms,list):
            raise ValueError
        valid = []
        for r in rooms:
            if not isinstance(r,dict): continue
            try:
                valid.append({"room":str(r["room"]),"block":str(r["block"]),
                    "status":str(r.get("status","Normal")),"power":float(r.get("power",0)),
                    "voltage":float(r.get("voltage",230)),"capacity":int(r.get("capacity",0)),
                    "faculty":r.get("faculty",[
                        {"role":"Advisor","name":"","email":""},
                        {"role":"Mentor 1","name":"","email":""},
                        {"role":"Mentor 2","name":"","email":""},
                    ])})
            except: continue
        cs = DEFAULT_SETTINGS.copy()
        if isinstance(sett,dict):
            for k in cs:
                try: cs[k]=float(sett.get(k,cs[k]))
                except: pass
        return {"blocks":sorted(set(str(b) for b in blocks)),"rooms":valid,"settings":cs,
                "camera_creds":d.get("camera_creds",{}) if isinstance(d.get("camera_creds"),dict) else {},
                "footage":d.get("footage",[]) if isinstance(d.get("footage"),list) else []}
    except:
        return {"blocks":DEFAULT_BLOCKS.copy(),"rooms":[r.copy() for r in DEFAULT_ROOMS],
                "settings":DEFAULT_SETTINGS.copy(),"camera_creds":{},"footage":[]}

# ─────────────────────────────────────────────────────────
# GLOBAL CSS  (hides Streamlit toolbar, full dark theme)
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── hide Streamlit's own header/toolbar ── */
[data-testid="stHeader"]          { display:none !important; }
[data-testid="stToolbar"]         { display:none !important; }
#MainMenu                         { display:none !important; }
footer                            { display:none !important; }

/* ── base ── */
html, body, .stApp { background:#071629 !important; color:#eaf4ff; }
.block-container   { padding-top:0 !important; padding-bottom:2rem; max-width:1680px; }

/* ── sidebar ── */
section[data-testid="stSidebar"]  { background:#041a2c; border-right:1px solid #0d3a55; }

/* ── custom top-bar ── */
.topbar {
    display:flex; align-items:center; gap:10px;
    background:#04192b; border-bottom:1px solid #0d3a55;
    padding:8px 18px; margin-bottom:6px;
    position:sticky; top:0; z-index:999;
}
.topbar-date  { color:#7fa3bf; font-size:11px; flex-shrink:0; min-width:140px; }
.topbar-search{
    flex:1; background:#061f35; border:1px solid #1a3f5c;
    border-radius:7px; padding:6px 12px; color:#cde4f5;
    font-size:12px; outline:none;
}
.topbar-search::placeholder { color:#4e7291; }
.topbar-btn {
    background:#061f35; border:1px solid #1a3f5c; color:#cde4f5;
    border-radius:7px; padding:5px 13px; font-size:12px; cursor:pointer;
    white-space:nowrap; position:relative;
}
.topbar-btn:hover { background:#0d3455; border-color:#2ad4e8; }
.notif-badge {
    position:absolute; top:-5px; right:-5px;
    background:#e84545; color:#fff; font-size:9px; font-weight:800;
    border-radius:50%; width:16px; height:16px;
    display:flex; align-items:center; justify-content:center;
}

/* ── dark popup (notifications / profile menu) ── */
.popup-card {
    background:#07213a; border:1px solid #1a3f5c;
    border-radius:11px; padding:18px 20px;
    box-shadow:0 8px 32px rgba(0,0,0,.55);
}
.popup-header {
    display:flex; justify-content:space-between; align-items:center;
    margin-bottom:12px; padding-bottom:10px; border-bottom:1px solid #132d44;
}
.popup-title  { font-size:14px; font-weight:750; color:#e8f4ff; }
.popup-action { font-size:11px; color:#2ab4c8; cursor:pointer; }
.notif-row {
    display:flex; align-items:flex-start; gap:10px;
    padding:9px 0; border-bottom:1px solid #0e2a3d; font-size:12px;
}
.notif-row:last-child { border-bottom:0; }
.ndot  { width:10px; height:10px; border-radius:50%; margin-top:3px; flex-shrink:0; }
.nmain { color:#c0d8ef; line-height:1.4; }
.ntime { font-size:10px; color:#4e7291; margin-top:2px; }

/* ── admin profile popup ── */
.profile-head {
    display:flex; align-items:center; gap:13px;
    margin-bottom:14px; padding-bottom:12px; border-bottom:1px solid #132d44;
}
.profile-av {
    width:44px; height:44px; border-radius:50%;
    background:#0879ad; display:flex; align-items:center;
    justify-content:center; font-size:22px; flex-shrink:0;
    border:2px solid #1a4d72;
}
.profile-name { font-size:15px; font-weight:750; color:#e8f4ff; }
.profile-role { font-size:10px; color:#7fa3bf; margin-top:2px; }
.pmenu-item {
    display:flex; align-items:center; gap:10px;
    padding:8px 10px; border-radius:7px; cursor:pointer;
    font-size:13px; color:#c0d8ef; margin-bottom:2px;
    border:1px solid transparent; transition:all .15s;
}
.pmenu-item:hover { background:#0d3455; border-color:#1a4d72; color:#fff; }
.pmenu-count {
    margin-left:auto; background:#1a3a57; border-radius:10px;
    padding:1px 7px; font-size:10px; color:#7fa3bf;
}

/* ── brand ── */
.brand { display:flex; align-items:center; gap:11px;
         padding:4px 0 20px; border-bottom:1px solid #0e2f46; margin-bottom:14px; }
.brand-mark { width:44px; height:44px; object-fit:contain;
              border-radius:0 !important; background:transparent !important; box-shadow:none !important; }
.brand-name { font-size:21px; font-weight:850; letter-spacing:2px; color:#f2f8ff; }
.brand-sub  { font-size:10px; color:#82a4c3; white-space:nowrap; }

/* ── cards ── */
.card,.block-card {
    background:#091e32; border:1px solid #173751;
    border-radius:9px; padding:14px; margin-bottom:10px;
}
.metric-label { color:#8eabc6; font-size:11px; }
.metric-value { font-size:24px; font-weight:800; color:#f2f8ff; margin-top:7px; }
.panel-title  { font-size:13px; font-weight:700; color:#e7f3ff; margin-bottom:4px; }
.panel-sub    { color:#8da9c6; font-size:11px; margin-top:4px; }
.block-title  { color:#edf7ff; font-size:15px; font-weight:750; }
.block-sub,.block-count { color:#91abc5; font-size:10px; margin-top:5px; }
.status-row   { display:flex; justify-content:space-between; align-items:center;
                padding:9px 0; border-bottom:1px solid #18334c; font-size:11px; }
.status-row:last-child { border-bottom:0; }
.status-left  { display:flex; align-items:center; gap:9px; color:#c5d8ea; }
.dot          { display:inline-block; border-radius:50%; }
.status-num   { color:#edf7ff; font-weight:750; }
.title        { font-size:27px; font-weight:800; color:#f0f7ff; line-height:1.25; }
.subtitle     { color:#8da9c6; font-size:12px; margin-top:5px; }
.small-note   { font-size:10px; color:#6485a5; margin-top:12px; }

/* ── footage card ── */
.footage-card { background:#091e32; border:1px solid #173751; border-radius:9px; padding:13px 16px; margin-bottom:9px; }
.footage-id   { font-size:10px; color:#5d83a4; font-family:monospace; }
.footage-meta { font-size:11px; color:#8da9c6; margin-top:4px; }
.footage-ttl  { font-size:13px; font-weight:700; color:#d8eeff; }

/* ── settings tabs look ── */
.set-section  { background:#091e32; border:1px solid #173751; border-radius:10px; padding:18px 20px; margin-bottom:14px; }
.set-section-title { font-size:13px; font-weight:750; color:#5dc8e2; text-transform:uppercase;
                     letter-spacing:.8px; margin-bottom:12px; }

/* ── inputs ── */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"],
div[data-testid="stPasswordInput"] input {
    background:#061e34 !important; color:#eaf4ff !important;
    border-color:#1a3f5c !important; border-radius:7px !important;
}
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stPasswordInput"] input::placeholder { color:#4e7291 !important; }

/* ── buttons (ALL variants including form submit) ── */
div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button,
div[data-testid="stFormSubmitButton"] > button {
    background:#0a2847 !important; border:1px solid #1b4567 !important;
    color:#dceafa !important; border-radius:7px !important; font-size:12px !important;
    min-height:38px !important;
}
div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background:#0f4066 !important; border-color:#2ad4e8 !important; color:#fff !important;
}
/* ── primary submit button (Sign In / Create Account / Reset) ── */
div[data-testid="stFormSubmitButton"][data-type="primary"] > button,
div[data-testid="stFormSubmitButton"]:first-child > button {
    background:#0e4a8a !important; border-color:#1a6fbf !important;
    font-weight:700 !important; font-size:13px !important;
}
[data-testid="stImage"] img { border-radius:6px; }
[data-testid="stSidebar"] [data-testid="stImage"] img { border-radius:0 !important; background:transparent !important; }
hr { border-color:#183750; }
div[data-testid="stDataFrame"] { border:1px solid #183750; border-radius:8px; }

/* ── auth pages ── */
.auth-bg {
    min-height:100vh; background:#071629;
    display:flex; align-items:center; justify-content:center;
    padding:30px 16px;
}
.auth-card {
    width:100%; max-width:420px;
    background:#091e32; border:1px solid #173751;
    border-radius:16px; padding:40px 36px;
    box-shadow:0 12px 48px rgba(0,0,0,.55);
}
.auth-logo-wrap { text-align:center; margin-bottom:20px; }
.auth-logo-icon { font-size:40px; }
.auth-app-name  { font-size:26px; font-weight:900; letter-spacing:2.5px;
                  color:#f0f7ff; margin-top:4px; }
.auth-tagline   { font-size:11px; color:#7fa3bf; margin-top:3px; }
.auth-heading   { font-size:18px; font-weight:750; color:#e8f4ff;
                  margin:18px 0 4px; text-align:center; }
.auth-sub       { font-size:12px; color:#7fa3bf; text-align:center; margin-bottom:20px; }
.auth-divider   { display:flex; align-items:center; gap:10px;
                  margin:16px 0; color:#3a5a78; font-size:11px; }
.auth-divider::before,.auth-divider::after {
    content:""; flex:1; border-top:1px solid #173751; }
.auth-google-btn {
    display:flex; align-items:center; justify-content:center; gap:10px;
    width:100%; padding:10px 16px; border-radius:8px;
    border:1px solid #1e4a6e; background:#061f35;
    color:#cde4f5; font-size:13px; font-weight:500; cursor:pointer;
    transition:all .15s; text-decoration:none;
}
.auth-google-btn:hover { background:#0d3455; border-color:#2ad4e8; color:#fff; }
.auth-google-icon { font-size:18px; }
.auth-link      { color:#2ab4c8; font-size:12px; cursor:pointer; text-decoration:none; }
.auth-link:hover{ text-decoration:underline; }
.auth-footer    { text-align:center; margin-top:20px; font-size:12px; color:#4e7291; }
.pw-rules       { background:#041527; border:1px solid #173751; border-radius:8px;
                  padding:10px 14px; margin:8px 0 0; font-size:11px; color:#7fa3bf; }
.pw-rule        { margin:3px 0; }
.pw-ok          { color:#35d69a; }
.pw-fail        { color:#ff6b6b; }
.role-badge     { display:inline-block; border-radius:20px; padding:2px 10px; font-size:11px;
                  font-weight:700; letter-spacing:.5px; }

@media (max-width:1050px) {
    .block-container { padding:10px !important; }
    .title { font-size:22px; }
    .metric-value { font-size:19px; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────────────────
for _k, _v in [("logged_in", False), ("auth_page", "signin"), ("auth_msg", ""),
                ("auth_msg_type", "error"), ("current_user", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═════════════════════════════════════════════════════════
# AUTH GATE  — Sign In / Sign Up / Forgot Password
# ═════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    _users = _load_users()

    # ── shared logo header ──────────────────────────────
    lb = LOGO_IMAGE.read_bytes() if LOGO_IMAGE.exists() else None
    logo_html = (f'<img src="data:image/png;base64,{base64.b64encode(lb).decode()}" '
                 f'style="height:54px;object-fit:contain" />' if lb else
                 '<div style="font-size:44px">⚡</div>')

    # helper to render the card top
    def _auth_header(heading, sub):
        st.markdown(
            f'<div class="auth-card">'
            f'<div class="auth-logo-wrap">{logo_html}'
            f'<div class="auth-app-name">SHADOW</div>'
            f'<div class="auth-tagline">Campus Energy Digital Twin</div></div>'
            f'<div class="auth-heading">{heading}</div>'
            f'<div class="auth-sub">{sub}</div>',
            unsafe_allow_html=True,
        )

    # ── message banner ──────────────────────────────────
    if st.session_state.auth_msg:
        if st.session_state.auth_msg_type == "success":
            st.success(st.session_state.auth_msg)
        else:
            st.error(st.session_state.auth_msg)
        st.session_state.auth_msg = ""

    # ════════════════════════════════════════════════════
    # SIGN IN
    # ════════════════════════════════════════════════════
    if st.session_state.auth_page == "signin":
        _auth_header("Welcome back", "Sign in to your SHADOW account")

        # Google-style button (visual only — local app can't do real OAuth
        # without a server; clicking it fills the username hint)
        if st.button("🔵  G   Continue with Google (demo)", key="google_btn",
                     use_container_width=True):
            st.session_state.auth_msg      = ("Google OAuth requires a deployed server with credentials. "
                                               "Use the form below to sign in with your local account.")
            st.session_state.auth_msg_type = "error"
            st.rerun()

        st.markdown('<div class="auth-divider">or sign in with your account</div>',
                    unsafe_allow_html=True)

        with st.form("signin_form"):
            si_user = st.text_input("Username or Email",
                                    placeholder="Enter your username or email",
                                    key="si_user")
            si_pass = st.text_input("Password", placeholder="Enter your password",
                                    type="password", key="si_pass")
            si_submit = st.form_submit_button("🔐  Sign In", use_container_width=True)

        # action buttons OUTSIDE the form so they render with full dark styling
        _b1, _b2, _b3 = st.columns([1, 1, 1])
        with _b2:
            if st.button("Forgot password?", key="si_forgot_btn", use_container_width=True):
                st.session_state.auth_page = "forgot"; st.rerun()
        with _b3:
            if st.button("Create account", key="go_signup", use_container_width=True):
                st.session_state.auth_page = "signup"; st.rerun()

        if si_submit:
            u = si_user.strip().lower()
            match = None
            for uname, udata in _users.items():
                if uname.lower() == u or udata.get("email","").lower() == u:
                    match = (uname, udata); break
            if match and match[1]["pw"] == _hash(si_pass):
                st.session_state.logged_in    = True
                st.session_state.current_user = {"username": match[0], **match[1]}
                st.session_state.auth_msg     = ""
                st.rerun()
            else:
                st.session_state.auth_msg      = "Incorrect username / email or password."
                st.session_state.auth_msg_type = "error"
                st.rerun()

        st.markdown(
            '<div style="text-align:center;font-size:10px;color:#3a5a78;margin-top:10px">'
            'Default admin — username: <b style="color:#5d83a4">admin</b> &nbsp;'
            'password: <b style="color:#5d83a4">Shadow@2024</b></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)  # close auth-card

    # ════════════════════════════════════════════════════
    # SIGN UP
    # ════════════════════════════════════════════════════
    elif st.session_state.auth_page == "signup":
        _auth_header("Create your account", "Join SHADOW — Campus Energy Digital Twin")

        with st.form("signup_form"):
            su_name  = st.text_input("Full Name",  placeholder="Dr. A. Kumar", key="su_name")
            su_email = st.text_input("Email",       placeholder="you@example.com", key="su_email")
            su_user  = st.text_input("Username",    placeholder="Choose a username", key="su_user")
            su_role  = st.selectbox("Role",
                        ["Faculty","Campus Admin","Viewer","Maintenance Staff"], key="su_role")
            su_pass  = st.text_input("Password",    placeholder="Min 8 chars, 1 uppercase, 1 number",
                                     type="password", key="su_pass")
            su_pass2 = st.text_input("Confirm Password", placeholder="Re-enter password",
                                     type="password", key="su_pass2")

            # live password strength display
            if su_pass:
                rules = [
                    (len(su_pass) >= 8,              "At least 8 characters"),
                    (bool(_re.search(r"[A-Z]", su_pass)), "One uppercase letter"),
                    (bool(_re.search(r"[0-9]", su_pass)), "One number"),
                ]
                rules_html = "".join(
                    f'<div class="pw-rule {'pw-ok' if ok else 'pw-fail'}">'
                    f'{"✓" if ok else "✗"} {txt}</div>'
                    for ok, txt in rules
                )
                st.markdown(f'<div class="pw-rules">{rules_html}</div>',
                            unsafe_allow_html=True)

            su_submit = st.form_submit_button("✅  Create Account", use_container_width=True)

        if st.button("← Back to Sign In", key="su_back_btn", use_container_width=True):
            st.session_state.auth_page = "signin"; st.rerun()

        if su_submit:
            errs = []
            if not su_name.strip():                     errs.append("Full name is required.")
            if not _valid_email(su_email.strip()):      errs.append("Enter a valid email address.")
            if not su_user.strip():                     errs.append("Username is required.")
            if " " in su_user.strip():                  errs.append("Username must not contain spaces.")
            if su_user.strip().lower() in [k.lower() for k in _users]:
                                                        errs.append("Username already taken.")
            if su_email.strip().lower() in [v.get("email","").lower() for v in _users.values()]:
                                                        errs.append("Email already registered.")
            pw_err = _pw_strong(su_pass)
            if pw_err:                                  errs.append(pw_err)
            if su_pass != su_pass2:                     errs.append("Passwords do not match.")
            if errs:
                st.session_state.auth_msg      = " · ".join(errs)
                st.session_state.auth_msg_type = "error"
                st.rerun()
            else:
                _users[su_user.strip()] = {
                    "name":  su_name.strip(),
                    "email": su_email.strip().lower(),
                    "role":  su_role,
                    "pw":    _hash(su_pass),
                }
                _save_users(_users)
                st.session_state.auth_msg      = f"Account created! Welcome, {su_name.strip().split()[0]}. Please sign in."
                st.session_state.auth_msg_type = "success"
                st.session_state.auth_page     = "signin"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # FORGOT PASSWORD
    # ════════════════════════════════════════════════════
    elif st.session_state.auth_page == "forgot":
        _auth_header("Reset your password",
                     "Enter your registered email and a new password")

        with st.form("forgot_form"):
            fp_email = st.text_input("Registered Email", placeholder="you@example.com", key="fp_email")
            fp_user  = st.text_input("Username",         placeholder="Your username",   key="fp_user")
            fp_new   = st.text_input("New Password",  placeholder="Min 8 chars, 1 uppercase, 1 number",
                                     type="password", key="fp_new")
            fp_new2  = st.text_input("Confirm New Password", placeholder="Re-enter new password",
                                     type="password", key="fp_new2")
            fp_submit = st.form_submit_button("🔑  Reset Password", use_container_width=True)

        if st.button("← Back to Sign In", key="fp_back_btn", use_container_width=True):
            st.session_state.auth_page = "signin"; st.rerun()

        if fp_submit:
            uname = fp_user.strip().lower()
            match_key = None
            for k, v in _users.items():
                if k.lower() == uname and v.get("email","").lower() == fp_email.strip().lower():
                    match_key = k; break
            if not match_key:
                st.session_state.auth_msg      = "No account found with that username and email combination."
                st.session_state.auth_msg_type = "error"
                st.rerun()
            else:
                pw_err = _pw_strong(fp_new)
                if pw_err:
                    st.session_state.auth_msg      = pw_err
                    st.session_state.auth_msg_type = "error"
                    st.rerun()
                elif fp_new != fp_new2:
                    st.session_state.auth_msg      = "Passwords do not match."
                    st.session_state.auth_msg_type = "error"
                    st.rerun()
                else:
                    _users[match_key]["pw"] = _hash(fp_new)
                    _save_users(_users)
                    st.session_state.auth_msg      = "Password reset successfully. Please sign in with your new password."
                    st.session_state.auth_msg_type = "success"
                    st.session_state.auth_page     = "signin"
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# ─────────────────────────────────────────────────────────
# INIT SESSION after login
# ─────────────────────────────────────────────────────────

# Tiny valid playable MP4 (160 bytes, pure container, no audio/video stream — loads in browser)
_TINY_MP4_B64 = (
    "AAAAHGZ0eXBpc29tAAACAGlzb21pc28ybXA0MQAAAHxtb292AAAAdG12aGQAAAAAAAAA"
    "AAAAAAAAAAAAPoAAAAAAABAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAEAAAAAAAIAAAAIbWRhdA=="
)

# Sample footage injected once so the Video Library is never empty on first run
_SAMPLE_FOOTAGE = [
    {"id":"SF001A","classroom_id":"A-104","camera_id":"CAM-A104-01",
     "date":"2024-10-14","start_time":"09:00:00","end_time":"09:45:00","duration":"0h 45m 0s",
     "filename":"A104_morning_session.mp4","status":"Processed","data":_TINY_MP4_B64},
    {"id":"SF002B","classroom_id":"B-103","camera_id":"CAM-B103-01",
     "date":"2024-10-14","start_time":"10:00:00","end_time":"10:50:00","duration":"0h 50m 0s",
     "filename":"B103_lecture_oct14.mp4","status":"Processed","data":_TINY_MP4_B64},
    {"id":"SF003C","classroom_id":"C-102","camera_id":"CAM-C102-01",
     "date":"2024-10-15","start_time":"11:00:00","end_time":"11:30:00","duration":"0h 30m 0s",
     "filename":"C102_afternoon_check.mp4","status":"Pending","data":_TINY_MP4_B64},
    {"id":"SF004A","classroom_id":"A-103","camera_id":"CAM-A103-02",
     "date":"2024-10-15","start_time":"14:00:00","end_time":"14:55:00","duration":"0h 55m 0s",
     "filename":"A103_energy_audit.mp4","status":"Uploaded","data":_TINY_MP4_B64},
    {"id":"SF005B","classroom_id":"B-101","camera_id":"CAM-B101-01",
     "date":"2024-10-16","start_time":"08:30:00","end_time":"09:15:00","duration":"0h 45m 0s",
     "filename":"B101_early_morning.mp4","status":"Processed","data":_TINY_MP4_B64},
]

if "initialized" not in st.session_state:
    ld = load_data()
    st.session_state.blocks             = ld["blocks"]
    st.session_state.rooms              = ld["rooms"]
    st.session_state.settings           = ld["settings"]
    st.session_state.camera_creds       = ld["camera_creds"]
    # Inject sample footage only if none stored yet
    existing = ld["footage"]
    sample_ids = {f["id"] for f in existing}
    for sf in _SAMPLE_FOOTAGE:
        if sf["id"] not in sample_ids:
            existing.append(sf)
    st.session_state.footage            = existing
    st.session_state.page               = "Home"
    st.session_state.show_notif         = False
    st.session_state.show_admin_menu    = False
    st.session_state.faculty_open       = {}
    st.session_state.initialized        = True
    save_data()   # persist sample footage immediately

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def go_to(page):
    st.session_state.page            = page
    st.session_state.show_admin_menu = False
    st.session_state.show_notif      = False

def safe_bytes(path):
    return path.read_bytes() if path.exists() else None

def b64(path):
    b = safe_bytes(path)
    return base64.b64encode(b).decode() if b else None

def rooms_df():
    return pd.DataFrame(
        [[r["room"],r["block"],r["status"],r["power"],r["voltage"],r["capacity"]]
         for r in st.session_state.rooms],
        columns=["Room","Block","Status","Power (kW)","Voltage (V)","Capacity"]
    )

def eff_status(room):
    if room["status"] == "Offline": return "Offline"
    p = float(room["power"]); s = st.session_state.settings
    if p >= s["power_high_kw"]:    return "High Consumption"
    if p >= s["power_warning_kw"]: return "Warning"
    return "Normal"

def get_alerts():
    out=[]; s=st.session_state.settings
    for room in st.session_state.rooms:
        if room["status"]=="Offline":
            out.append({"room":room["room"],"type":"Offline","msg":"Room offline","val":"Offline","sev":"Info"}); continue
        p=float(room["power"]); v=float(room["voltage"])
        if p>=s["power_high_kw"]:   out.append({"room":room["room"],"type":"High Consumption","msg":"Power exceeds high threshold","val":f"{p:.1f} kW","sev":"High"})
        elif p>=s["power_warning_kw"]: out.append({"room":room["room"],"type":"Power Warning","msg":"Power exceeds warning threshold","val":f"{p:.1f} kW","sev":"Warning"})
        if v>0 and (v<s["voltage_min"] or v>s["voltage_max"]):
            out.append({"room":room["room"],"type":"Voltage","msg":"Voltage outside range","val":f"{v:.1f} V","sev":"High"})
    return out

def send_email(smtp_host, smtp_port, smtp_user, smtp_pass, sender, recipient, subject, body):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = recipient
    msg.set_content(body)
    ctx = ssl.create_default_context()
    if smtp_port == 465:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=ctx, timeout=20) as srv:
            srv.login(smtp_user, smtp_pass)
            srv.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as srv:
            srv.ehlo(); srv.starttls(context=ctx); srv.ehlo()
            srv.login(smtp_user, smtp_pass)
            srv.send_message(msg)

# ─────────────────────────────────────────────────────────
# RUNTIME
# ─────────────────────────────────────────────────────────
now       = datetime.now()
greeting  = "Good Morning" if now.hour<12 else ("Good Afternoon" if now.hour<17 else "Good Evening")
all_rooms = rooms_df()
alerts    = get_alerts()
t_rooms   = len(all_rooms)
t_power   = sum(float(r["power"]) for r in st.session_state.rooms)
s_counts  = {s: sum(eff_status(r)==s for r in st.session_state.rooms)
             for s in ["Normal","Warning","High Consumption","Offline"]}

# ═════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════
with st.sidebar:
    lb = safe_bytes(LOGO_IMAGE)
    mark = (f'<img class="brand-mark" src="data:image/png;base64,{base64.b64encode(lb).decode()}" />'
            if lb else
            '<div class="brand-mark" style="display:flex;align-items:center;justify-content:center;'
            'background:#0879ad;font-size:25px">⚡</div>')
    st.markdown(
        f'<div class="brand">{mark}'
        '<div><div class="brand-name">SHADOW</div>'
        '<div class="brand-sub">Campus Energy Digital Twin</div></div></div>',
        unsafe_allow_html=True,
    )
    for icon, label in [
        ("⌂","Home"), ("♧","Campus Map"), ("▣","Rooms"),
        ("▥","Analytics"), ("🎥","Footage Storage"), ("▤","Reports"),
    ]:
        if st.button(f"{icon}   {label}", key=f"nav_{label}", use_container_width=True,
                     type="primary" if st.session_state.page==label else "secondary"):
            go_to(label); st.rerun()
    st.markdown("---")
    st.caption("PROTOTYPE MODE · Readings simulated · No live sensors")

# ═════════════════════════════════════════════════════════
# TOP BAR (sticky, dark, always visible)
# ═════════════════════════════════════════════════════════
tb_date, tb_search, tb_bell, tb_admin = st.columns([1.1, 2.6, 0.4, 0.85],
                                                    gap="small", vertical_alignment="center")
with tb_date:
    st.markdown(f'<div style="color:#7fa3bf;font-size:11px;padding-top:6px">'
                f'{now.strftime("%A, %d %b %Y")}</div>', unsafe_allow_html=True)

with tb_search:
    query = st.text_input("Search", placeholder="⌕  Search room, block or status…",
                          label_visibility="collapsed", key="gsearch")

with tb_bell:
    if st.button(f"🔔 {len(alerts)}", key="bell_btn", use_container_width=True,
                 help="Notifications"):
        st.session_state.show_notif      = not st.session_state.show_notif
        st.session_state.show_admin_menu = False

with tb_admin:
    _uname = (st.session_state.current_user or {}).get("name","Admin").split()[0]
    if st.button(f"👤  {_uname}  ▾", key="admin_btn", use_container_width=True):
        st.session_state.show_admin_menu = not st.session_state.show_admin_menu
        st.session_state.show_notif      = False

st.markdown('<hr style="margin:4px 0 8px;border-color:#0d3a55">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# NOTIFICATIONS POPUP
# ─────────────────────────────────────────────────────────
if st.session_state.show_notif:
    # Force the Streamlit container wrapper to be transparent/dark so no white box shows
    st.markdown("""
    <style>
    [data-testid="stVerticalBlockBorderWrapper"],[data-testid="stVerticalBlock"] {
        background:transparent !important;
        border:none !important;
        box-shadow:none !important;
    }
    div.element-container:has(.popup-card) { background:transparent !important; }
    </style>
    """, unsafe_allow_html=True)

    sev_col = {"High":"#e84545","Warning":"#ffbd4a","Info":"#8296ad"}
    times   = ["2 min ago","12 min ago","25 min ago","1 hour ago","3 hours ago"]
    rows_html = ""
    if not alerts:
        rows_html = '<div style="color:#7fa3bf;font-size:12px;padding:10px 0">No active alerts at current thresholds.</div>'
    else:
        for i, a in enumerate(alerts[:5]):
            c = sev_col.get(a["sev"],"#8296ad")
            t = times[i] if i<len(times) else "—"
            rows_html += (
                f'<div class="notif-row">'
                f'<div class="ndot" style="background:{c}"></div>'
                f'<div><div class="nmain"><b>{a["room"]}</b> — {a["type"]}: {a["msg"]} · {a["val"]}</div>'
                f'<div class="ntime">{t}</div></div></div>'
            )
    # Build the entire popup — including the button — as pure HTML inside one dark card
    # The st.button below is the only interactive element needed
    st.markdown(
        f'<div class="popup-card" style="background:#07213a;border:1px solid #1a3f5c;'
        f'border-radius:11px;padding:16px 20px;box-shadow:0 8px 32px rgba(0,0,0,.6)">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #132d44">'
        f'<span style="font-size:14px;font-weight:700;color:#e8f4ff">🔔 Notifications</span>'
        f'<span style="font-size:11px;color:#2ab4c8">Mark all as read</span>'
        f'</div>{rows_html}</div>',
        unsafe_allow_html=True,
    )
    if st.button("View all alerts →", key="notif_view_all", use_container_width=True):
        go_to("Alerts"); st.rerun()

# ─────────────────────────────────────────────────────────
# ADMIN PROFILE POPUP
# ─────────────────────────────────────────────────────────
if st.session_state.show_admin_menu:
    av = b64(AVATAR_IMAGE)
    av_html = (f'<img src="data:image/png;base64,{av}" style="width:44px;height:44px;border-radius:50%;'
               f'object-fit:cover;border:2px solid #1a4d72" />'
               if av else '<div class="profile-av">👤</div>')
    _cu = st.session_state.current_user or {}
    _disp_name  = _cu.get("name", "Admin")
    _disp_role  = _cu.get("role", "Campus Administrator")
    _disp_email = _cu.get("email", "")
    st.markdown(
        f'<div class="popup-card">'
        f'<div class="profile-head">{av_html}'
        f'<div><div class="profile-name">{_disp_name}</div>'
        f'<div class="profile-role">{_disp_role}'
        f'{" · " + _disp_email if _disp_email else ""}</div></div></div></div>',
        unsafe_allow_html=True,
    )
    menu = [
        ("👤","My Profile","Profile"),
        ("🏛️","Campus Management","Admin Settings"),
        ("▣","Manage Blocks & Rooms","Admin Settings"),
        ("⚡","Energy Thresholds","Settings"),
        ("🔔","Notification Preferences","Alerts"),
        ("📋","Activity Log","Reports"),
        ("⚙","System Settings","Settings"),
    ]
    counts = {"My Profile":"","Campus Management":str(len(st.session_state.blocks)),
              "Manage Blocks & Rooms":str(len(st.session_state.rooms)),
              "Energy Thresholds":"","Notification Preferences":str(len(alerts)),
              "Activity Log":"","System Settings":""}
    c1, c2 = st.columns(2)
    for i,(ic,lbl,tgt) in enumerate(menu):
        col = c1 if i%2==0 else c2
        with col:
            if st.button(f"{ic} {lbl}", key=f"pmenu_{lbl}", use_container_width=True):
                go_to(tgt); st.rerun()
    st.markdown('<hr style="margin:8px 0;border-color:#0d3a55">', unsafe_allow_html=True)
    if st.button("🔴  Sign Out", key="signout", use_container_width=True):
        st.session_state.logged_in       = False
        st.session_state.initialized     = False
        st.session_state.show_admin_menu = False
        st.session_state.current_user    = None
        st.session_state.auth_page       = "signin"
        st.rerun()

# ─────────────────────────────────────────────────────────
# GLOBAL SEARCH
# ─────────────────────────────────────────────────────────
if query.strip():
    term = query.strip().lower()
    m = all_rooms[all_rooms.astype(str).apply(
        lambda c: c.str.lower().str.contains(term, regex=False, na=False)).any(axis=1)]
    with st.container(border=True):
        st.markdown("**Search Results**")
        if m.empty: st.info("No matching rooms found.")
        else:
            st.dataframe(m, use_container_width=True, hide_index=True)
            if st.button("Open Rooms page", key="sr_open"):
                st.session_state.page="Rooms"; st.rerun()

# ═════════════════════════════════════════════════════════
# HOME
# ═════════════════════════════════════════════════════════
if st.session_state.page == "Home":
    wc,m1,m2,m3 = st.columns([2.1,0.95,0.95,1.1], gap="small", vertical_alignment="center")
    with wc:
        st.markdown(f'<div class="title">{greeting},</div>'
                    '<div class="subtitle">Here is how your campus demo is doing.</div>',
                    unsafe_allow_html=True)
    for col,ic,lbl,val,clr in [
        (m1,"◉","Total Rooms",str(t_rooms),"#54d6e8"),
        (m2,"●","Active Alerts",str(len(alerts)),"#ff536a"),
        (m3,"ϟ","Total Load",f"{t_power:.1f} kW","#43d89a"),
    ]:
        with col:
            st.markdown(f'<div class="card"><div class="metric-label">'
                        f'<span style="color:{clr};font-size:14px">{ic}</span> {lbl}</div>'
                        f'<div class="metric-value">{val}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    left, right = st.columns([2.2,1], gap="small")

    with left:
        st.markdown("#### Campus Overview")
        if MAP_IMAGE.exists(): st.image(str(MAP_IMAGE), use_container_width=True)
        else: st.info("Add assets/campus_map.jpg to show the campus image.")

        st.markdown("#### Blocks")
        if not st.session_state.blocks:
            st.info("No blocks yet — add one in Admin Settings.")
        else:
            bcols = st.columns(min(3, len(st.session_state.blocks)))
            for idx,blk in enumerate(st.session_state.blocks):
                with bcols[idx % len(bcols)]:
                    br = [r for r in st.session_state.rooms if r["block"]==blk]
                    n=sum(eff_status(r)=="Normal" for r in br)
                    w=sum(eff_status(r)=="Warning" for r in br)
                    h=sum(eff_status(r)=="High Consumption" for r in br)
                    o=sum(eff_status(r)=="Offline" for r in br)
                    st.markdown(f'<div class="block-card">'
                                f'<div class="block-title">Block {blk}</div>'
                                f'<div class="block-sub">{len(br)} Rooms</div>'
                                f'<div class="block-count">🟢 {n} Normal &nbsp; 🟡 {w} Warning</div>'
                                f'<div class="block-count">🔴 {h} High &nbsp; ⚪ {o} Offline</div>'
                                f'</div>', unsafe_allow_html=True)
                    if st.button(f"Open Block {blk} →", key=f"ob_{blk}", use_container_width=True):
                        st.session_state.block_filter=blk; go_to("Rooms"); st.rerun()

    with right:
        st.markdown('<div class="panel-title">Room Status</div>', unsafe_allow_html=True)
        for lbl,clr in [("Normal","#35d69a"),("Warning","#ffbd4a"),
                        ("High Consumption","#ff5269"),("Offline","#8296ad")]:
            st.markdown(f'<div class="status-row"><div class="status-left">'
                        f'<span class="dot" style="background:{clr};width:11px;height:11px"></span>'
                        f'{lbl}</div><div class="status-num">{s_counts[lbl]}</div></div>',
                        unsafe_allow_html=True)
        if st.button("View All Rooms →", key="har_btn", use_container_width=True):
            go_to("Rooms"); st.rerun()

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Demo Power Usage</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-sub">Configured room total</div>'
                    f'<div style="font-size:22px;font-weight:800;color:#f0f6ff">{t_power:.1f} kW</div>',
                    unsafe_allow_html=True)
        hrs = ["6 AM","7 AM","8 AM","9 AM","10 AM","11 AM","12 PM","1 PM","2 PM","3 PM","4 PM","5 PM","6 PM","7 PM","8 PM","9 PM","10 PM"]
        vals= [29,27,32,37,34,41,38,45,39,42,48,40,44,51,43,47,41]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hrs,y=vals,mode="lines",
            line=dict(color="#32c9e7",width=2.2,shape="spline"),
            fill="tozeroy",fillcolor="rgba(35,195,224,0.14)",
            hovertemplate="%{x}<br>%{y} kW<extra></extra>"))
        fig.update_layout(height=185,margin=dict(l=0,r=0,t=4,b=0),
            paper_bgcolor="#091e32",plot_bgcolor="#091e32",showlegend=False,
            font=dict(color="#9bb2cb",size=9),
            xaxis=dict(showgrid=False,zeroline=False),
            yaxis=dict(visible=False,showgrid=True,gridcolor="#15334d"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<div class="small-note">SHADOW prototype · All readings are simulated. No live sensors connected.</div>',
                unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════
# CAMPUS MAP
# ═════════════════════════════════════════════════════════
elif st.session_state.page == "Campus Map":
    st.markdown('<div class="title">Campus Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Illustrative campus overview.</div><br>', unsafe_allow_html=True)
    if MAP_IMAGE.exists(): st.image(str(MAP_IMAGE), use_container_width=True)
    else: st.info("Place assets/campus_map.jpg to show campus image.")
    st.info("Map is illustrative — not linked to live sensors or a real floor plan.")
    for blk in st.session_state.blocks:
        br=[r for r in st.session_state.rooms if r["block"]==blk]
        with st.expander(f"Block {blk} — {len(br)} rooms"):
            if br:
                st.dataframe(pd.DataFrame([{"Room":r["room"],"Status":eff_status(r),
                    "Power (kW)":r["power"],"Voltage (V)":r["voltage"]} for r in br]),
                    use_container_width=True, hide_index=True)
            else: st.caption("No rooms in this block yet.")

# ═════════════════════════════════════════════════════════
# ROOMS
# ═════════════════════════════════════════════════════════
elif st.session_state.page == "Rooms":
    st.markdown('<div class="title">Rooms</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Browse configured rooms and simulated readings.</div><br>',
                unsafe_allow_html=True)
    fc1,fc2,fc3 = st.columns([2,1,1])
    with fc1: rq=st.text_input("Search",key="rps",value=st.session_state.get("rooms_search_filter",""),placeholder="Room name or status…")
    with fc2: sb=st.selectbox("Block",["All Blocks"]+st.session_state.blocks,key="rbs")
    with fc3: ss=st.selectbox("Status",["All Statuses","Normal","Warning","High Consumption","Offline"],key="rss")
    rows=[]
    for r in st.session_state.rooms:
        st2=eff_status(r)
        if sb!="All Blocks" and r["block"]!=sb: continue
        if ss!="All Statuses" and st2!=ss: continue
        if rq.strip().lower() not in f"{r['room']} {r['block']} {st2}".lower(): continue
        rows.append({"Room":r["room"],"Block":r["block"],"Status":st2,
                     "Power (kW)":r["power"],"Voltage (V)":r["voltage"],"Capacity":r["capacity"]})
    df2=pd.DataFrame(rows,columns=["Room","Block","Status","Power (kW)","Voltage (V)","Capacity"])
    st.dataframe(df2, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df2.to_csv(index=False).encode(), "shadow_rooms.csv","text/csv")
    st.caption("Demo values only — not live measurements.")

# ═════════════════════════════════════════════════════════
# ANALYTICS
# ═════════════════════════════════════════════════════════
elif st.session_state.page == "Analytics":
    st.markdown('<div class="title">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Room-level demo power comparisons.</div><br>', unsafe_allow_html=True)
    df3=rooms_df()
    if df3.empty: st.info("Add rooms to see analytics.")
    else:
        fig2=go.Figure()
        fig2.add_trace(go.Bar(x=df3["Room"],y=df3["Power (kW)"],marker_color="#29c9e4",
                             hovertemplate="%{x}<br>%{y:.1f} kW<extra></extra>"))
        fig2.update_layout(height=350,paper_bgcolor="#091e32",plot_bgcolor="#091e32",
            font=dict(color="#9bb2cb"),xaxis=dict(showgrid=False,title="Room"),
            yaxis=dict(gridcolor="#183750",title="Power (kW)"),margin=dict(l=15,r=15,t=15,b=15))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
        st.markdown("#### Block power summary")
        st.dataframe(df3.groupby("Block",as_index=False)["Power (kW)"].sum(),
                     use_container_width=True, hide_index=True)
    st.caption("Configured demo values only.")

# ═════════════════════════════════════════════════════════
# FOOTAGE STORAGE
# ═════════════════════════════════════════════════════════
elif st.session_state.page == "Footage Storage":
    st.markdown('<div class="title">📹 Footage Storage</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload, browse and manage classroom camera recordings.</div><br>',
                unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["📂 Video Library","⬆️ Upload","🔑 Camera Credentials","💾 Storage Info"])

    with t1:
        st.markdown("### Browse Recordings")
        fl1,fl2,fl3 = st.columns([1.5,1.5,1.5])
        with fl1: lib_room=st.selectbox("Classroom",["All Rooms"]+[r["room"] for r in st.session_state.rooms],key="lr")
        with fl2: lib_cam=st.text_input("Camera ID filter",key="lc",placeholder="CAM-A101-01")
        with fl3: lib_date=st.text_input("Date (YYYY-MM-DD)",key="ld",placeholder="2024-10-12")
        def fmatch(f):
            if lib_room!="All Rooms" and f.get("classroom_id")!=lib_room: return False
            if lib_cam.strip() and lib_cam.strip().lower() not in f.get("camera_id","").lower(): return False
            if lib_date.strip() and lib_date.strip() not in f.get("date",""): return False
            return True
        ff=[f for f in st.session_state.footage if fmatch(f)]
        if not ff: st.info("No recordings found. Use Upload tab to add one.")
        else:
            for rec in ff:
                pc={"Processed":"#35d69a","Pending":"#ffbd4a","Failed":"#ff5269","Uploaded":"#54d6e8"}.get(rec.get("status",""),"#8296ad")
                # card header
                st.markdown(f'<div class="footage-card">'
                            f'<div class="footage-id">ID: {rec.get("id","—")} · Camera: {rec.get("camera_id","—")}</div>'
                            f'<div class="footage-ttl">🎥 {rec.get("filename","Recording")}</div>'
                            f'<div class="footage-meta">📍 {rec.get("classroom_id","—")} | 📅 {rec.get("date","—")} | '
                            f'⏱ {rec.get("start_time","—")} → {rec.get("end_time","—")} ({rec.get("duration","—")}) | '
                            f'<span style="color:{pc}">● {rec.get("status","—")}</span></div></div>',
                            unsafe_allow_html=True)

                # ── inline video player ──────────────────
                if rec.get("data",""):
                    try:
                        video_bytes = base64.b64decode(rec["data"])
                        # Embed as HTML5 video so it plays directly in the page
                        vb64 = base64.b64encode(video_bytes).decode()
                        st.markdown(
                            f'<video controls style="width:100%;max-height:260px;border-radius:8px;'
                            f'background:#000;margin:6px 0 10px">'
                            f'<source src="data:video/mp4;base64,{vb64}" type="video/mp4" />'
                            f'Your browser does not support the video tag.</video>',
                            unsafe_allow_html=True,
                        )
                    except Exception:
                        st.caption("⚠ Could not decode video data.")
                else:
                    st.caption("📁 No video file attached to this record.")

                rc1,rc2,rc3=st.columns(3)
                with rc1:
                    if rec.get("data",""):
                        st.download_button("⬇ Download",base64.b64decode(rec["data"]),
                            rec.get("filename","rec.mp4"),"video/mp4",key=f"dl_{rec['id']}",use_container_width=True)
                    else: st.button("⬇ No file",key=f"dl_{rec['id']}",disabled=True,use_container_width=True)
                with rc2:
                    if st.button("🤖 Analyse",key=f"an_{rec['id']}",use_container_width=True):
                        st.info("AI analysis queued. In production this sends footage to the CV pipeline.")
                with rc3:
                    if st.button("🗑 Delete",key=f"de_{rec['id']}",use_container_width=True):
                        st.session_state.footage=[x for x in st.session_state.footage if x["id"]!=rec["id"]]
                        save_data(); st.success("Deleted."); st.rerun()
        st.markdown("---")
        sa,sb2,sc=st.columns(3)
        sa.metric("Total",len(st.session_state.footage))
        sb2.metric("Processed",sum(1 for f in st.session_state.footage if f.get("status")=="Processed"))
        sc.metric("Pending",sum(1 for f in st.session_state.footage if f.get("status") in ("Pending","Uploaded")))

    with t2:
        st.markdown("### Upload a Classroom Recording")
        with st.form("uf"):
            ur=st.selectbox("Classroom",[r["room"] for r in st.session_state.rooms],key="uur")
            uc=st.text_input("Camera ID",placeholder="CAM-A101-01",key="uuc")
            ud=st.date_input("Date",value=now.date(),key="uud")
            uc1,uc2=st.columns(2)
            with uc1: us=st.time_input("Start",key="uus")
            with uc2: ue=st.time_input("End",key="uue")
            uff=st.file_uploader("Video file",type=["mp4","avi","mov","mkv"],key="uuf")
            ufs=st.form_submit_button("Save Recording",use_container_width=True)
        if ufs:
            if not uc.strip(): st.error("Enter Camera ID.")
            elif ue<=us: st.error("End time must be after start time.")
            else:
                ss2=us.hour*3600+us.minute*60+us.second
                es2=ue.hour*3600+ue.minute*60+ue.second
                ds2=max(0,es2-ss2)
                rec={"id":str(uuid.uuid4())[:8].upper(),"classroom_id":ur,"camera_id":uc.strip(),
                     "date":str(ud),"start_time":str(us),"end_time":str(ue),
                     "duration":f"{ds2//3600}h {(ds2%3600)//60}m {ds2%60}s",
                     "filename":uff.name if uff else f"{ur}_{ud}.mp4","status":"Uploaded",
                     "data":base64.b64encode(uff.read()).decode() if uff else ""}
                st.session_state.footage.append(rec); save_data()
                st.success(f"Saved — ID: {rec['id']}"); st.rerun()

    with t3:
        st.markdown("### Camera Credentials")
        st.warning("⚠ Prototype credential store — do NOT use real production passwords here.")
        with st.form("ccf"):
            ccr=st.selectbox("Room",[r["room"] for r in st.session_state.rooms],key="ccr")
            ex=st.session_state.camera_creds.get(ccr,{})
            cid=st.text_input("Camera ID", value=ex.get("camera_id",""),key="ccid")
            cu=st.text_input("Username",  value=ex.get("username",""),key="ccu")
            cp=st.text_input("Password",  value=ex.get("password",""),type="password",key="ccp")
            ci=st.text_input("IP / Stream URL (optional)",value=ex.get("ip",""),key="cci",
                             placeholder="rtsp://192.168.1.x:554/stream")
            ccs=st.form_submit_button("Save Credentials",use_container_width=True)
        if ccs:
            if not cu.strip(): st.error("Username required.")
            elif not cp.strip(): st.error("Password required.")
            else:
                st.session_state.camera_creds[ccr]={"camera_id":cid.strip(),"username":cu.strip(),
                                                     "password":cp.strip(),"ip":ci.strip()}
                save_data(); st.success(f"Credentials saved for {ccr}.")
        st.markdown("#### Configured Cameras")
        cr2=[{"Room":rid,"Camera ID":c.get("camera_id","—"),"Username":c.get("username","—"),
              "Password":"●●●●●●","IP / URL":c.get("ip","—")}
             for rid,c in st.session_state.camera_creds.items()]
        if cr2: st.dataframe(pd.DataFrame(cr2), use_container_width=True, hide_index=True)
        else: st.caption("No credentials configured yet.")

    with t4:
        st.markdown("### Storage Overview")
        so1,so2=st.columns(2)
        so1.metric("Total recordings",len(st.session_state.footage))
        so2.metric("Rooms with footage",len(set(f.get("classroom_id","") for f in st.session_state.footage)))
        st.markdown("#### Retention policy")
        st.markdown("""| Rule | Value |\n|---|---|\n| Default retention | 30 days |\n| Delete permissions | Admin only |\n| Processed | Kept until manual delete |\n| Failed uploads | Flagged for review |""")
        if st.session_state.footage:
            if st.button("🗑 Delete ALL recordings",key="da"):
                st.session_state.footage=[]; save_data(); st.success("All cleared."); st.rerun()

# ═════════════════════════════════════════════════════════
# REPORTS
# ═════════════════════════════════════════════════════════
elif st.session_state.page == "Reports":
    st.markdown('<div class="title">Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Export current room configuration.</div><br>', unsafe_allow_html=True)
    rdf=rooms_df()
    if not rdf.empty: rdf["Status"]=[eff_status(r) for r in st.session_state.rooms]
    st.dataframe(rdf, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", rdf.to_csv(index=False).encode(), "shadow_report.csv","text/csv")
    st.caption("Reflects current prototype configuration.")

# ═════════════════════════════════════════════════════════
# ALERTS
# ═════════════════════════════════════════════════════════
elif st.session_state.page == "Alerts":
    import time as _time

    # ── extra CSS just for alert cards ──────────────────
    st.markdown("""
    <style>
    .alert-card {
        background:#091e32; border:1px solid #1a3f5c;
        border-radius:14px; padding:20px 22px; margin-bottom:16px;
        position:relative;
    }
    .alert-card-high   { border-left:4px solid #e84545; }
    .alert-card-warn   { border-left:4px solid #ffbd4a; }
    .alert-card-info   { border-left:4px solid #8296ad; }
    .alert-badge {
        display:inline-block; border-radius:20px; padding:3px 11px;
        font-size:10px; font-weight:800; letter-spacing:.6px; margin-bottom:8px;
    }
    .badge-high { background:#3d0e0e; color:#ff6b6b; border:1px solid #e84545; }
    .badge-warn { background:#3d2d00; color:#ffbd4a; border:1px solid #ffbd4a; }
    .badge-info { background:#1a2535; color:#8296ad; border:1px solid #8296ad; }
    .alert-room { font-size:18px; font-weight:800; color:#f0f7ff; margin-bottom:4px; }
    .alert-desc { font-size:13px; color:#c0d8ef; margin-bottom:10px; line-height:1.5; }
    .alert-val  { font-size:12px; color:#5d83a4; margin-bottom:14px; }
    .countdown-ring {
        text-align:center; padding:10px 0 6px;
    }
    .countdown-num {
        font-size:38px; font-weight:900; color:#f0f7ff;
        font-variant-numeric:tabular-nums; letter-spacing:2px;
    }
    .countdown-label { font-size:10px; color:#5d83a4; margin-top:2px; }
    .faculty-row {
        background:#061a2c; border:1px solid #173751; border-radius:8px;
        padding:10px 14px; margin-bottom:6px; font-size:12px;
    }
    .faculty-role  { color:#5d83a4; font-size:10px; }
    .faculty-name  { color:#e8f4ff; font-weight:700; }
    .faculty-email { color:#2ab4c8; font-size:11px; }
    .sent-badge    { color:#35d69a; font-size:11px; font-weight:700; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">⚠ Alert & Countdown</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Active room alerts with countdown, faculty notification and shutdown control.</div><br>',
        unsafe_allow_html=True,
    )

    # init per-alert state store
    if "alert_state" not in st.session_state:
        st.session_state.alert_state = {}   # {alert_key: {"kept_on":bool,"shutdown":bool,"notified":bool}}

    if not alerts:
        st.markdown(
            '<div style="background:#0a1e30;border:1px solid #173751;border-radius:12px;'
            'padding:28px;text-align:center;color:#35d69a;font-size:16px;font-weight:700">'
            '✅ No active alerts at current thresholds.<br>'
            '<span style="color:#5d83a4;font-size:12px;font-weight:400">'
            'All rooms are within configured limits.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        for a in alerts:
            akey = f"{a['room']}_{a['type']}"
            if akey not in st.session_state.alert_state:
                st.session_state.alert_state[akey] = {
                    "kept_on": False, "shutdown": False, "notified": False,
                    "notif_results": [],
                }
            astate = st.session_state.alert_state[akey]

            sev      = a["sev"]
            badge_cl = {"High":"badge-high","Warning":"badge-warn","Info":"badge-info"}.get(sev,"badge-info")
            card_cl  = {"High":"alert-card-high","Warning":"alert-card-warn","Info":"alert-card-info"}.get(sev,"alert-card-info")
            sev_icon = {"High":"🔴","Warning":"🟡","Info":"⚪"}.get(sev,"⚪")

            # ── find the room data ─────────────────────
            room_data = next((r for r in st.session_state.rooms if r["room"]==a["room"]), None)
            faculty   = room_data.get("faculty",[]) if room_data else []
            email_fac = [f for f in faculty if isinstance(f,dict) and f.get("email","").strip()]

            st.markdown(
                f'<div class="alert-card {card_cl}">'
                f'<span class="alert-badge {badge_cl}">{sev_icon} {sev.upper()}</span>'
                f'<div class="alert-room">⚡ SHADOW Alert — {a["room"]}</div>'
                f'<div class="alert-desc">{a["msg"]} while {'unoccupied (estimated)' if sev=="High" else "occupied"}.</div>'
                f'<div class="alert-val">Current load: <b style="color:#f0f7ff">{a["val"]}</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── layout: countdown | buttons | faculty ─
            col_cd, col_act, col_fac = st.columns([1, 1.4, 1.8], gap="small")

            with col_cd:
                if astate["shutdown"]:
                    st.markdown(
                        '<div class="countdown-ring">'
                        '<div class="countdown-num" style="color:#e84545">OFF</div>'
                        '<div class="countdown-label">Shutdown approved</div></div>',
                        unsafe_allow_html=True,
                    )
                elif astate["kept_on"]:
                    st.markdown(
                        '<div class="countdown-ring">'
                        '<div class="countdown-num" style="color:#35d69a">ON</div>'
                        '<div class="countdown-label">Faculty kept power ON</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # static countdown display (prototype)
                    cd_seconds = {"High":42,"Warning":90,"Info":180}.get(sev,60)
                    mm, ss2 = divmod(cd_seconds, 60)
                    st.markdown(
                        f'<div class="countdown-ring">'
                        f'<div style="font-size:11px;color:#5d83a4;margin-bottom:4px">Auto shutdown in</div>'
                        f'<div class="countdown-num">{mm:02d}:{ss2:02d}</div>'
                        f'<div class="countdown-label">🔔 Faculty can intervene if needed.</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            with col_act:
                st.markdown("**Action**")
                if not astate["kept_on"] and not astate["shutdown"]:
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        if st.button("✅ Keep ON", key=f"kon_{akey}", use_container_width=True):
                            st.session_state.alert_state[akey]["kept_on"] = True
                            st.rerun()
                    with bc2:
                        if st.button("🔌 Allow Shutdown", key=f"asd_{akey}", use_container_width=True):
                            st.session_state.alert_state[akey]["shutdown"] = True
                            st.rerun()
                elif astate["kept_on"]:
                    st.success("✅ Power kept ON by faculty decision.")
                    if st.button("↩ Undo", key=f"undo_{akey}", use_container_width=True):
                        st.session_state.alert_state[akey]["kept_on"] = False; st.rerun()
                else:
                    st.error("🔌 Shutdown approved. Appliances should be off.")
                    if st.button("↩ Undo", key=f"undo_{akey}", use_container_width=True):
                        st.session_state.alert_state[akey]["shutdown"] = False; st.rerun()

            with col_fac:
                st.markdown("**Notify Faculty**")

                # Always include nithishkumaran816@gmail.com as the test/admin recipient
                TEST_RECIPIENT = "nithishkumaran816@gmail.com"
                # Build the full recipients list: assigned faculty + test email (deduplicated)
                all_recipients = list(email_fac)
                existing_emails = {f["email"].lower() for f in email_fac}
                if TEST_RECIPIENT.lower() not in existing_emails:
                    all_recipients.append({"role":"Test / Admin","name":"Nithish Kumar","email":TEST_RECIPIENT})

                # Show who will receive
                for f in all_recipients:
                    already = f["email"] in [r.get("to","") for r in astate["notif_results"]]
                    tick = "✅" if already else "📧"
                    st.markdown(
                        f'<div class="faculty-row">'
                        f'<div class="faculty-role">{f.get("role","Faculty")}</div>'
                        f'<div class="faculty-name">{tick} {f.get("name") or "—"}</div>'
                        f'<div class="faculty-email">{f.get("email","")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                if not email_fac:
                    st.caption("No room faculty assigned — alert will go to admin only. Add faculty via ⚙ Settings → Faculty.")

                # Send button
                btn_lbl = "📤 Send Alert to Faculty" if not astate["notified"] else "🔁 Re-send Alert"
                if st.button(btn_lbl, key=f"notif_{akey}", use_container_width=True):
                    email_cfg = {}
                    try: email_cfg = dict(st.secrets.get("email", {}))
                    except: pass

                    if not all([email_cfg.get("smtp_host"), email_cfg.get("smtp_username"),
                                email_cfg.get("smtp_password")]):
                        st.error("⚠ SMTP not configured. Add your Gmail app-password to .streamlit/secrets.toml")
                    else:
                        results = []
                        for f in all_recipients:
                            subject = f"⚡ SHADOW Alert — {a['room']} [{sev}]"
                            body = (
                                f"Dear {f.get('name') or f.get('role','Faculty')},\n\n"
                                f"SHADOW has detected an energy alert.\n\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"  Room         : {a['room']}\n"
                                f"  Alert Type   : {a['type']}\n"
                                f"  Severity     : {sev}\n"
                                f"  Current Load : {a['val']}\n"
                                f"  Details      : {a['msg']}\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                                f"⏱ Countdown to auto-shutdown is active.\n"
                                f"Please open the SHADOW dashboard and choose:\n"
                                f"  ✅ Keep ON  — if the room is occupied and power is needed.\n"
                                f"  🔌 Allow Shutdown — to approve switching off unused appliances.\n\n"
                                f"Sent at: {datetime.now().strftime('%d %b %Y, %I:%M %p')}\n"
                                f"— SHADOW Campus Energy Digital Twin\n"
                                f"  http://localhost:8501"
                            )
                            try:
                                send_email(
                                    smtp_host=str(email_cfg["smtp_host"]),
                                    smtp_port=int(email_cfg.get("smtp_port", 587)),
                                    smtp_user=str(email_cfg["smtp_username"]),
                                    smtp_pass=str(email_cfg["smtp_password"]),
                                    sender=str(email_cfg.get("sender_email", email_cfg["smtp_username"])),
                                    recipient=f["email"],
                                    subject=subject,
                                    body=body,
                                )
                                results.append({"to": f["email"], "status": "sent"})
                            except Exception as ex:
                                results.append({"to": f["email"], "status": f"failed: {ex}"})

                        st.session_state.alert_state[akey]["notified"]      = True
                        st.session_state.alert_state[akey]["notif_results"] = results
                        st.rerun()

                # Show send results
                for res in astate["notif_results"]:
                    if "sent" in res["status"]:
                        st.markdown(f'<div class="sent-badge">✅ Sent → {res["to"]}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.error(f"❌ {res['to']}: {res['status']}")

            st.markdown('<hr style="margin:8px 0;border-color:#0d3a55">', unsafe_allow_html=True)

    if st.button("← Home", key="alb"): go_to("Home"); st.rerun()

# ═════════════════════════════════════════════════════════
# SETTINGS  — professional tabbed layout
# ═════════════════════════════════════════════════════════
elif st.session_state.page == "Settings":
    st.markdown('<div class="title">⚙ System Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">System status · Alert thresholds · Email configuration · Faculty management</div><br>',
                unsafe_allow_html=True)

    s1,s2,s3,s4 = st.tabs(["📊 System Status","⚡ Alert Thresholds","📧 Email / SMTP","👥 Faculty"])

    # ── Tab 1: System Status ──────────────────────────────
    with s1:
        st.markdown('<div class="set-section">'
                    '<div class="set-section-title">System Status</div>', unsafe_allow_html=True)
        rows_sys = [
            ("Dashboard",      "Running ✅"),
            ("Local time",     now.strftime("%d %b %Y, %I:%M %p")),
            ("Campus image",   "Available ✅" if MAP_IMAGE.exists() else "Not found ⚠"),
            ("Configured blocks", str(len(st.session_state.blocks))),
            ("Configured rooms",  str(len(st.session_state.rooms))),
            ("Footage records",   str(len(st.session_state.footage))),
            ("Camera creds set",  str(len(st.session_state.camera_creds))),
            ("Readings",       "Simulated demo data"),
            ("Live sensors",   "Not connected"),
            ("Auto notifications","Not connected"),
        ]
        for k,v in rows_sys:
            c1,c2=st.columns([1.5,2])
            c1.markdown(f'<span style="color:#7fa3bf;font-size:12px">{k}</span>',
                        unsafe_allow_html=True)
            c2.markdown(f'<span style="color:#d0e8ff;font-size:12px">{v}</span>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 2: Alert Thresholds ───────────────────────────
    with s2:
        st.markdown('<div class="set-section">'
                    '<div class="set-section-title">Alert Thresholds</div>', unsafe_allow_html=True)
        st.caption("These values control when rooms are marked Warning, High Consumption, or voltage alert.")
        sett = st.session_state.settings
        with st.form("thf"):
            tc1,tc2=st.columns(2)
            with tc1:
                wk=st.number_input("⚠ Power Warning (kW)",min_value=0.1,max_value=1000.0,
                                   value=float(sett["power_warning_kw"]),step=0.5)
                vm=st.number_input("🔽 Low Voltage (V)",  min_value=0.0, max_value=1000.0,
                                   value=float(sett["voltage_min"]),step=5.0)
            with tc2:
                hk=st.number_input("🔴 High Power (kW)",  min_value=0.1,max_value=1000.0,
                                   value=float(sett["power_high_kw"]),step=0.5)
                vx=st.number_input("🔼 High Voltage (V)", min_value=0.0, max_value=2000.0,
                                   value=float(sett["voltage_max"]),step=5.0)
            ts=st.form_submit_button("Save Thresholds",use_container_width=True)
        if ts:
            if wk>=hk: st.error("Warning threshold must be below High threshold.")
            elif vm>=vx: st.error("Low voltage must be below High voltage.")
            else:
                st.session_state.settings.update({"power_warning_kw":wk,"power_high_kw":hk,
                                                   "voltage_min":vm,"voltage_max":vx})
                save_data(); st.success("✅ Thresholds saved."); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 3: Email / SMTP ───────────────────────────────
    with s3:
        st.markdown('<div class="set-section">'
                    '<div class="set-section-title">Email Configuration & Test</div>',
                    unsafe_allow_html=True)
        st.caption("Configure SMTP once in .streamlit/secrets.toml. Test with a real recipient email.")

        with st.expander("📋 secrets.toml template — click to expand"):
            st.code('[email]\nsmtp_host     = "smtp.gmail.com"\nsmtp_port     = 587\n'
                    'smtp_username = "your@gmail.com"\nsmtp_password = "your-app-password"\n'
                    'sender_email  = "your@gmail.com"', language="toml")
            st.caption("For Gmail: enable 2FA then create an App Password at myaccount.google.com/apppasswords. "
                       "Never commit secrets.toml.")

        st.markdown("#### Send a test email")
        email_cfg = {}
        try: email_cfg = dict(st.secrets.get("email", {}))
        except: pass

        configured = all([email_cfg.get("smtp_host"), email_cfg.get("smtp_username"),
                          email_cfg.get("smtp_password")])

        if configured:
            st.success("✅ SMTP is configured in secrets.toml")
        else:
            st.warning("⚠ SMTP not yet configured. Fill secrets.toml first (template above).")

        with st.form("email_test_form"):
            et_to   = st.text_input("Recipient email", placeholder="test@example.com", key="et_to")
            et_subj = st.text_input("Subject", value="SHADOW — Test notification", key="et_subj")
            et_body = st.text_area("Message body",
                value=f"Hello,\n\nThis is a test message from SHADOW | Campus Energy Digital Twin.\n"
                      f"Sent at: {datetime.now().strftime('%d %b %Y %I:%M %p')}\n\n"
                      "This confirms the email path is working. Campus readings in this prototype are simulated.",
                height=130, key="et_body")
            et_send = st.form_submit_button("📤 Send Test Email", use_container_width=True,
                                            disabled=not configured)

        if et_send:
            if not et_to.strip() or "@" not in et_to:
                st.error("Enter a valid recipient email.")
            else:
                try:
                    send_email(
                        smtp_host=str(email_cfg["smtp_host"]),
                        smtp_port=int(email_cfg.get("smtp_port",587)),
                        smtp_user=str(email_cfg["smtp_username"]),
                        smtp_pass=str(email_cfg["smtp_password"]),
                        sender=str(email_cfg.get("sender_email", email_cfg["smtp_username"])),
                        recipient=et_to.strip(),
                        subject=et_subj,
                        body=et_body,
                    )
                    st.success(f"✅ Email sent successfully to **{et_to.strip()}** — check your inbox!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Failed to send: {e}")
                    st.info("Check: SMTP host/port, username, app password, and that 'Less secure app access' "
                            "or App Passwords are enabled on your Gmail account.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 4: Faculty ────────────────────────────────────
    with s4:
        st.markdown('<div class="set-section">'
                    '<div class="set-section-title">Room Faculty In-Charge</div>',
                    unsafe_allow_html=True)
        st.caption("Assign Advisor + 2 Mentors per room. Use ＋ to expand a room's faculty form.")

        rnames = [r["room"] for r in st.session_state.rooms]
        if not rnames:
            st.info("Add rooms first in Admin Settings.")
        else:
            for room in st.session_state.rooms:
                rid = room["room"]
                fac = room.get("faculty", [
                    {"role":"Advisor","name":"","email":""},
                    {"role":"Mentor 1","name":"","email":""},
                    {"role":"Mentor 2","name":"","email":""},
                ])
                # count filled
                filled = sum(1 for f in fac if f.get("name","").strip() or f.get("email","").strip())
                lbl_clr = "#35d69a" if filled==3 else ("#ffbd4a" if filled>0 else "#5d83a4")

                col_lbl, col_btn = st.columns([4,1])
                with col_lbl:
                    st.markdown(
                        f'<div style="padding:8px 0;font-size:12px;color:#c0d8ef">'
                        f'<b style="color:#e8f4ff">{rid}</b> &nbsp;'
                        f'<span style="color:{lbl_clr};font-size:10px">●</span> '
                        f'{filled}/3 faculty assigned</div>',
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    open_key = f"fopen_{rid}"
                    is_open  = st.session_state.faculty_open.get(rid, False)
                    btn_lbl  = "➖ Close" if is_open else "➕ Add / Edit"
                    if st.button(btn_lbl, key=f"ftoggle_{rid}", use_container_width=True):
                        st.session_state.faculty_open[rid] = not is_open
                        st.rerun()

                if st.session_state.faculty_open.get(rid, False):
                    def_fac = [
                        {"role":"Advisor","name":"","email":""},
                        {"role":"Mentor 1","name":"","email":""},
                        {"role":"Mentor 2","name":"","email":""},
                    ]
                    if not isinstance(fac, list): fac = def_fac
                    fac = [dict(f) if isinstance(f,dict) else dict(def_fac[i])
                           for i,f in enumerate(fac[:3])]
                    while len(fac)<3: fac.append(def_fac[len(fac)].copy())

                    with st.form(f"fform_{rid}"):
                        fv=[]
                        fc1,fc2,fc3=st.columns(3)
                        for ci,(col,role) in enumerate([(fc1,"Advisor"),(fc2,"Mentor 1"),(fc3,"Mentor 2")]):
                            with col:
                                st.markdown(f"**{role}**")
                                fn=st.text_input("Name",  value=fac[ci].get("name",""),  key=f"fn_{rid}_{ci}")
                                fe=st.text_input("Email", value=fac[ci].get("email",""), key=f"fe_{rid}_{ci}")
                                fv.append({"role":role,"name":fn.strip(),"email":fe.strip()})
                        fsave=st.form_submit_button("Save Faculty",use_container_width=True)

                    if fsave:
                        bad=[f["role"] for f in fv if f["email"] and ("@" not in f["email"] or "." not in f["email"].split("@")[-1])]
                        if bad: st.error("Invalid email for: "+", ".join(bad))
                        else:
                            room["faculty"]=fv; save_data()
                            st.success(f"Faculty saved for {rid}.")
                            st.session_state.faculty_open[rid]=False
                            st.rerun()

                    # Test email to this room's faculty
                    efac=[f for f in fac if f.get("email","").strip()]
                    if efac:
                        with st.form(f"ftest_{rid}"):
                            ft_rec=st.selectbox("Send test to",efac,
                                format_func=lambda f:f"{f.get('role','?')} — {f.get('name') or f.get('email')} ({f.get('email')})",
                                key=f"ftrec_{rid}")
                            ftsend=st.form_submit_button("📤 Send test email",use_container_width=True)
                        if ftsend:
                            try:
                                ecfg={}
                                try: ecfg=dict(st.secrets.get("email",{}))
                                except: pass
                                if not all([ecfg.get("smtp_host"),ecfg.get("smtp_username"),ecfg.get("smtp_password")]):
                                    st.error("SMTP not configured. Set up email in Settings → Email/SMTP tab first.")
                                else:
                                    send_email(
                                        smtp_host=str(ecfg["smtp_host"]),
                                        smtp_port=int(ecfg.get("smtp_port",587)),
                                        smtp_user=str(ecfg["smtp_username"]),
                                        smtp_pass=str(ecfg["smtp_password"]),
                                        sender=str(ecfg.get("sender_email",ecfg["smtp_username"])),
                                        recipient=ft_rec["email"],
                                        subject=f"SHADOW notification — {rid}",
                                        body=(f"Hello {ft_rec.get('name') or ft_rec['role']},\n\n"
                                              f"Test email from SHADOW for room {rid}.\n"
                                              f"Sent: {datetime.now().strftime('%d %b %Y %I:%M %p')}\n\n"
                                              "Readings in this prototype are simulated.")
                                    )
                                    st.success(f"✅ Sent to {ft_rec['email']}")
                            except Exception as ex:
                                st.error(f"❌ {ex}")

                    st.markdown('<hr style="margin:6px 0;border-color:#0d3a55">', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("← Back to Home", key="stg_back"): go_to("Home"); st.rerun()

# ═════════════════════════════════════════════════════════
# PROFILE
# ═════════════════════════════════════════════════════════
elif st.session_state.page == "Profile":
    st.markdown('<div class="title">Administrator Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Local prototype administrator details.</div><br>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("### SHADOW Administrator")
        st.write("Role: Campus dashboard administrator")
        st.write("Permissions: Manage blocks, rooms and thresholds")
        st.write("Storage: shadow_data.json (local)")
        st.caption("Demo profile — no real authentication system.")
    if st.button("Open Admin Settings",key="p2as"): go_to("Admin Settings"); st.rerun()
    if st.button("← Home",key="p2h"):               go_to("Home");          st.rerun()

# ═════════════════════════════════════════════════════════
# ADMIN SETTINGS
# ═════════════════════════════════════════════════════════
elif st.session_state.page == "Admin Settings":
    st.markdown('<div class="title">Admin Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Manage campus blocks, rooms and demo readings.</div><br>',
                unsafe_allow_html=True)
    st.info("Prototype management screen — no login protection. Do not treat as a secure portal.")

    blks = st.session_state.blocks
    tab_ab,tab_rb,tab_db=st.tabs(["Add Block","Rename Block","Delete Block"])

    with tab_ab:
        with st.form("abf"):
            nb=st.text_input("New block name",placeholder="D")
            abs2=st.form_submit_button("Add Block")
        if abs2:
            nb=nb.strip()
            if not nb: st.error("Enter block name.")
            elif nb in blks: st.error("Already exists.")
            else:
                st.session_state.blocks.append(nb)
                st.session_state.blocks=sorted(set(st.session_state.blocks))
                save_data(); st.success(f"Block {nb} added."); st.rerun()

    with tab_rb:
        if not blks: st.info("No blocks to rename.")
        else:
            with st.form("rbf"):
                ob=st.selectbox("Block to rename",blks)
                rnb=st.text_input("New name")
                rbs=st.form_submit_button("Rename")
            if rbs:
                rnb=rnb.strip()
                if not rnb: st.error("Enter new name.")
                elif rnb in blks and rnb!=ob: st.error("Already exists.")
                else:
                    st.session_state.blocks=[rnb if b==ob else b for b in st.session_state.blocks]
                    for r in st.session_state.rooms:
                        if r["block"]==ob:
                            r["block"]=rnb
                            suf=r["room"].split("-",1)
                            if len(suf)==2 and suf[0]==ob: r["room"]=f"{rnb}-{suf[1]}"
                    save_data(); st.success(f"Renamed {ob} → {rnb}."); st.rerun()

    with tab_db:
        if not blks: st.info("No blocks to delete.")
        else:
            with st.form("dbf"):
                bd=st.selectbox("Block to delete",blks)
                bdr=st.checkbox("Also delete all rooms in this block")
                dbs=st.form_submit_button("Delete Block")
            if dbs:
                inb=[r for r in st.session_state.rooms if r["block"]==bd]
                if inb and not bdr: st.error("Block has rooms — tick 'delete rooms too' first.")
                else:
                    st.session_state.blocks=[b for b in st.session_state.blocks if b!=bd]
                    st.session_state.rooms=[r for r in st.session_state.rooms if r["block"]!=bd]
                    save_data(); st.success(f"Block {bd} deleted."); st.rerun()

    st.markdown("---")
    st.markdown("### Add Room")
    with st.form("arf"):
        nr=st.text_input("Room ID",placeholder="A-105")
        rb=st.selectbox("Block",blks if blks else ["Create a block first"])
        rc=st.number_input("Capacity",min_value=0,max_value=10000,value=40,step=1)
        rp=st.number_input("Power (kW)",min_value=0.0,max_value=10000.0,value=1.5,step=0.1)
        rv=st.number_input("Voltage (V)",min_value=0.0,max_value=2000.0,value=230.0,step=1.0)
        rs=st.selectbox("Status",["Normal","Warning","High Consumption","Offline"])
        ars=st.form_submit_button("Add Room",disabled=not bool(blks))
    if ars:
        nr=nr.strip()
        if not nr: st.error("Enter room ID.")
        elif nr.lower() in [r["room"].lower() for r in st.session_state.rooms]: st.error("ID exists.")
        elif rb not in st.session_state.blocks: st.error("Invalid block.")
        else:
            st.session_state.rooms.append({"room":nr,"block":rb,"status":rs,
                "power":float(rp),"voltage":float(rv),"capacity":int(rc)})
            save_data(); st.success(f"Room {nr} added."); st.rerun()

    st.markdown("---")
    st.markdown("### Edit / Delete Room")
    if not st.session_state.rooms: st.info("No rooms yet.")
    else:
        sel_id=st.selectbox("Select room",[r["room"] for r in st.session_state.rooms],key="ersel")
        sel_r=next(r for r in st.session_state.rooms if r["room"]==sel_id)
        with st.form("erf"):
            eri=st.text_input("Room ID",value=sel_r["room"])
            erb=st.selectbox("Block",st.session_state.blocks,
                index=(st.session_state.blocks.index(sel_r["block"])
                       if sel_r["block"] in st.session_state.blocks else 0))
            erc=st.number_input("Capacity",min_value=0,max_value=10000,value=int(sel_r["capacity"]),step=1)
            erp=st.number_input("Power (kW)",min_value=0.0,max_value=10000.0,value=float(sel_r["power"]),step=0.1)
            erv=st.number_input("Voltage (V)",min_value=0.0,max_value=2000.0,value=float(sel_r["voltage"]),step=1.0)
            sop=["Normal","Warning","High Consumption","Offline"]
            ers=st.selectbox("Status",sop,index=(sop.index(sel_r["status"]) if sel_r["status"] in sop else 0))
            ers2=st.form_submit_button("Save Changes")
        if ers2:
            eri=eri.strip()
            if not eri: st.error("Room ID empty.")
            elif any(r["room"].lower()==eri.lower() and r["room"]!=sel_id for r in st.session_state.rooms):
                st.error("ID taken.")
            else:
                sel_r.update({"room":eri,"block":erb,"capacity":int(erc),"power":float(erp),"voltage":float(erv),"status":ers})
                save_data(); st.success("Saved."); st.rerun()
        with st.form("drf"):
            st.caption(f"Delete: {sel_id}")
            conf=st.checkbox("I confirm deletion")
            dels=st.form_submit_button("Delete Room")
        if dels:
            if not conf: st.error("Tick confirmation.")
            else:
                st.session_state.rooms=[r for r in st.session_state.rooms if r["room"]!=sel_id]
                save_data(); st.success(f"{sel_id} deleted."); st.rerun()

    st.markdown("---")
    st.markdown("### Current Configuration")
    st.write("Blocks:", ", ".join(st.session_state.blocks) or "None")
    st.dataframe(rooms_df(), use_container_width=True, hide_index=True)
    st.download_button("Export JSON",
        json.dumps({"blocks":st.session_state.blocks,"rooms":st.session_state.rooms,
                    "settings":st.session_state.settings},indent=2),
        "shadow_config.json","application/json")
    if st.button("← Home",key="as_back"): go_to("Home"); st.rerun()
