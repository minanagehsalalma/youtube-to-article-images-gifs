import re
from html import escape


BASIC_CSS = """
:root {
  color-scheme: light;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  color: #1f2937;
  background: #ffffff;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
  line-height: 1.56;
}
.page {
  padding: 14px 10px 36px;
}
.article {
  max-width: 980px;
  margin: 0 auto;
}
h1, h2, h3 {
  line-height: 1.2;
  margin: 1.1em 0 0.45em;
}
h1 {
  margin-top: 0;
  font-size: 2rem;
}
h2 {
  font-size: 1.48rem;
}
h3 {
  font-size: 1.2rem;
}
p {
  margin: 0.7em 0 1em;
}
img {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
  margin: 12px 0 20px;
  border-radius: 8px;
}
hr {
  border: none;
  border-top: 1px solid #d6dbe3;
  margin: 24px 0;
}
"""


ARTICLE_CSS = """
:root {
  --bg: #f5f7fb;
  --paper: #ffffff;
  --ink: #1b2431;
  --muted: #495768;
  --rule: #e3e8ef;
  --accent: #0f6bbd;
  --accent-soft: #dbeafe;
  --edge: #e2e8f0;
  --shadow: 0 8px 28px rgba(15, 23, 42, 0.09);
  --content-max: 1180px;
  --text-max: 74ch;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  min-height: 100vh;
  color: var(--ink);
  background: linear-gradient(180deg, #f8fafd 0%, var(--bg) 100%);
  font-family: "Charter", "Georgia", "Cambria", serif;
  line-height: 1.7;
  text-rendering: optimizeLegibility;
}
.page {
  padding: clamp(16px, 2vw, 34px) clamp(12px, 1.7vw, 30px) 52px;
}
.article {
  width: 100%;
  max-width: var(--content-max);
  margin: 0 auto;
  background: var(--paper);
  border: 1px solid var(--edge);
  border-radius: 14px;
  box-shadow: var(--shadow);
  padding: clamp(20px, 2.8vw, 40px) clamp(18px, 3vw, 44px) clamp(28px, 3.4vw, 48px);
}
h1, h2, h3 {
  margin: 1em 0 0.45em;
  color: #101828;
  line-height: 1.22;
  letter-spacing: 0.01em;
  font-family: "Avenir Next", "Trebuchet MS", "Segoe UI", sans-serif;
}
h1 {
  margin-top: 0;
  font-size: clamp(2rem, 2.6vw, 2.85rem);
  padding-bottom: 0.3em;
  border-bottom: 2px solid var(--accent-soft);
}
h2 {
  font-size: clamp(1.35rem, 1.8vw, 1.9rem);
  padding-top: 0.4em;
  border-top: 1px solid var(--rule);
}
h3 {
  font-size: clamp(1.08rem, 1.3vw, 1.3rem);
}
p {
  max-width: var(--text-max);
  margin: 0.65em 0 1.02em;
  color: var(--muted);
  font-size: clamp(1rem, 0.96rem + 0.2vw, 1.1rem);
}
img {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
  margin: 14px 0 24px;
  border-radius: 12px;
  border: 1px solid #d7deea;
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
  background: #ffffff;
}
hr {
  border: none;
  border-top: 1px solid var(--rule);
  margin: 28px 0;
}
@media (max-width: 760px) {
  .article {
    border-radius: 10px;
    padding: 18px 14px 26px;
  }
  p {
    max-width: none;
    line-height: 1.64;
  }
}
@media (min-width: 1700px) {
  :root {
    --content-max: 1240px;
  }
}
@media (prefers-reduced-motion: no-preference) {
  .article {
    animation: rise-in 420ms ease-out both;
  }
  @keyframes rise-in {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }
}
"""


def html_escape(s: str) -> str:
    return escape(s, quote=False)


def _render_md_lines(md_text: str) -> str:
    # Tiny markdown -> HTML: headings, paragraphs, images, horizontal rules.
    # (No extra dependencies)
    lines = md_text.splitlines()
    html = []
    for line in lines:
        if line.strip() == "":
            continue
        if line.strip() == "---":
            html.append("<hr/>")
            continue
        if line.startswith("### "):
            html.append(f"<h3>{html_escape(line[4:])}</h3>")
            continue
        if line.startswith("## "):
            html.append(f"<h2>{html_escape(line[3:])}</h2>")
            continue
        if line.startswith("# "):
            html.append(f"<h1>{html_escape(line[2:])}</h1>")
            continue
        m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
        if m:
            alt = html_escape(m.group(1))
            src = html_escape(m.group(2))
            html.append(f'<img src="{src}" alt="{alt}"/>')
            continue
        html.append(f"<p>{html_escape(line)}</p>")
    return "\n".join(html)


def md_to_html_basic(md_text: str, style: str = "article") -> str:
    style_key = (style or "article").strip().lower()
    if style_key not in {"basic", "article"}:
        style_key = "article"
    css = BASIC_CSS if style_key == "basic" else ARTICLE_CSS
    body = _render_md_lines(md_text)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Article</title>
  <style>
{css}
  </style>
</head>
<body>
  <main class="page">
    <article class="article">
{body}
    </article>
  </main>
</body>
</html>
"""
