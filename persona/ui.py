"""Gradio page: chat with Troy's AI twin, in the style of Troy's portfolio website."""

import hashlib
import inspect
import time
from collections.abc import Callable, Iterator

import gradio as gr

from persona import __version__
from persona.models import (
    APP_TITLE,
    CONTACT_EMAIL,
    LIMIT_MESSAGE,
    MAX_QUESTION_CHARS,
    AnswerUpdate,
    VisitorLimiter,
    validate_question,
)

AnswerFn = Callable[[str, list[dict]], Iterator[AnswerUpdate]]

EXAMPLES = [
    "What was your role on the United Nations Waterfront Esplanade?",
    "Are you open to new roles, and when can you start?",
    "Do you need visa sponsorship to work in the US?",
    "What AI and LLM work have you done?",
    "Tell me about the IFC Viewer you built.",
    "Why did you move from civil engineering into software?",
]
DISCLAIMER = (
    "AI assistant — answers come from Troy's knowledge base. Please verify important details with Troy."
)
UNAVAILABLE_MESSAGE = (
    "### The assistant is taking a short break\n"
    f"It couldn't start right now. Please try again later, or email Troy directly at {CONTACT_EMAIL}."
)
CONCURRENCY_LIMIT = 4  # parallel answers per event (N6); further visitors wait in Gradio's queue

# Troy's website (troyyu527.github.io/ReactWebsite, measured 2026-10-17): light blue-grey page, navy header and
# primary buttons, blue accent, Inter. The dark-mode values are the same, so the page looks like the site everywhere.
PAGE, NAVY, NAVY_HOVER, BLUE, HEADING, TEXT, MUTED, BORDER = (
    "#eef2f7",
    "#0d2540",
    "#16365c",
    "#2f6fed",
    "#0f1d2e",
    "#475569",
    "#93a4bd",
    "#cbd5e1",
)
_COLORS = {
    "body_background_fill": PAGE,
    "body_text_color": HEADING,
    "body_text_color_subdued": TEXT,
    "background_fill_primary": "white",
    "background_fill_secondary": PAGE,
    "block_background_fill": "white",
    "block_border_color": "#dbe3ee",
    "border_color_primary": "#dbe3ee",
    "input_background_fill": "white",
    "input_border_color": BLUE,
    "input_border_color_focus": NAVY,
    "button_primary_background_fill": NAVY,
    "button_primary_background_fill_hover": NAVY_HOVER,
    "button_primary_text_color": "white",
    "button_primary_border_color": NAVY,
    "button_secondary_background_fill": "white",
    "button_secondary_background_fill_hover": "#f5f8fc",
    "button_secondary_border_color": BORDER,
    "button_secondary_text_color": NAVY,
    "link_text_color": BLUE,
    "color_accent": BLUE,
    "color_accent_soft": "#e3ecfd",
}
_THEME_VARIABLES = inspect.signature(gr.themes.Base.set).parameters
THEME = gr.themes.Base(font=[gr.themes.GoogleFont("Inter"), "Roboto", "Arial", "sans-serif"]).set(
    **_COLORS,
    **{f"{k}_dark": v for k, v in _COLORS.items() if f"{k}_dark" in _THEME_VARIABLES},
    input_border_width="2px",
    input_radius="10px",
    input_shadow_focus=f"0 0 0 3px {BLUE}33",
    block_radius="14px",
    button_large_radius="10px",
    button_medium_radius="10px",
    button_small_radius="10px",
)
CSS = f"""
footer {{ display: none !important; }}
.gradio-container {{ max-width: 880px !important; margin: 0 auto !important; }}
#header {{ background: {NAVY}; border-radius: 14px; padding: 18px 22px; display: flex; align-items: center; gap: 14px; }}
#header .badge {{ width: 44px; height: 44px; border-radius: 10px; display: grid; place-items: center; color: white;
  font-weight: 800; background: linear-gradient(135deg, {BLUE}, #1d4fc4); }}
#header .name {{ color: white; font-size: 1.25rem; font-weight: 800; line-height: 1.2; }}
#header .role {{ color: {MUTED}; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }}
#model-status {{ color: {TEXT}; font-size: 0.85rem; padding-left: 2px; }}
#model-status code {{ color: {NAVY}; background: #e3ecfd; }}
#question textarea {{ font-size: 1rem !important; }}
#send {{ font-weight: 700 !important; min-height: 48px; }}
.example-row button {{ font-weight: 600 !important; font-size: 0.85rem !important; }}
#disclaimer {{ color: {MUTED}; font-size: 0.8rem; text-align: center; }}
"""
HEADER = """<div id="header"><div class="badge">TY</div>
<div><div class="name">Troy Yu</div><div class="role">AI assistant · ask about my career</div></div></div>"""


def visitor_id(request: gr.Request | None) -> str:
    """A short hash of the visitor's IP address (the first X-Forwarded-For entry behind the Space's proxy)."""
    if request is None:
        return "local"
    forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    ip = forwarded or (request.client.host if request.client else "unknown")
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def format_status(model_name: str, update: AnswerUpdate | None = None) -> str:
    """The line at the top left: which model answers, and how long the last answer took."""
    if update is not None and update.failed:
        return f"Model: `{model_name}` · ⚠️ no model available right now"
    if update is not None and update.done and update.model:
        return f"Model: `{update.model}` · answered in {update.seconds:.1f} s"
    if update is not None and not update.done:
        return f"Model: `{model_name}` · thinking…"
    return f"Model: `{model_name}`"


def make_respond(answer: AnswerFn, model_name: str, limiter: VisitorLimiter):
    """The chat handler: checks the visitor's limits, streams one answer into the chat, then counts its tokens."""

    def respond(message: str, history: list[dict], request: gr.Request = None):
        if not message.strip():
            yield "", history, gr.skip()
            return
        prior = list(history or [])
        chat = prior + [{"role": "user", "content": message}]
        visitor = visitor_id(request)
        counted = validate_question(message) is None  # a rejected question costs nothing
        if counted and not limiter.allow(visitor, message, time.time()):
            yield "", chat + [{"role": "assistant", "content": LIMIT_MESSAGE}], format_status(model_name)
            return
        yield (
            "",
            chat + [{"role": "assistant", "content": ""}],
            format_status(model_name, AnswerUpdate(text="")),
        )
        update = None
        for update in answer(message, prior):
            yield (
                "",
                chat + [{"role": "assistant", "content": update.text}],
                format_status(model_name, update),
            )
        if counted and update is not None and not update.failed:  # the busy message isn't charged
            limiter.add_answer(visitor, update.text, time.time())

    return respond


def create(answer: AnswerFn, model_name: str, limiter: VisitorLimiter | None = None) -> gr.Blocks:
    """Build the page; `answer` is rag.Rag.answer (or a fake in tests)."""
    respond = make_respond(answer, model_name, limiter or VisitorLimiter())
    with gr.Blocks(title=APP_TITLE, theme=THEME, css=CSS, analytics_enabled=False) as demo:
        gr.HTML(HEADER)
        status = gr.Markdown(format_status(model_name), elem_id="model-status")
        chatbot = gr.Chatbot(
            type="messages",
            height=480,
            show_label=False,
            show_copy_button=True,
            allow_tags=False,
            placeholder="**Hi, I'm Troy's AI twin.** Ask me about my projects, skills, experience or availability.",
        )
        with gr.Row(equal_height=True):
            box = gr.Textbox(
                placeholder="Ask about Troy's career…",
                show_label=False,
                max_length=MAX_QUESTION_CHARS,
                autofocus=True,
                scale=5,
                min_width=260,  # on a phone, Send wraps below instead of squeezing the box
                elem_id="question",
            )
            send = gr.Button("Send", variant="primary", scale=1, min_width=96, elem_id="send")
        with gr.Row(elem_classes="example-row"):
            example_buttons = [gr.Button(q, size="sm", variant="secondary") for q in EXAMPLES]
        clear = gr.Button("Clear chat", size="sm", variant="secondary")
        gr.Markdown(f"{DISCLAIMER} · v{__version__}", elem_id="disclaimer")

        outputs = [box, chatbot, status]
        box.submit(respond, [box, chatbot], outputs, concurrency_limit=CONCURRENCY_LIMIT)
        send.click(respond, [box, chatbot], outputs, concurrency_limit=CONCURRENCY_LIMIT)
        for button, question in zip(example_buttons, EXAMPLES):
            button.click(respond, [gr.State(question), chatbot], outputs, concurrency_limit=CONCURRENCY_LIMIT)
        clear.click(lambda: ("", [], format_status(model_name)), None, outputs, queue=False)
    demo.queue(default_concurrency_limit=CONCURRENCY_LIMIT)
    return demo


def create_unavailable() -> gr.Blocks:
    """The page shown when the app can't start (knowledge base or LLM settings missing): same look, no chat."""
    with gr.Blocks(title=APP_TITLE, theme=THEME, css=CSS, analytics_enabled=False) as demo:
        gr.HTML(HEADER)
        gr.Markdown(UNAVAILABLE_MESSAGE, elem_id="unavailable")
    return demo
