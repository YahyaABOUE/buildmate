import gradio as gr
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.orchestrator import run_pipeline
from fastapi.middleware.cors import CORSMiddleware


def run_buildmate(device_type, use_case, budget_tier, budget_max, preferences_json, free_text):
    use_case_map = {
        "Gaming": "gaming", "Video Editing": "video_editing",
        "Machine Learning": "machine_learning", "Office": "office",
        "3D Rendering": "3d_rendering", "Music Production": "music_production"
    }
    device_map = {"Desktop": "desktop", "Laptop": "laptop"}
    tier_map   = {"Low": "low", "Mid": "mid", "High": "high"}
    try:
        prefs = json.loads(preferences_json) if preferences_json else []
    except:
        prefs = []

    device_override = device_map.get(device_type) if device_type and device_type != 'null' else None
    use_case_override = use_case_map.get(use_case) if use_case and use_case != 'null' else None
    tier_override = tier_map.get(budget_tier) if budget_tier and budget_tier != 'null' else None

    result = run_pipeline(
        free_text=free_text if free_text and free_text.strip() else "",
        device_type_override=device_override,
        budget_max=float(budget_max or 1200),
        use_case_override=use_case_override,
        budget_tier_override=tier_override,
        preferences=prefs,
        skip_hitl=True
    )
    return json.dumps(result)

demo = gr.Interface(
    fn=run_buildmate,
    inputs=[
        gr.Textbox(label="device_type"),
        gr.Textbox(label="use_case"),
        gr.Textbox(label="budget_tier"),
        gr.Number(label="budget_max", value=1200),
        gr.Textbox(label="preferences_json"),
        gr.Textbox(label="free_text"),
    ],
    outputs=gr.Textbox(label="result"),
    title="BuildMate API",
)

if __name__ == "__main__":
    import threading
    import http.server
    import webbrowser

    # Serve the HTML file on port 3000
    html_dir = os.path.dirname(__file__)
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=html_dir, **kwargs)
        def log_message(self, format, *args):
            pass  # suppress logs

    server = http.server.HTTPServer(("", 3000), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    print("\n✅ BuildMate UI: http://localhost:3000/index.html")
    print("✅ Gradio API:   http://localhost:7860\n")

    webbrowser.open("http://localhost:3000/index.html")

    @demo.app.middleware("http")
    async def add_cors(request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    print(demo.get_api_info())

    demo.launch(server_port=7860, quiet=True, share=False)
