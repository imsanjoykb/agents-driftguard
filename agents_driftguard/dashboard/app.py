"""Zero-dependency HTTP server for the DriftGuard dashboard."""

import json
import logging
import os
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from agents_driftguard.dashboard.tracker import tracker

logger = logging.getLogger("agents_driftguard.dashboard")


class DashboardHandler(SimpleHTTPRequestHandler):
    """Handles API requests and serves the static HTML."""

    def __init__(self, *args, **kwargs):
        # Serve static files from the 'static' directory next to this file
        directory = os.path.join(os.path.dirname(__file__), "static")
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/"):
            self._handle_api(path)
            return

        # Fallback to static file serving
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def _handle_api(self, path: str) -> None:
        """Simple routing for /api/* endpoints."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        response = {}

        if path == "/api/runs":
            response = {"runs": tracker.get_all_runs()}
        elif path.startswith("/api/runs/"):
            parts = path.split("/")
            if len(parts) >= 4:
                run_id = parts[3]
                run = tracker.get_run(run_id)
                if run:
                    if len(parts) == 4:
                        # /api/runs/<id>
                        response = {
                            "info": run.to_dict(),
                            "states": run.states,
                            "rollbacks": run.rollbacks,
                            "graph": run.graph_mermaid,
                        }
                    elif parts[4] == "graph":
                        response = {"graph": run.graph_mermaid}
                    elif parts[4] == "states":
                        response = {"states": run.states}
                else:
                    response = {"error": "Run not found"}

        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, format, *args):
        # Suppress standard logging to keep terminal clean
        pass


def run_server(port: int = 8050):
    server_address = ("", port)
    httpd = HTTPServer(server_address, DashboardHandler)
    logger.info(f"Dashboard available at http://localhost:{port}")
    httpd.serve_forever()


def start_dashboard_server(port: int = 8050, background: bool = True) -> threading.Thread | None:
    """Start the dashboard HTTP server.
    
    Args:
        port: Port to listen on.
        background: If True, runs in a daemon thread.
    """
    if background:
        thread = threading.Thread(target=run_server, args=(port,), daemon=True)
        thread.start()
        return thread
    else:
        run_server(port)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_dashboard_server(background=False)
