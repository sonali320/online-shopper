"""
Interactive Web Application for Online Shoppers Purchasing Intention.
Supports both Flask (if installed) and built-in Python http.server (zero dependencies).
"""

import sys
import json
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import WEB_HOST, WEB_PORT, PDF_BENCHMARKS, REPORTS_DIR, MODELS_DIR
from src.predict import predict_single

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

def get_benchmark_data():
    json_path = REPORTS_DIR / "pdf_benchmark_comparison.json"
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Fallback to config benchmarks
    return {
        k: {
            "metrics": {
                "accuracy": v["accuracy"],
                "sensitivity": v["sensitivity"],
                "specificity": v["specificity"],
                "true_positives": v["true_positives"],
                "true_negatives": v["true_negatives"],
                "false_positives": v["false_positives"],
                "false_negatives": v["false_negatives"],
                "cv_mean_accuracy": v["cv_mean_accuracy"],
                "cv_std_accuracy": v["cv_std_accuracy"]
            },
            "model_name": v["model_name"]
        }
        for k, v in PDF_BENCHMARKS.items()
    }

class ShopperAppRequestHandler(BaseHTTPRequestHandler):
    def _send_response(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path == "/" or path == "/index.html":
            html_file = TEMPLATES_DIR / "index.html"
            if html_file.exists():
                content = html_file.read_bytes()
                self._send_response(200, "text/html; charset=utf-8", content)
            else:
                self._send_response(404, "text/plain", b"index.html not found")
        elif path == "/api/benchmark":
            data = get_benchmark_data()
            body = json.dumps(data).encode("utf-8")
            self._send_response(200, "application/json", body)
        elif path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            file_path = STATIC_DIR / rel_path
            if file_path.exists() and file_path.is_file():
                content_type = "text/plain"
                if path.endswith(".css"):
                    content_type = "text/css"
                elif path.endswith(".js"):
                    content_type = "application/javascript"
                elif path.endswith(".svg"):
                    content_type = "image/svg+xml"
                elif path.endswith(".png"):
                    content_type = "image/png"
                self._send_response(200, content_type, file_path.read_bytes())
            else:
                self._send_response(404, "text/plain", b"Static asset not found")
        else:
            self._send_response(404, "text/plain", b"Not Found")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(raw_body)
                model_name = data.get("model_name", "Random Forest")
                session_features = data.get("features", data)

                # Ensure numerical features are converted to correct types
                clean_features = {
                    "Administrative": int(session_features.get("Administrative", 0)),
                    "Administrative_Duration": float(session_features.get("Administrative_Duration", 0.0)),
                    "Informational": int(session_features.get("Informational", 0)),
                    "Informational_Duration": float(session_features.get("Informational_Duration", 0.0)),
                    "ProductRelated": int(session_features.get("ProductRelated", 1)),
                    "ProductRelated_Duration": float(session_features.get("ProductRelated_Duration", 0.0)),
                    "BounceRates": float(session_features.get("BounceRates", 0.0)),
                    "ExitRates": float(session_features.get("ExitRates", 0.0)),
                    "PageValues": float(session_features.get("PageValues", 0.0)),
                    "SpecialDay": float(session_features.get("SpecialDay", 0.0)),
                    "Month": str(session_features.get("Month", "Nov")),
                    "OperatingSystems": int(session_features.get("OperatingSystems", 2)),
                    "Browser": int(session_features.get("Browser", 2)),
                    "Region": int(session_features.get("Region", 1)),
                    "TrafficType": int(session_features.get("TrafficType", 2)),
                    "VisitorType": str(session_features.get("VisitorType", "Returning_Visitor")),
                    "Weekend": bool(session_features.get("Weekend", False))
                }

                result = predict_single(clean_features, model_name=model_name)
                resp_bytes = json.dumps({"status": "success", "result": result}).encode("utf-8")
                self._send_response(200, "application/json", resp_bytes)
            except Exception as e:
                err_bytes = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
                self._send_response(500, "application/json", err_bytes)
        else:
            self._send_response(404, "text/plain", b"Not Found")

def start_server(host=WEB_HOST, port=WEB_PORT):
    # Ensure models are trained first
    rf_model_file = MODELS_DIR / "random_forest_pipeline.joblib"
    if not rf_model_file.exists():
        print("Trained models not detected. Auto-training models before launching server...")
        from src.train import train_and_evaluate_all
        train_and_evaluate_all(save_models=True, generate_report=True)

    server = HTTPServer((host, port), ShopperAppRequestHandler)
    print(f"\n=======================================================")
    print(f"  Online Shoppers Intention Web Dashboard Running!")
    print(f"  Open in your browser: http://{host}:{port}")
    print(f"  Press Ctrl+C to terminate the server.")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server gracefully...")
        server.server_close()

if __name__ == "__main__":
    start_server()
