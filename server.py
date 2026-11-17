"""
Standalone HTTP Server for Mossline Applied AI Landing Page
Course: Applied AI (ISE 402) - Pentecost University
Lecturer: Mr. Harry Attieku-Boateng
Student: Emmanuel Asante (PU092301 / PUIT23210016)
"""

import http.server
import socketserver
import webbrowser
import os

PORT = 8501
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def run():
    os.chdir(DIRECTORY)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🌿 Mossline AI Classification Deck running at http://localhost:{PORT}")
        print(f"👉 Opening your browser automatically...")
        webbrowser.open(f"http://localhost:{PORT}/index.html")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    run()
