#!/usr/bin/env python3
"""
Simple CORS-enabled HTTP server for the transcript labeling interface.
"""

import http.server
import socketserver
import os
import sys
from urllib.parse import unquote

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        # Handle requests for transcript files from parent directory
        if self.path.startswith('/transcripts/'):
            try:
                # Remove leading slash and construct path
                relative_path = self.path[1:]  # Remove leading '/'
                
                # Go to parent directory to access transcripts
                current_dir = os.getcwd()
                parent_dir = os.path.dirname(current_dir)
                full_path = os.path.join(parent_dir, relative_path)
                
                print(f"Requesting: {self.path}")
                print(f"Looking for: {full_path}")
                
                if os.path.exists(full_path):
                    with open(full_path, 'rb') as f:
                        content = f.read()
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Content-length', str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return
                else:
                    print(f"File not found: {full_path}")
                    self.send_error(404, "File not found")
                    return
            except Exception as e:
                print(f"Error serving transcript: {e}")
                self.send_error(500, f"Server error: {e}")
                return
        
        # Default handling for files in current directory
        super().do_GET()

if __name__ == "__main__":
    PORT = 9999
    
    print(f"Starting server on port {PORT}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Open your browser to: http://localhost:{PORT}")
    print("Press Ctrl+C to stop")
    
    try:
        with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)