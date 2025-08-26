#!/usr/bin/env python3
"""
Simple HTTP server to serve the transcript labeling interface.
This handles CORS issues when loading local files.
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
        # Handle requests for transcript files
        if self.path.startswith('/transcripts/'):
            # Remove the leading slash and construct the full path
            relative_path = unquote(self.path[1:])  # Remove leading '/'
            
            # Go up one directory to access the transcripts folder
            parent_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(parent_dir)
            full_path = os.path.join(root_dir, relative_path)
            
            if os.path.exists(full_path) and full_path.endswith('.json'):
                try:
                    with open(full_path, 'rb') as f:
                        content = f.read()
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Content-length', str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return
                except Exception as e:
                    print(f"Error serving {full_path}: {e}")
                    self.send_error(500, f"Internal server error: {e}")
                    return
            else:
                self.send_error(404, "Transcript file not found")
                return
        
        # Default handling for other files
        super().do_GET()

def main():
    PORT = 9999
    
    # Change to the data_label directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"Starting server in directory: {script_dir}")
    print(f"Server will be available at: http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server")
    
    try:
        with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
            print(f"Serving at port {PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()