"""
SHADOW - Local SMTP Test Receiver
Listens on localhost:1025 and prints incoming alert emails in full formatting.
Zero credentials required. Works completely offline.
Run with: python local_email_receiver.py
"""

import socket
import sys
from datetime import datetime

HOST = "127.0.0.1"
PORT = 1025

def run_local_receiver():
    # Ensure UTF-8 output on Windows terminal
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 64)
    print("⚡ SHADOW LOCAL SMTP TEST RECEIVER")
    print(f"Listening on: {HOST}:{PORT}")
    print("Status: READY to receive SHADOW alert emails offline without credentials.")
    print("Press Ctrl+C to terminate.")
    print("=" * 64 + "\n")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        srv.bind((HOST, PORT))
        srv.listen(5)
    except Exception as ex:
        print(f"Error binding to {HOST}:{PORT}: {ex}")
        return

    while True:
        try:
            client, addr = srv.accept()
            client.sendall(b"220 localhost SHADOW-SMTP-Debugger Service Ready\r\n")
            
            mail_data = []
            in_data = False
            
            while True:
                line = b""
                while not line.endswith(b"\r\n"):
                    chunk = client.recv(1)
                    if not chunk:
                        break
                    line += chunk
                
                if not line:
                    break
                
                dec = line.decode("utf-8", errors="replace")
                
                if in_data:
                    if dec.strip() == ".":
                        in_data = False
                        client.sendall(b"250 2.0.0 OK message queued for delivery\r\n")
                        break
                    else:
                        mail_data.append(dec)
                else:
                    cmd = dec.upper().strip()
                    if cmd.startswith("HELO") or cmd.startswith("EHLO"):
                        client.sendall(b"250-localhost Hello\r\n250-8BITMIME\r\n250 HELP\r\n")
                    elif cmd.startswith("MAIL FROM:"):
                        client.sendall(b"250 2.1.0 Sender OK\r\n")
                    elif cmd.startswith("RCPT TO:"):
                        client.sendall(b"250 2.1.5 Recipient OK\r\n")
                    elif cmd.startswith("DATA"):
                        in_data = True
                        client.sendall(b"354 Start mail input; end with <CRLF>.<CRLF>\r\n")
                    elif cmd.startswith("QUIT"):
                        client.sendall(b"221 2.0.0 localhost closing connection\r\n")
                        break
                    elif cmd.startswith("RSET"):
                        mail_data = []
                        client.sendall(b"250 OK\r\n")
                    elif cmd.startswith("NOOP"):
                        client.sendall(b"250 OK\r\n")
                    else:
                        client.sendall(b"250 OK\r\n")
            
            client.close()
            
            if mail_data:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("\n" + "█" * 64)
                print(f"⚡ [SHADOW EMAIL TRANSFER RECEIVED] {now_str}")
                print("█" * 64)
                full_body = "".join(mail_data)
                print(full_body.strip())
                print("█" * 64)
                print("✔ Transmission Verified: 250 OK Delivered via Local Socket\n")
                
        except KeyboardInterrupt:
            print("\nShutting down SHADOW Local SMTP Receiver.")
            break
        except Exception as e:
            print(f"Error handling connection: {e}")

    srv.close()

if __name__ == "__main__":
    run_local_receiver()
