import sys
import os
import argparse
import threading
import uvicorn
from src.config import CONFIG
from src.database import DatabaseManager
from src.pipeline import PipelineOrchestrator

def run_pipeline(orchestrator, rtsp_url):
    print(f"[*] Starting CCTV Monitoring Pipeline on: {rtsp_url}")
    try:
        orchestrator.run_on_stream(rtsp_url)
    except KeyboardInterrupt:
        print("[*] Stopping pipeline...")

def main():
    parser = argparse.ArgumentParser(description="House Presence Monitoring System Daemon")
    parser.add_argument("--rtsp", type=str, default=CONFIG.rtsp_url, help="RTSP stream URL")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="FastAPI host")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port")
    parser.add_argument("--no-api", action="store_true", help="Disable the FastAPI server")
    parser.add_argument("--no-pipeline", action="store_true", help="Disable the stream monitoring pipeline")
    
    args = parser.parse_args()
    
    # Initialize DB
    print(f"[*] Initializing presence database at: {CONFIG.db_path}")
    db_manager = DatabaseManager(CONFIG.db_path)
    
    threads = []
    orchestrator = None
    
    if not args.no_pipeline:
        orchestrator = PipelineOrchestrator(db_manager=db_manager, fps_limit=CONFIG.fps_limit)
        pipeline_thread = threading.Thread(
            target=run_pipeline, 
            args=(orchestrator, args.rtsp), 
            daemon=True
        )
        pipeline_thread.start()
        threads.append(pipeline_thread)
        
    if not args.no_api:
        print(f"[*] Starting Query API on http://{args.host}:{args.port}")
        try:
            # We run uvicorn in the main thread
            uvicorn.run("src.api:app", host=args.host, port=args.port, log_level="info")
        except KeyboardInterrupt:
            print("[*] Stopping API server...")
        finally:
            if orchestrator:
                orchestrator.stop()
    else:
        # If API is disabled, just block on the pipeline thread
        if threads:
            try:
                for t in threads:
                    t.join()
            except KeyboardInterrupt:
                print("[*] Exiting...")
                if orchestrator:
                    orchestrator.stop()

if __name__ == "__main__":
    main()
