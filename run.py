import sys
import os
import argparse
import threading
import json
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
    parser.add_argument("--tripwire", type=str, default=None, 
                        help="Coordinates of the tripwire as a comma-separated list of four floats (x1,y1,x2,y2) or JSON string")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="FastAPI host")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port")
    parser.add_argument("--no-api", action="store_true", help="Disable the FastAPI server")
    parser.add_argument("--no-pipeline", action="store_true", help="Disable the stream monitoring pipeline")
    
    args = parser.parse_args()
    
    # Override tripwire line if provided via CLI
    tripwire_line = CONFIG.tripwire_line
    if args.tripwire:
        try:
            # Try to parse as JSON
            parsed = json.loads(args.tripwire)
            if isinstance(parsed, list) and len(parsed) == 2:
                tripwire_line = [(float(p[0]), float(p[1])) for p in parsed]
                print(f"[*] Overriding tripwire line from CLI: {tripwire_line}")
        except Exception:
            try:
                # Fallback to comma-separated floats
                floats = [float(x.strip()) for x in args.tripwire.split(",") if x.strip()]
                if len(floats) == 4:
                    tripwire_line = [(floats[0], floats[1]), (floats[2], floats[3])]
                    print(f"[*] Overriding tripwire line from CLI list: {tripwire_line}")
                else:
                    print("[-] Error: Tripwire CLI format must be x1,y1,x2,y2")
                    sys.exit(1)
            except Exception as e:
                print(f"[-] Error parsing tripwire parameter: {e}")
                sys.exit(1)
                
    # Initialize DB
    print(f"[*] Initializing presence database at: {CONFIG.db_path}")
    db_manager = DatabaseManager(CONFIG.db_path)
    
    threads = []
    orchestrator = None
    
    if not args.no_pipeline:
        # Pass custom tripwire_line override to object tracker
        from src.object_tracker import ObjectTracker
        tracker = ObjectTracker(tripwire_line=tripwire_line, snapshot_dir=CONFIG.snapshot_dir)
        orchestrator = PipelineOrchestrator(
            db_manager=db_manager, 
            object_tracker=tracker,
            fps_limit=CONFIG.fps_limit,
            cooldown_frames=CONFIG.motion_cooldown_frames
        )
        
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
            uvicorn.run("src.api:app", host=args.host, port=args.port, log_level="info")
        except KeyboardInterrupt:
            print("[*] Stopping API server...")
        finally:
            if orchestrator:
                orchestrator.stop()
    else:
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
