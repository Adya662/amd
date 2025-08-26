#!/usr/bin/env python3
"""
Script to download transcripts from AWS S3 using transcript URLs from CSV file.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

def download_transcript(transcript_url, output_dir="transcripts"):
    """
    Download a transcript using AWS CLI.
    
    Args:
        transcript_url (str): The S3 path to the transcript file
        output_dir (str): Local directory to save the transcript
    """
    if not transcript_url or transcript_url == 'NULL':
        return False, "No transcript URL provided"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract filename from URL
    filename = transcript_url.split('/')[-1]
    call_id = transcript_url.split('/')[-2]
    
    # Create subdirectory for each call
    call_output_dir = Path(output_dir) / call_id
    call_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = call_output_dir / filename
    
    # Skip if file already exists
    if output_path.exists():
        return True, f"File already exists: {output_path}"
    
    # Construct AWS CLI command
    # S3 bucket: voicex-call-recordings
    s3_url = f"s3://voicex-call-recordings/{transcript_url}"
    
    cmd = [
        "aws", "s3", "cp", 
        s3_url, 
        str(output_path),
        "--profile", "Power-root"
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        if result.returncode == 0:
            return True, f"Successfully downloaded: {output_path}"
        else:
            return False, f"AWS CLI error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Download timeout"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    csv_file = "call_recs_ElevateNowDM2025.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found")
        sys.exit(1)
    
    print(f"Reading CSV file: {csv_file}")
    
    # Counters
    total_rows = 0
    downloaded = 0
    skipped = 0
    failed = 0
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # Verify transcript_url column exists
            if 'transcript_url' not in reader.fieldnames:
                print("Error: 'transcript_url' column not found in CSV")
                sys.exit(1)
            
            print("Starting transcript downloads...")
            
            for row in reader:
                total_rows += 1
                
                # Skip rows up to 1009 (resume from 1010)
                if total_rows <= 1009:
                    continue
                    
                transcript_url = row.get('transcript_url', '').strip()
                call_id = row.get('call_id', 'unknown').strip()
                
                if not transcript_url or transcript_url == 'NULL':
                    skipped += 1
                    print(f"Row {total_rows}: Skipping {call_id} - No transcript URL")
                    continue
                
                print(f"Row {total_rows}: Downloading transcript for call {call_id}")
                success, message = download_transcript(transcript_url)
                
                if success:
                    downloaded += 1
                    print(f"  ✓ {message}")
                else:
                    failed += 1
                    print(f"  ✗ {message}")
                
                # Progress update every 100 rows
                if total_rows % 100 == 0:
                    print(f"Progress: {total_rows} rows processed")
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Summary
    print(f"\n=== Download Summary ===")
    print(f"Total rows processed: {total_rows}")
    print(f"Successfully downloaded: {downloaded}")
    print(f"Skipped (no URL): {skipped}")
    print(f"Failed downloads: {failed}")

if __name__ == "__main__":
    main()
