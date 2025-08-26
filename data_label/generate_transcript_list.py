#!/usr/bin/env python3
"""
Script to generate a JSON file with all transcript IDs from the transcripts directory.
This allows the web interface to know which transcripts are available.
"""

import os
import json
import sys

def get_transcript_ids(transcripts_dir):
    """Get all transcript IDs from the transcripts directory."""
    transcript_ids = []
    
    if not os.path.exists(transcripts_dir):
        print(f"Error: Transcripts directory '{transcripts_dir}' does not exist.")
        return transcript_ids
    
    try:
        for item in os.listdir(transcripts_dir):
            item_path = os.path.join(transcripts_dir, item)
            
            # Check if it's a directory and contains a transcript.json file
            if os.path.isdir(item_path):
                transcript_file = os.path.join(item_path, 'transcript.json')
                if os.path.exists(transcript_file):
                    transcript_ids.append(item)
        
        # Sort the IDs for consistent ordering
        transcript_ids.sort()
        print(f"Found {len(transcript_ids)} transcripts")
        
    except Exception as e:
        print(f"Error reading transcripts directory: {e}")
    
    return transcript_ids

def main():
    # Get the parent directory (amd) and construct path to transcripts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    transcripts_dir = os.path.join(parent_dir, 'transcripts')
    
    print(f"Looking for transcripts in: {transcripts_dir}")
    
    # Get all transcript IDs
    transcript_ids = get_transcript_ids(transcripts_dir)
    
    if not transcript_ids:
        print("No transcripts found!")
        sys.exit(1)
    
    # Write to JSON file in the data_label directory
    output_file = os.path.join(current_dir, 'transcript-ids.json')
    
    try:
        with open(output_file, 'w') as f:
            json.dump(transcript_ids, f, indent=2)
        
        print(f"Successfully wrote {len(transcript_ids)} transcript IDs to {output_file}")
        
        # Show first few examples
        print("\nFirst 10 transcript IDs:")
        for i, tid in enumerate(transcript_ids[:10]):
            print(f"  {i+1}. {tid}")
        
        if len(transcript_ids) > 10:
            print(f"  ... and {len(transcript_ids) - 10} more")
            
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()