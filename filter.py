import json, re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import pandas as pd
from difflib import SequenceMatcher
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

@dataclass
class FilteringStats:
    initial_calls: int = 0
    after_empty_filter: int = 0
    after_single_utt_filter: int = 0
    assistant_only_count: int = 0
    after_user_filter: int = 0
    machine_detected: int = 0
    human_detected: int = 0

class FuzzyAMD:
    def __init__(self, s3_bucket=None, transcripts_prefix="transcripts", 
                 transcripts_folder="transcripts", dict_path="dict.json", fuzzy_threshold=0.9):
        # S3 setup
        self.s3_bucket = s3_bucket
        self.transcripts_prefix = transcripts_prefix
        self.s3_client = boto3.client('s3') if s3_bucket else None
        
        # Keep existing local setup as fallback
        self.folder = Path(transcripts_folder) if not s3_bucket else None
        self.stats = FilteringStats()
        self.fuzzy_threshold = fuzzy_threshold

        # Store call data (call_id -> (transcript_url, duration))
        self.call_data_map = {}

        # Load dict.json
        with open(dict_path, "r", encoding="utf-8") as f:
            self.cfg: Dict[str, Any] = json.load(f)

        # Extract phrase groups
        self.voicemail = self.cfg["voicemail"]
        self.ivr_menu = self.cfg["ivr_menu"]
        self.system = self.cfg["system"]
        self.callback = self.cfg["callback_request"]

        # Number regex
        self.num_pattern = re.compile(self.cfg["number_regex"]["digits"])

        # Build decisive set = voicemail + ivr + system
        self.decisive_phrases = self.voicemail + self.ivr_menu + self.system

    # ------------------- helpers -------------------
    def _load_transcript(self, call_id: str):
        # Check if we have transcript URL for this call
        if call_id in self.call_data_map:
            transcript_url, _ = self.call_data_map[call_id]
            return self._load_transcript_from_url(transcript_url, call_id)
        
        if self.s3_bucket:
            # S3 mode - fallback to old method
            s3_key = f"{self.transcripts_prefix}/{call_id}/transcript.json"
            try:
                response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
                content = response['Body'].read().decode('utf-8')
                return json.loads(content)
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    return None
                print(f"[ERROR] S3 error for {call_id}: {e}")
                return None
            except Exception as e:
                print(f"[WARN] {call_id}: {e}")
                return None
        else:
            # Local mode (existing code)
            p = self.folder / call_id / "transcript.json"
            if not p.exists():
                return None
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[WARN] {call_id}: {e}")
                return None

    def _load_transcript_from_url(self, transcript_url: str, call_id: str):
        """Load transcript from S3 URL"""
        try:
            # Parse S3 URL to extract bucket and key
            # Format: s3://bucket-name/path/to/file or direct S3 path
            if transcript_url.startswith('s3://'):
                # Full S3 URL: s3://bucket/key
                parts = transcript_url[5:].split('/', 1)
                bucket = parts[0]
                key = parts[1] if len(parts) > 1 else ''
            elif 'calls/' in transcript_url:
                # Relative path: calls/2025/03/11/.../transcript.json
                bucket = self.s3_bucket or 'voicex-call-recordings'  # Default bucket
                key = transcript_url
            else:
                print(f"[WARN] {call_id}: Unrecognized transcript URL format: {transcript_url}")
                return None
            
            # Create S3 client if not exists
            if not self.s3_client:
                self.s3_client = boto3.client('s3')
            
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            return json.loads(content)
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            print(f"[ERROR] S3 error for {call_id}: {e}")
            return None
        except Exception as e:
            print(f"[WARN] {call_id}: {e}")
            return None

    def _is_empty(self, t): 
        return not t or all((u.get("content") or "").strip() == "" for u in t)

    def _has_single_utt(self, t): 
        return len(t) == 1

    def _user_text(self, t):
        return " ".join(
            (u.get("content") or "").strip().lower()
            for u in t if (u.get("speaker","").lower() == "user")
        )

    def _contains_phrase(self, text: str, phrases: List[str]) -> Tuple[bool, str, float]:
        """Check if any phrase is contained within the text using multiple methods"""
        text_lower = text.lower()
        
        # Method 1: Direct substring match (most reliable)
        for phrase in phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in text_lower:
                return True, phrase, 1.0
        
        # Method 2: Fuzzy matching for partial/error cases (lowered threshold)
        for phrase in phrases:
            phrase_lower = phrase.lower()
            # Try fuzzy matching on the phrase within the text
            ratio = SequenceMatcher(None, phrase_lower, text_lower).ratio()
            if ratio >= 0.7:  # Much lower threshold for fuzzy
                return True, phrase, ratio
                
            # Also try finding the phrase as substring with some flexibility
            words_in_phrase = phrase_lower.split()
            words_in_text = text_lower.split()
            
            # Check if most words from phrase appear in text
            if len(words_in_phrase) >= 2:
                matches = sum(1 for word in words_in_phrase if any(word in text_word for text_word in words_in_text))
                if matches >= len(words_in_phrase) * 0.7:  # 70% of words match
                    return True, phrase, matches / len(words_in_phrase)
        
        return False, "", 0.0

    # ------------------- machine detection -------------------
    def _detect_machine(self, user_text: str) -> Tuple[bool, str]:
        if not user_text.strip():
            return False, "no_user_text"

        # 1) Check voicemail patterns first (most common)
        hit, phrase, score = self._contains_phrase(user_text, self.voicemail)
        if hit:
            return True, f"voicemail:'{phrase}'~{score:.2f}"

        # 2) Check IVR menu patterns
        hit, phrase, score = self._contains_phrase(user_text, self.ivr_menu)
        if hit:
            return True, f"ivr_menu:'{phrase}'~{score:.2f}"

        # 3) Check system messages
        hit, phrase, score = self._contains_phrase(user_text, self.system)
        if hit:
            return True, f"system:'{phrase}'~{score:.2f}"

        # 4) callback + number combination
        cb_hit, phrase, score = self._contains_phrase(user_text, self.callback)
        if cb_hit and self.num_pattern.search(user_text):
            return True, f"callback+digits:'{phrase}'~{score:.2f}"

        return False, "no_match"

    # ------------------- pipeline -------------------
    def analyze(self, call_data: List[Tuple[str, str, int]]) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Analyze calls using call data (call_id, transcript_url, duration)"""
        # Store call data mapping for transcript loading
        self.call_data_map = {call_id: (transcript_url, duration) for call_id, transcript_url, duration in call_data}
        call_ids = [call_id for call_id, _, _ in call_data]
        
        self.stats.initial_calls = len(call_ids)
        
        # empty filter
        non_empty = [(cid, t) for cid in call_ids if (t := self._load_transcript(cid)) and not self._is_empty(t)]
        self.stats.after_empty_filter = len(non_empty)

        # single utt filter
        multi = [(cid, t) for cid, t in non_empty if not self._has_single_utt(t)]
        self.stats.after_single_utt_filter = len(multi)

        # assistant-only filter
        with_user, assistant_only = [], 0
        for cid, t in multi:
            utext = self._user_text(t)
            if not utext.strip():
                assistant_only += 1
            else:
                with_user.append((cid, utext))
        self.stats.assistant_only_count = assistant_only
        self.stats.after_user_filter = len(with_user)

        # detection
        rows = []
        m_count = h_count = 0
        machine_ids = []
        machine_data = []  # Store (call_id, duration) for machine calls
        total_calls = len(with_user)
        
        print(f"\nRunning machine detection on {total_calls} calls...")
        for i, (cid, utext) in enumerate(with_user):
            # Progress tracking for large datasets
            if i > 0 and i % 500 == 0:
                print(f"Processed {i}/{total_calls} calls ({i/total_calls*100:.1f}%)")
                
            is_m, reason = self._detect_machine(utext)
            
            # Get duration for this call
            duration = self.call_data_map.get(cid, (None, 0))[1]
            
            rows.append({
                "call_id": cid,
                "is_machine": is_m,
                "reason": reason,
                "user_excerpt": utext[:120],
                "duration": duration
            })
            if is_m: 
                m_count += 1
                machine_ids.append(cid)
                machine_data.append((cid, duration))
            else:
                h_count += 1
        
        print(f"Completed detection on {total_calls} calls")
        
        self.stats.machine_detected = m_count
        self.stats.human_detected = h_count
        
        summary = {
            **self.stats.__dict__,
            "machine_call_ids": machine_ids,
            "machine_call_data": machine_data
        }

        return summary, pd.DataFrame(rows)

    def analyze_legacy(self, call_ids: List[str]) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Legacy analyze method for backward compatibility"""
        # Convert call_ids to call_data format with no duration info
        call_data = [(call_id, "", 0) for call_id in call_ids]
        return self.analyze(call_data)

# --------------- Runner ---------------
def load_call_ids(folder: str) -> List[str]:
    """Load call IDs from local directory"""
    return [d.name for d in Path(folder).iterdir() if d.is_dir()]

def load_call_data_from_csv(csv_path: str) -> List[Tuple[str, str, int]]:
    """Load call data (call_id, transcript_url, duration) from CSV file"""
    
    # Try multiple parsing strategies
    call_data = []
    
    # Strategy 1: Use pandas with quoting
    try:
        df = pd.read_csv(csv_path, quotechar='"', escapechar='\\', on_bad_lines='skip', low_memory=False)
        print(f"Strategy 1 - Successfully read CSV with {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if we have the right columns
        if all(col in df.columns for col in ['call_id', 'transcript_url', 'duration']):
            print("All required columns found!")
            
            # Filter valid rows
            valid_rows = df.dropna(subset=['call_id', 'transcript_url'])
            valid_rows = valid_rows[valid_rows['transcript_url'].astype(str).str.contains('transcript.json', na=False)]
            
            print(f"Found {len(valid_rows)} rows with valid transcript URLs")
            
            for _, row in valid_rows.iterrows():
                try:
                    call_id = str(row['call_id']).strip()
                    transcript_url = str(row['transcript_url']).strip()
                    
                    # Convert relative path to full S3 URL
                    if transcript_url and not transcript_url.startswith('s3://'):
                        transcript_url = f"s3://voicex-call-recordings/{transcript_url}"
                    
                    # Parse duration safely
                    try:
                        duration = int(float(row['duration'])) if pd.notna(row['duration']) else 0
                    except (ValueError, TypeError):
                        duration = 0
                    
                    call_data.append((call_id, transcript_url, duration))
                except Exception as e:
                    continue
                    
            if call_data:
                print(f"Successfully extracted {len(call_data)} call records")
                return call_data
                
    except Exception as e:
        print(f"Strategy 1 failed: {e}")
    
    # Strategy 2: Manual parsing for specific columns
    print("Trying manual parsing strategy...")
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            headers = [h.strip() for h in header_line.split(',')]
            
            # Find column indices
            call_id_idx = next((i for i, h in enumerate(headers) if 'call_id' in h.lower()), None)
            duration_idx = next((i for i, h in enumerate(headers) if 'duration' in h.lower()), None)
            transcript_url_idx = next((i for i, h in enumerate(headers) if 'transcript_url' in h.lower()), None)
            
            if None in [call_id_idx, duration_idx, transcript_url_idx]:
                print(f"Could not find required columns. Headers: {headers}")
                return []
            
            print(f"Found columns - call_id: {call_id_idx}, duration: {duration_idx}, transcript_url: {transcript_url_idx}")
            
            line_count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                line_count += 1
                if line_count > 10000:  # Limit for safety
                    break
                
                # Simple split approach - might not work perfectly with quoted fields
                parts = line.split(',')
                if len(parts) > max(call_id_idx, duration_idx, transcript_url_idx):
                    try:
                        call_id = parts[call_id_idx].strip('"').strip()
                        duration_str = parts[duration_idx].strip('"').strip()
                        transcript_url = parts[transcript_url_idx].strip('"').strip()
                        
                        # Parse duration
                        try:
                            duration = int(float(duration_str)) if duration_str and duration_str != 'NULL' else 0
                        except (ValueError, TypeError):
                            duration = 0
                        
                        # Check if transcript URL looks valid and convert to full S3 URL
                        if transcript_url and 'transcript.json' in transcript_url:
                            # Convert relative path to full S3 URL
                            if not transcript_url.startswith('s3://'):
                                transcript_url = f"s3://voicex-call-recordings/{transcript_url}"
                            call_data.append((call_id, transcript_url, duration))
                            
                    except Exception as e:
                        continue
                        
        print(f"Manual parsing extracted {len(call_data)} call records")
        
    except Exception as e:
        print(f"Manual parsing failed: {e}")
    
    return call_data

def load_call_ids_from_csv(csv_path: str) -> List[str]:
    """Load call IDs from CSV file (backward compatibility)"""
    call_data = load_call_data_from_csv(csv_path)
    return [call_id for call_id, _, _ in call_data]

def main():
    # Configuration - modify these for your setup
    USE_CSV_WITH_S3 = True  # Set to True to use CSV with transcript URLs (requires AWS creds)
    USE_S3 = False  # Set to True for S3 processing (legacy mode)
    S3_BUCKET = "voicex-call-recordings"  # Default S3 bucket for transcript URLs
    S3_PREFIX = "transcripts"  # S3 path prefix (legacy mode)
    CSV_FILE = "all_EN_calls.csv"  # CSV with call IDs and transcript URLs
    
    if USE_CSV_WITH_S3:
        # New mode: Use CSV with transcript URLs 
        print(f"Using CSV with S3 transcript URLs from {CSV_FILE}")
        analyzer = FuzzyAMD(
            s3_bucket=S3_BUCKET,  # Default bucket for transcript URLs
            dict_path="dict.json",
            fuzzy_threshold=0.9
        )
        call_data = load_call_data_from_csv(CSV_FILE)
        print(f"Loaded {len(call_data)} call records from {CSV_FILE}")
        
        # Run analysis
        summary, df = analyzer.analyze(call_data)
        
    elif USE_S3:
        # S3 mode (legacy)
        print(f"Using S3 mode: bucket={S3_BUCKET}, prefix={S3_PREFIX}")
        analyzer = FuzzyAMD(
            s3_bucket=S3_BUCKET,
            transcripts_prefix=S3_PREFIX,
            dict_path="dict.json",
            fuzzy_threshold=0.9
        )
        call_ids = load_call_ids_from_csv(CSV_FILE)
        print(f"Loaded {len(call_ids)} call IDs from {CSV_FILE}")
        summary, df = analyzer.analyze_legacy(call_ids)
    else:
        # Local mode (existing behavior)
        print("Using local mode")
        analyzer = FuzzyAMD(
            transcripts_folder="transcripts",
            dict_path="dict.json",
            fuzzy_threshold=0.9
        )
        call_ids = load_call_ids("transcripts")
        print(f"Loaded {len(call_ids)} call IDs from transcripts directory")
        summary, df = analyzer.analyze_legacy(call_ids)

    print("\n=== FUZZY AMD SUMMARY ===")
    for k, v in summary.items():
        if k not in ["machine_call_ids", "machine_call_data"]:  # Skip the lists in summary
            print(f"{k:30}: {v}")

    # Print machine detected call IDs with durations
    machine_data = summary.get("machine_call_data", [])
    machine_ids = summary.get("machine_call_ids", [])
    print(f"\n=== MACHINE DETECTED CALLS ({len(machine_ids)} calls) ===")
    if machine_data:
        # Print first 10 calls with durations
        for i, (call_id, duration) in enumerate(machine_data[:10]):
            print(f"{i+1:2d}. {call_id} (duration: {duration}s)")
        if len(machine_data) > 10:
            print(f"... and {len(machine_data) - 10} more machine calls")

    # Save detailed results
    df.to_csv("amd_fuzzy_results.csv", index=False)
    print(f"\nSaved detailed results to amd_fuzzy_results.csv")
    
    # Save machine detected calls with durations
    if machine_data:
        # Save to TXT file
        with open("machine_detected_calls_with_duration.txt", "w") as f:
            f.write(f"Machine Detected Calls with Durations ({len(machine_data)} calls)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"{'#':<4} {'Call ID':<40} {'Duration (s)':<12}\n")
            f.write("-" * 60 + "\n")
            total_duration = 0
            for i, (call_id, duration) in enumerate(machine_data, 1):
                f.write(f"{i:<4} {call_id:<40} {duration:<12}\n")
                total_duration += duration
            f.write("-" * 60 + "\n")
            f.write(f"Total machine calls: {len(machine_data)}\n")
            f.write(f"Total duration: {total_duration} seconds ({total_duration/60:.1f} minutes)\n")
            f.write(f"Average duration: {total_duration/len(machine_data):.1f} seconds\n")
        
        # Save to CSV file
        machine_df = pd.DataFrame(machine_data, columns=['call_id', 'duration_seconds'])
        machine_df.to_csv("machine_detected_calls_with_duration.csv", index=False)
        
        print(f"Saved {len(machine_data)} machine calls with durations to:")
        print(f"  - machine_detected_calls_with_duration.txt")
        print(f"  - machine_detected_calls_with_duration.csv")
        print(f"Total machine call duration: {sum(d for _, d in machine_data)/60:.1f} minutes")

if __name__ == "__main__":
    main()