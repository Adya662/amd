import json, re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import pandas as pd
from difflib import SequenceMatcher

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
    def __init__(self, transcripts_folder="transcripts", dict_path="dict.json", fuzzy_threshold=0.9):
        self.folder = Path(transcripts_folder)
        self.stats = FilteringStats()
        self.fuzzy_threshold = fuzzy_threshold

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
        p = self.folder / call_id / "transcript.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except:
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
    def analyze(self, call_ids: List[str]) -> Tuple[Dict[str, Any], pd.DataFrame]:
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
        for cid, utext in with_user:
            is_m, reason = self._detect_machine(utext)
            rows.append({
                "call_id": cid,
                "is_machine": is_m,
                "reason": reason,
                "user_excerpt": utext[:120]
            })
            if is_m: 
                m_count += 1
                machine_ids.append(cid)
            else: 
                h_count += 1

        self.stats.machine_detected = m_count
        self.stats.human_detected = h_count

        summary = {
            **self.stats.__dict__,
            "machine_call_ids": machine_ids
        }

        return summary, pd.DataFrame(rows)

# --------------- Runner ---------------
def load_call_ids(folder: str) -> List[str]:
    return [d.name for d in Path(folder).iterdir() if d.is_dir()]

def main():
    analyzer = FuzzyAMD("transcripts", "dict.json", fuzzy_threshold=0.9)
    call_ids = load_call_ids("transcripts")
    summary, df = analyzer.analyze(call_ids)

    print("\n=== FUZZY AMD SUMMARY ===")
    for k, v in summary.items():
        if k != "machine_call_ids":  # Skip the IDs list in the summary
            print(f"{k:30}: {v}")

    # Print machine detected call IDs
    machine_ids = summary.get("machine_call_ids", [])
    print(f"\n=== MACHINE DETECTED CALL IDs ({len(machine_ids)} calls) ===")
    if machine_ids:
        # Print first 10 IDs for brevity
        for i, call_id in enumerate(machine_ids[:10]):
            print(f"{i+1:2d}. {call_id}")
        if len(machine_ids) > 10:
            print(f"... and {len(machine_ids) - 10} more machine calls")

    df.to_csv("amd_fuzzy_results.csv", index=False)
    print(f"\nSaved results to amd_fuzzy_results.csv")
    
    # Save machine detected call IDs to text file
    if machine_ids:
        with open("machine_detected_call_ids.txt", "w") as f:
            f.write(f"Machine Detected Call IDs ({len(machine_ids)} calls)\n")
            f.write("=" * 50 + "\n\n")
            for i, call_id in enumerate(machine_ids, 1):
                f.write(f"{i:3d}. {call_id}\n")
        print(f"Saved {len(machine_ids)} machine call IDs to machine_detected_call_ids.txt")

if __name__ == "__main__":
    main()