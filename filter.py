import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import pandas as pd
from collections import Counter

@dataclass
class FilteringStats:
    """Class to track filtering statistics at each step"""
    initial_calls: int = 0
    after_empty_filter: int = 0
    after_single_utterance_filter: int = 0
    machine_answered: int = 0
    human_answered: int = 0

class CallTranscriptAnalyzer:
    def __init__(self, transcripts_folder: str = "transcripts"):
        self.transcripts_folder = Path(transcripts_folder)
        self.stats = FilteringStats()
        
        # Enhanced machine detection patterns based on real data
        self.machine_patterns = {
            'voicemail_indicators': [
                r'your call has been forwarded to voicemail',
                r'the person you are trying to reach is not available',
                r'please record your message',
                r'you may hang up',
                r'forwarded to voicemail',
                r'leave a message after the tone',
                r'at the tone',
                r'at the turn',  # Common transcription error
                r'you have reached the voicemail',
                r'mailbox is full',
                r'please leave a message',
                r'after the beep',
                r'when you have finished recording',
                r'forward forwarded to voicemail'  # From patterns.txt
            ],
            'automated_responses': [
                r'this call is now being recorded',
                r'this call may be recorded',
                r'for quality assurance',
                r'press \d+ for',
                r'if this is an emergency',
                r'our office hours are',
                r'we are currently closed',
                r'thank you for calling',
                r'your call is important to us',
                r'please stay on the line',
                r'all representatives are busy',
                r'all agents are currently busy',
                r'please hold'
            ],
            'assistant_indicators': [
                r'i noticed you\'?ve been quiet',
                r'are you still there',
                r'hello\?.*hello\?',
                r'can you hear me',
                r'is anyone there',
                r'assistant speaking.*quiet.*still',  # From patterns.txt
                r'hello.*hello.*hello'  # Multiple hellos
            ],
            'system_messages': [
                r'the number you have dialed',
                r'this number has been disconnected',
                r'the subscriber you have called',
                r'please check the number',
                r'invalid number',
                r'number is not in service',
                r'please try again later'
            ],
            'silence_indicators': [
                r'no response',
                r'silence',
                r'quiet for',
                r'no answer',
                r'not responding',
                r'line is quiet'
            ]
        }
        
        # Conversation flow patterns that indicate machines
        self.machine_flow_patterns = [
            r'hello.*forwarded.*voicemail',
            r'not available.*record.*message',
            r'voicemail.*tone.*message',
            r'recording.*hang up',
            r'quiet.*still there'
        ]

    def load_transcript(self, call_id: str) -> Dict[str, Any]:
        """Load transcript JSON for a given call ID"""
        transcript_path = self.transcripts_folder / call_id / "transcript.json"
        
        if not transcript_path.exists():
            return None
            
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading transcript for {call_id}: {e}")
            return None

    def is_empty_transcript(self, transcript: List[Dict]) -> bool:
        """Check if transcript is empty or contains no meaningful content"""
        if not transcript:
            return True
        
        # Check if all utterances are empty or contain only whitespace
        meaningful_content = False
        for utterance in transcript:
            content = utterance.get('content', '').strip()
            if content and len(content) > 0:
                meaningful_content = True
                break
                
        return not meaningful_content

    def has_single_utterance(self, transcript: List[Dict]) -> bool:
        """Check if transcript has only one utterance"""
        return len(transcript) == 1

    def extract_transcript_text(self, transcript: List[Dict]) -> str:
        """Extract all text content from transcript"""
        text_parts = []
        for utterance in transcript:
            content = utterance.get('content', '').strip()
            speaker = utterance.get('speaker', '')
            if content:
                text_parts.append(f"{speaker}: {content}")
        return " ".join(text_parts).lower()

    def calculate_machine_score(self, transcript_text: str) -> Tuple[float, List[str]]:
        """Calculate machine likelihood score based on patterns"""
        matches = []
        total_score = 0
        
        # Check each pattern category
        for category, patterns in self.machine_patterns.items():
            category_matches = 0
            for pattern in patterns:
                if re.search(pattern, transcript_text, re.IGNORECASE):
                    matches.append(f"{category}: {pattern}")
                    category_matches += 1
            
            # Weight different categories
            if category == 'voicemail_indicators':
                total_score += category_matches * 3
            elif category == 'automated_responses':
                total_score += category_matches * 2
            elif category == 'assistant_indicators':
                total_score += category_matches * 2.5
            elif category == 'system_messages':
                total_score += category_matches * 4
            elif category == 'silence_indicators':
                total_score += category_matches * 2
        
        # Check flow patterns (higher weight)
        for flow_pattern in self.machine_flow_patterns:
            if re.search(flow_pattern, transcript_text, re.IGNORECASE):
                matches.append(f"flow_pattern: {flow_pattern}")
                total_score += 5
        
        # Additional heuristics
        words = transcript_text.split()
        
        # Very short responses often indicate machines
        if len(words) < 10:
            total_score += 1
            matches.append("heuristic: very_short_response")
        
        # Repetitive patterns
        if re.search(r'(\b\w+\b).*\1.*\1', transcript_text):
            total_score += 2
            matches.append("heuristic: repetitive_content")
        
        # High frequency of numbers (phone menus)
        number_count = len(re.findall(r'\b\d+\b', transcript_text))
        if number_count > 3:
            total_score += 2
            matches.append("heuristic: multiple_numbers")
        
        return total_score, matches

    def is_machine_answered(self, transcript: List[Dict], threshold: float = 3.0) -> Tuple[bool, float, List[str]]:
        """Determine if call was answered by machine - FULL VERSION"""
        transcript_text = self.extract_transcript_text(transcript)
        score, matches = self.calculate_machine_score(transcript_text)
        
        return score >= threshold, score, matches

    def is_machine_answered_conservative(self, transcript: List[Dict], threshold: float = 2.0) -> Tuple[bool, float, List[str]]:
        """Conservative machine detection - only automated responses and system messages"""
        transcript_text = self.extract_transcript_text(transcript)
        score, matches = self.calculate_machine_score_conservative(transcript_text)
        
        return score >= threshold, score, matches

    def calculate_machine_score_conservative(self, transcript_text: str) -> Tuple[float, List[str]]:
        """Conservative scoring - only automated responses and system messages"""
        matches = []
        total_score = 0
        
        # Only check automated responses and system messages
        conservative_patterns = {
            'automated_responses': self.machine_patterns['automated_responses'],
            'system_messages': self.machine_patterns['system_messages']
        }
        
        for category, patterns in conservative_patterns.items():
            category_matches = 0
            for pattern in patterns:
                if re.search(pattern, transcript_text, re.IGNORECASE):
                    matches.append(f"{category}: {pattern}")
                    category_matches += 1
            
            # Weight different categories
            if category == 'automated_responses':
                total_score += category_matches * 2
            elif category == 'system_messages':
                total_score += category_matches * 4
        
        # Only include multiple numbers heuristic (phone menus)
        number_count = len(re.findall(r'\b\d+\b', transcript_text))
        if number_count > 3:
            total_score += 2
            matches.append("heuristic: multiple_numbers")
        
        return total_score, matches

    def analyze_all_transcripts(self, call_ids: List[str]) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Analyze all transcripts and create data funnel"""
        self.stats.initial_calls = len(call_ids)
        
        results = []
        
        print("Starting transcript analysis pipeline...")
        print(f"Initial calls: {self.stats.initial_calls}")
        
        # Step 1: Filter empty transcripts
        non_empty_calls = []
        for call_id in call_ids:
            transcript = self.load_transcript(call_id)
            if transcript is not None and not self.is_empty_transcript(transcript):
                non_empty_calls.append((call_id, transcript))
        
        self.stats.after_empty_filter = len(non_empty_calls)
        print(f"After empty filter: {self.stats.after_empty_filter} calls")
        
        # Step 2: Filter single utterance transcripts
        multi_utterance_calls = []
        for call_id, transcript in non_empty_calls:
            if not self.has_single_utterance(transcript):
                multi_utterance_calls.append((call_id, transcript))
        
        self.stats.after_single_utterance_filter = len(multi_utterance_calls)
        print(f"After single utterance filter: {self.stats.after_single_utterance_filter} calls")
        
        # Step 3: Machine detection - BOTH VERSIONS
        machine_calls_full = []
        human_calls_full = []
        machine_calls_conservative = []
        human_calls_conservative = []
        
        for call_id, transcript in multi_utterance_calls:
            # Full version
            is_machine_full, score_full, matches_full = self.is_machine_answered(transcript)
            
            # Conservative version
            is_machine_conservative, score_conservative, matches_conservative = self.is_machine_answered_conservative(transcript)
            
            result = {
                'call_id': call_id,
                'is_machine_full': is_machine_full,
                'machine_score_full': score_full,
                'matches_full': matches_full,
                'is_machine_conservative': is_machine_conservative,
                'machine_score_conservative': score_conservative,
                'matches_conservative': matches_conservative,
                'utterance_count': len(transcript),
                'transcript_text': self.extract_transcript_text(transcript)[:200] + "..."  # First 200 chars
            }
            
            results.append(result)
            
            # Track both versions
            if is_machine_full:
                machine_calls_full.append(call_id)
            else:
                human_calls_full.append(call_id)
                
            if is_machine_conservative:
                machine_calls_conservative.append(call_id)
            else:
                human_calls_conservative.append(call_id)
        
        # Update stats for full version (primary)
        self.stats.machine_answered = len(machine_calls_full)
        self.stats.human_answered = len(human_calls_full)
        
        # Print comparison results
        print(f"\nFULL VERSION RESULTS:")
        print(f"Machine answered calls: {len(machine_calls_full)}")
        print(f"Human answered calls: {len(human_calls_full)}")
        
        print(f"\nCONSERVATIVE VERSION RESULTS:")
        print(f"Machine answered calls: {len(machine_calls_conservative)}")
        print(f"Human answered calls: {len(human_calls_conservative)}")
        
        print(f"\nCOMPARISON:")
        print(f"Difference in machine detection: {len(machine_calls_full) - len(machine_calls_conservative)} calls")
        
        # Create summary
        summary = {
            'filtering_stats': self.stats,
            'machine_calls_full': machine_calls_full,
            'human_calls_full': human_calls_full,
            'machine_calls_conservative': machine_calls_conservative,
            'human_calls_conservative': human_calls_conservative,
            'conversion_rates': {
                'empty_filter_rate': (self.stats.after_empty_filter / self.stats.initial_calls * 100) if self.stats.initial_calls > 0 else 0,
                'single_utterance_filter_rate': (self.stats.after_single_utterance_filter / self.stats.after_empty_filter * 100) if self.stats.after_empty_filter > 0 else 0,
                'machine_detection_rate_full': (len(machine_calls_full) / self.stats.after_single_utterance_filter * 100) if self.stats.after_single_utterance_filter > 0 else 0,
                'machine_detection_rate_conservative': (len(machine_calls_conservative) / self.stats.after_single_utterance_filter * 100) if self.stats.after_single_utterance_filter > 0 else 0
            }
        }
        
        return summary, pd.DataFrame(results)

    def create_data_funnel_visualization(self, summary: Dict[str, Any]):
        """Create a text-based data funnel visualization"""
        stats = summary['filtering_stats']
        rates = summary['conversion_rates']
        
        print("\n" + "="*60)
        print("DATA FUNNEL ANALYSIS")
        print("="*60)
        
        # Show both versions
        machine_full = len(summary['machine_calls_full'])
        human_full = len(summary['human_calls_full'])
        machine_conservative = len(summary['machine_calls_conservative'])
        human_conservative = len(summary['human_calls_conservative'])
        
        funnel_steps = [
            ("Initial ElevateNow Calls (<30s)", stats.initial_calls, 100.0),
            ("After Empty Filter", stats.after_empty_filter, rates['empty_filter_rate']),
            ("After Single Utterance Filter", stats.after_single_utterance_filter, rates['single_utterance_filter_rate']),
        ]
        
        for i, (step_name, count, percentage) in enumerate(funnel_steps):
            bar_length = int(percentage / 100 * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"{step_name:.<35} {count:>6} calls │{bar}│ {percentage:5.1f}%")
        
        # Show both detection versions
        print("    └── FULL VERSION (All patterns):")
        full_machine_rate = (machine_full / stats.after_single_utterance_filter * 100) if stats.after_single_utterance_filter > 0 else 0
        full_human_rate = (human_full / stats.after_single_utterance_filter * 100) if stats.after_single_utterance_filter > 0 else 0
        
        bar_length = int(full_machine_rate / 100 * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"        Machine Answered......... {machine_full:>6} calls │{bar}│ {full_machine_rate:5.1f}%")
        
        bar_length = int(full_human_rate / 100 * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"        Human Answered........... {human_full:>6} calls │{bar}│ {full_human_rate:5.1f}%")
        
        print("    └── CONSERVATIVE VERSION (Auto + System only):")
        cons_machine_rate = (machine_conservative / stats.after_single_utterance_filter * 100) if stats.after_single_utterance_filter > 0 else 0
        cons_human_rate = (human_conservative / stats.after_single_utterance_filter * 100) if stats.after_single_utterance_filter > 0 else 0
        
        bar_length = int(cons_machine_rate / 100 * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"        Machine Answered......... {machine_conservative:>6} calls │{bar}│ {cons_machine_rate:5.1f}%")
        
        bar_length = int(cons_human_rate / 100 * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"        Human Answered........... {human_conservative:>6} calls │{bar}│ {cons_human_rate:5.1f}%")
        
        print("="*60)

def generate_sample_call_ids(num_calls: int = 100) -> List[str]:
    """Generate sample call IDs for testing"""
    return [f"call_{i:04d}" for i in range(1, num_calls + 1)]

def load_call_ids_from_csv(csv_path: str = "call_recs_ElevateNowDM2025.csv") -> List[str]:
    """Load call IDs from the CSV file"""
    try:
        df = pd.read_csv(csv_path)
        call_ids = df['call_id'].astype(str).tolist()
        print(f"Loaded {len(call_ids)} call IDs from {csv_path}")
        return call_ids
    except Exception as e:
        print(f"Error loading CSV file {csv_path}: {e}")
        return []

def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = CallTranscriptAnalyzer()
    
    print("Call Transcript Analysis Pipeline")
    print("="*50)
    
    # Check if transcripts directory exists
    if not analyzer.transcripts_folder.exists():
        print(f"Transcripts directory '{analyzer.transcripts_folder}' not found!")
        return
    
    # Get call IDs directly from transcripts folder
    call_ids = [d.name for d in analyzer.transcripts_folder.iterdir() if d.is_dir()]
    
    if not call_ids:
        print("No transcript directories found!")
        return
    
    print(f"Processing {len(call_ids)} call IDs from transcripts folder")
    
    # Run analysis
    summary, results_df = analyzer.analyze_all_transcripts(call_ids)
    
    # Display funnel visualization
    analyzer.create_data_funnel_visualization(summary)
    
    # Show sample results
    print("\nSAMPLE ANALYSIS RESULTS (First 3 calls):")
    print("-" * 80)
    if not results_df.empty:
        for _, row in results_df.head(3).iterrows():
            print(f"Call ID: {row['call_id']}")
            print(f"  FULL VERSION:")
            print(f"    Machine: {'Yes' if row['is_machine_full'] else 'No'} (Score: {row['machine_score_full']:.1f})")
            if row['matches_full']:
                print(f"    Matches: {', '.join(row['matches_full'][:2])}")  # Show first 2 matches
            print(f"  CONSERVATIVE VERSION:")
            print(f"    Machine: {'Yes' if row['is_machine_conservative'] else 'No'} (Score: {row['machine_score_conservative']:.1f})")
            if row['matches_conservative']:
                print(f"    Matches: {', '.join(row['matches_conservative'][:2])}")
            print(f"  Utterances: {row['utterance_count']}")
            print(f"  Preview: {row['transcript_text'][:100]}...")
            print()
    
    # Save results
    results_df.to_csv('call_analysis_results.csv', index=False)
    print(f"Results saved to 'call_analysis_results.csv'")
    
    # Summary statistics for both versions
    print(f"\nSUMMARY STATISTICS:")
    print(f"FULL VERSION:")
    print(f"  - Average machine score: {results_df[results_df['is_machine_full']]['machine_score_full'].mean():.2f}")
    print(f"  - Average human score: {results_df[~results_df['is_machine_full']]['machine_score_full'].mean():.2f}")
    
    print(f"\nCONSERVATIVE VERSION:")
    print(f"  - Average machine score: {results_df[results_df['is_machine_conservative']]['machine_score_conservative'].mean():.2f}")
    print(f"  - Average human score: {results_df[~results_df['is_machine_conservative']]['machine_score_conservative'].mean():.2f}")
    
    # Show overlap analysis
    both_machine = len(results_df[(results_df['is_machine_full'] == True) & (results_df['is_machine_conservative'] == True)])
    only_full = len(results_df[(results_df['is_machine_full'] == True) & (results_df['is_machine_conservative'] == False)])
    only_conservative = len(results_df[(results_df['is_machine_full'] == False) & (results_df['is_machine_conservative'] == True)])
    
    print(f"\nDETECTION OVERLAP:")
    print(f"  - Both methods detected as machine: {both_machine}")
    print(f"  - Only full version detected: {only_full}")
    print(f"  - Only conservative detected: {only_conservative}")
    
    return summary, results_df

if __name__ == "__main__":
    summary, results = main()