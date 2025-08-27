# Fuzzy AMD - Call Transcript Machine Detection

## Overview

Automated system to analyze call transcripts and identify machine-answered calls using fuzzy pattern matching. Designed for ElevateNow calls under 30 seconds.

## Pipeline

```
2,224 Initial Calls
    ↓ Remove Empty Transcripts
2,043 Non-Empty Calls (91.9%)
    ↓ Remove Single Utterances  
1,020 Multi-Utterance Calls (45.8%)
    ↓ Remove Assistant-Only Calls
753 Calls with User Responses (33.9%)
    ↓ Fuzzy Machine Detection
165 Machine Calls (21.9%) + 588 Human Calls (78.1%)
```

## Detection Logic

### Machine Patterns Detected

**Voicemail Messages**
- "voicemail", "voice mail"
- "your call has been forwarded to voicemail"
- "the person you are trying to reach is not available"
- "at the tone", "at the turn" (transcription error)
- "please record your message"
- "when you have finished recording"

**IVR Menu Systems**
- "press 1", "press 2", etc.
- "for sales press", "for support press"
- "please select an option"

**System Messages** 
- "not in service", "invalid number"
- "this number has been disconnected"
- "all circuits are busy now"

**Callback Requests + Numbers**
- "call me back" + phone numbers
- "you can reach me at" + digits

### Matching Strategy

1. **Direct substring match** (100% confidence)
2. **Fuzzy matching** (70%+ similarity for transcription errors)
3. **Word-based matching** (70%+ word overlap for partial phrases)

## Usage

```bash
python3 filter.py
```

**Requirements:**
- Python 3.x
- pandas, difflib
- `transcripts/` folder with call data
- `dict.json` with detection patterns

**Generated Files:**
- `amd_fuzzy_results.csv` - Complete analysis results
- `machine_detected_call_ids.txt` - List of all machine call IDs

## Results Summary

- **Total Processed**: 753 calls with user responses
- **Machine Detected**: 165 calls (21.9%)
- **Human Detected**: 588 calls (78.1%)
- **Detection Method**: Fuzzy pattern matching with high accuracy

## Key Features

✅ **Robust transcription error handling** ("tone" ↔ "turn")  
✅ **Multiple spelling variations** ("voicemail" ↔ "voice mail")  
✅ **Partial phrase matching** for incomplete transcripts  
✅ **Multi-language support** (English + Hindi patterns)  
✅ **Real-time filtering pipeline** with clear statistics  

## File Structure

```
├── filter.py                    # Main analysis script
├── dict.json                    # Detection patterns dictionary
├── transcripts/                 # Call transcript directories
├── amd_fuzzy_results.csv        # Analysis results
├── machine_detected_call_ids.txt # Machine call IDs list
└── README.md                    # This documentation
```

This system provides accurate machine detection with robust handling of real-world transcription challenges.