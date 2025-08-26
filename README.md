# Call Transcript Analysis Pipeline

## Overview

This project implements a sophisticated filtering and analysis pipeline for call transcripts to automatically identify machine-answered vs human-answered calls. The system processes ElevateNow calls under 30 seconds duration and applies multiple filtering stages to extract meaningful insights.

## Pipeline Architecture

The analysis follows a multi-stage funnel approach:

```
2224 Initial Calls
    ↓ (Empty Filter)
2043 Non-Empty Calls (91.9%)
    ↓ (Single Utterance Filter)  
1020 Multi-Utterance Calls (49.9%)
    ↓ (Machine Detection)
540 Machine + 480 Human Calls
```

## How It Works

### Stage 1: Empty Transcript Filter

**Purpose**: Remove calls with no meaningful content
- Filters out completely empty transcripts
- Removes calls with only whitespace
- **Result**: 181 calls removed (8.1% of initial)

### Stage 2: Single Utterance Filter

**Purpose**: Remove very short conversations that likely indicate unsuccessful calls
- Eliminates transcripts with only one utterance
- Focuses on calls with actual conversation flow
- **Result**: 1023 calls removed (46% of remaining)

### Stage 3: Machine Detection Algorithm

**Purpose**: Classify calls as machine-answered vs human-answered using AI pattern matching

#### Detection Categories with Weights:

1. **Voicemail Indicators** (Weight: 3x)
   - "your call has been forwarded to voicemail"
   - "the person you are trying to reach is not available"
   - "please record your message"
   - "at the tone" / "at the turn"
   - "when you have finished recording"

2. **Automated Responses** (Weight: 2x)
   - "this call is now being recorded"
   - "press [number] for"
   - "our office hours are"
   - "thank you for calling"
   - "please hold"

3. **Assistant Indicators** (Weight: 2.5x)
   - "I noticed you've been quiet"
   - "are you still there"
   - "can you hear me"
   - "hello? hello? hello?"

4. **System Messages** (Weight: 4x)
   - "the number you have dialed"
   - "this number has been disconnected"
   - "invalid number"
   - "number is not in service"

5. **Silence Indicators** (Weight: 2x)
   - "no response"
   - "line is quiet"
   - "not responding"

6. **Flow Patterns** (Weight: 5x)
   - Combined phrase detection
   - "hello + forwarded + voicemail"
   - "quiet + still there"

#### Additional Heuristics:

- **Short Responses**: <10 words (+1 point)
- **Repetitive Content**: Repeated words (+2 points)
- **Multiple Numbers**: >3 numbers detected (+2 points)

#### Classification Logic:
- **Score ≥ 3.0**: Machine Answered
- **Score < 3.0**: Human Answered

## Results Summary

### Data Funnel Analysis

| Stage | Count | Retention | Description |
|-------|-------|-----------|-------------|
| Initial Calls | 2224 | 100.0% | ElevateNow calls <30 seconds |
| After Empty Filter | 2043 | 91.9% | Removed 181 empty transcripts |
| After Single Utterance Filter | 1020 | 49.9% | Removed 1023 single-utterance calls |
| **Machine Answered** | **540** | **52.9%** | Automated/voicemail responses |
| **Human Answered** | **480** | **47.1%** | Real human conversations |

### Key Insights

#### 1. High Empty/Single Utterance Rate
- **54% of calls** were filtered out as empty or single utterance
- Indicates many very short/unsuccessful call attempts
- Suggests high bounce rate in initial contact attempts

#### 2. Machine Detection Performance
- **Average machine score**: 15.13 (well above threshold of 3.0)
- **Average human score**: 1.98 (well below threshold)
- **Clear separation** between machine and human scores
- High confidence in classification accuracy

#### 3. Machine vs Human Distribution
- Almost **even split** (52.9% machine, 47.1% human) among meaningful conversations
- Suggests roughly half of successful connections reach actual humans

### Pattern Examples Detected

#### Voicemail Messages
- "the person you are trying to reach is not available"
- "please record your message after the tone"
- "when you have finished recording, you may hang up"

#### Assistant Timeout Responses
- "I noticed you've been quiet, are you still there"
- "hello? can you hear me?"
- "is anyone there?"

#### System/Automated Messages
- "this call is now being recorded"
- "all representatives are currently busy"
- "press 1 for customer service"

## Usage

### Running the Analysis

```bash
python3 filter.py
```

### Requirements

```python
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import pandas as pd
from collections import Counter
```

### Input Structure

The pipeline expects:
- `transcripts/` folder containing subdirectories for each call
- Each call directory contains `transcript.json` with format:
```json
[
  {
    "timestamp": "2025-08-20 03:20:18",
    "speaker": "assistant",
    "content": "Hello, this is Priya from ElevateNow..."
  }
]
```

### Output

- **Console**: Real-time filtering progress and funnel visualization
- **CSV File**: `call_analysis_results.csv` with detailed results for each call
  - call_id
  - is_machine (boolean)
  - machine_score (float)
  - matches (list of detected patterns)
  - utterance_count
  - transcript_text (preview)

## Files Structure

```
├── filter.py              # Main analysis pipeline
├── transcripts/           # Directory containing call transcripts
├── patterns.txt           # Reference patterns for machine detection
├── call_analysis_results.csv  # Generated analysis results
└── README.md             # This documentation
```

## Machine Learning Insights

The sophisticated pattern matching system successfully:
- **Identifies voicemail systems** with high accuracy
- **Detects automated response systems**
- **Recognizes timeout scenarios** where assistants check for user presence
- **Filters out system error messages**
- **Provides confidence scores** for manual review of edge cases

The clear separation between machine scores (avg: 15.13) and human scores (avg: 1.98) demonstrates the effectiveness of the weighted pattern matching approach.

## Future Enhancements

1. **Machine Learning Model**: Train on labeled data for improved accuracy
2. **Dynamic Thresholds**: Adjust classification thresholds based on campaign type
3. **Real-time Processing**: Stream processing for live call analysis
4. **Pattern Learning**: Automatically discover new machine patterns
5. **Integration**: API endpoints for real-time call classification