# Transcript Labeling Tool

A web-based interface for labeling conversation transcripts with four categories: Machine, Human, Discard, or Not Sure.

## Features

- **Interactive Web Interface**: Clean, modern UI for reviewing transcripts
- **Progress Tracking**: Visual progress bar showing completion status
- **Keyboard Shortcuts**: Quick labeling using number keys (1-4)
- **Navigation**: Forward/backward navigation through transcripts
- **CSV Export**: Export labeled data to CSV format
- **Auto-advance**: Automatically moves to next transcript after labeling

## Setup Instructions

### 1. Generate Transcript List
First, generate the list of available transcripts:

```bash
cd data_label
python3 generate_transcript_list.py
```

This will create `transcript-ids.json` with all available transcript IDs.

### 2. Start the Local Server
Start the HTTP server to serve the interface:

```bash
python3 server.py
```

The server will start on `http://localhost:8000`

### 3. Open the Interface
Open your web browser and navigate to:
```
http://localhost:8000
```

## Usage

### Labeling Options
Click the appropriate button or use keyboard shortcuts:

- **🤖 Machine** (Key: 1) - For automated/bot conversations
- **👤 Human** (Key: 2) - For human-to-human conversations  
- **🗑️ Discard** (Key: 3) - For conversations to be discarded
- **❓ Not Sure** (Key: 4) - When unsure about the category

### Navigation
- **Previous/Next Buttons**: Navigate between transcripts
- **Arrow Keys**: Use ← → keys for navigation
- **Auto-advance**: Interface automatically moves to next transcript after labeling

### Export Results
- Click **"Export CSV"** button at any time to download your labels
- CSV contains two columns: `Call_ID` and `Label`
- Unlabeled transcripts will show as "unlabeled" in the export

## File Structure

```
data_label/
├── index.html              # Main web interface
├── app.js                  # JavaScript application logic
├── server.py              # Local HTTP server
├── generate_transcript_list.py  # Script to generate transcript list
├── transcript-ids.json    # Generated list of transcript IDs
└── README.md             # This file
```

## Technical Details

### Data Format
- Transcripts are expected in `../transcripts/{id}/transcript.json`
- Each transcript contains an array of conversation messages
- Messages have: `timestamp`, `speaker`, `content` fields

### Browser Compatibility
- Modern browsers (Chrome, Firefox, Safari, Edge)
- JavaScript ES6+ support required
- Local file access handled via HTTP server

### CSV Output Format
```csv
"Call_ID","Label"
"00031634-197b-4088-bebe-29e162eb705a","machine"
"0004a2b3-6976-4061-87e3-8c70b0db3ed9","human"
"00147336-d97b-460b-b02a-bc899cad057c","discard"
...
```

## Troubleshooting

### Server Won't Start
- Check if port 8000 is available
- Try running with `sudo` if permission issues occur
- Verify Python 3 is installed

### Transcripts Not Loading
- Ensure transcript files exist in `../transcripts/` directory
- Check that `transcript-ids.json` was generated correctly
- Verify file permissions allow reading

### Browser Issues
- Enable JavaScript in your browser
- Try a different browser if issues persist
- Check browser console for error messages

## Keyboard Shortcuts Summary

| Key | Action |
|-----|--------|
| 1 | Label as Machine |
| 2 | Label as Human |
| 3 | Label as Discard |
| 4 | Label as Not Sure |
| ← | Previous transcript |
| → | Next transcript |

Press Ctrl+C in the terminal to stop the server when done.