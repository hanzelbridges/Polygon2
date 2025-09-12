## Setting POLYGON_API_KEY Permanently

To set your Polygon API key permanently on Windows (PowerShell):

1. Open your PowerShell profile for editing:
	```powershell
	notepad $PROFILE
	```
2. Add this line to the end of the file (replace with your actual key):
	```powershell
	$env:POLYGON_API_KEY = "your_actual_api_key_here"
	```
3. Save and close Notepad. Restart your terminal or run `. $PROFILE` to reload.

Now, every new PowerShell session will have the API key set automatically.

**Alternative:**
If you want to use a `.env` file and load it automatically in your Python scripts, you can install the `python-dotenv` package and add this to the top of your script:

```python
from dotenv import load_dotenv
load_dotenv()
```
Then create a `.env` file in your project root with:
```
POLYGON_API_KEY=your_actual_api_key_here
```
Polygon.io Minimal Client (Python)

This repo provides a tiny, dependencyâ€‘free Python client and CLI to fetch common data from Polygon.io using only the Python standard library.

Setup

- Prerequisites: Python 3.10+ and a Polygon API key.
- Set your API key in the `POLYGON_API_KEY` environment variable.

Windows PowerShell (current session only):

```
$env:POLYGON_API_KEY = "YOUR_KEY_HERE"
```

Windows PowerShell (persist for your user):

```
[System.Environment]::SetEnvironmentVariable('POLYGON_API_KEY','YOUR_KEY_HERE','User')
```

macOS/Linux (bash/zsh):

```
export POLYGON_API_KEY="YOUR_KEY_HERE"
```

Usage

Run commands directly via the CLI script in `src/`.

- Previous close for a ticker:

```
python src/cli.py prev-close AAPL
```

- Aggregates (candles) for a date range:

```
python src/cli.py aggs AAPL 1 day 2024-01-01 2024-01-31 --sort asc --limit 50000
```

- Daily open/close for a date:

```
python src/cli.py open-close AAPL 2024-01-12
```

- Last trade for a ticker:

```
python src/cli.py last-trade AAPL
```

Notes

- The client uses only the standard library (`urllib`), so no installs are required.
- All output is JSON printed to stdout. Pipe to a file or `jq` as needed.
- If you see an error about a missing API key, confirm `POLYGON_API_KEY` is set in your environment.
- Rate limits and data entitlements are enforced by Polygon.io based on your account.

Project Structure

- `src/polygon_client.py`: Minimal HTTP helpers and endpoint wrappers.
- `src/cli.py`: Commandâ€‘line interface built on top of the client.


Gapper CSV Columns

For a simple explanation of each column produced by the gapper analysis script, see  `docs/columns.md`. 

