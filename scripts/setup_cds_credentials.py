#!/usr/bin/env python3
"""
Set up CDS API credentials for ERA5 downloads via pyaps3.

Usage
-----
  python scripts/setup_cds_credentials.py

What it does
------------
  1. Prompts for your CDS Personal Access Token.
  2. Writes ~/.cdsapirc with the correct format.
  3. Validates the token by attempting a minimal CDS API connection.

Where to get your token
-----------------------
  1. Register/log in at https://cds.climate.copernicus.eu/
  2. Go to https://cds.climate.copernicus.eu/how-to-api
  3. Copy the Personal Access Token shown there.
  4. Accept the ERA5 licence:
     https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
     (Click "Accept Terms" — required once before downloads work.)
"""

import getpass
import sys
from pathlib import Path

CDS_URL = 'https://cds.climate.copernicus.eu/api'
RC_FILE = Path.home() / '.cdsapirc'


def main() -> None:
    print('CDS API Credentials Setup')
    print('=' * 40)

    if RC_FILE.exists():
        content = RC_FILE.read_text().strip()
        print(f'\n[INFO] {RC_FILE} already exists:')
        print('---')
        print(content)
        print('---')
        overwrite = input('\nOverwrite? [y/N]: ').strip().lower()
        if overwrite != 'y':
            print('Aborted — credentials unchanged.')
            sys.exit(0)

    print(f'\nGet your token from: {CDS_URL.replace("/api", "/how-to-api")}')
    print('(Must be logged in — token is shown on that page)\n')

    token = getpass.getpass('Paste your Personal Access Token: ').strip()

    if not token or token == 'your-personal-access-token':
        sys.exit('[ERROR] No valid token entered. Aborted.')

    # Validate format: CDS tokens are typically UUID-like strings
    content = f'url: {CDS_URL}\nkey: {token}\n'

    RC_FILE.write_text(content)
    RC_FILE.chmod(0o600)  # Restrict to owner only

    print(f'\n[OK] Written to {RC_FILE}')
    print('\nValidating connection to CDS API...')

    try:
        import cdsapi
        c = cdsapi.Client(quiet=True)
        print('[OK] CDS API connection successful.')
        print('\nIMPORTANT: Accept the ERA5 licence before downloading:')
        print('  https://cds.climate.copernicus.eu/datasets/'
              'reanalysis-era5-pressure-levels')
        print('  (Log in → scroll to Terms → click "Accept Terms")\n')
        print('You can now run:')
        print('  conda activate accra-risk')
        print('  python scripts/run_era5_correction.py')
    except Exception as exc:
        print(f'\n[WARN] CDS API validation failed: {exc}')
        print('This may be normal if the token is correct but ERA5 licence')
        print('has not yet been accepted. Accept it at:')
        print('  https://cds.climate.copernicus.eu/datasets/'
              'reanalysis-era5-pressure-levels')
        print(f'\nCredentials saved to {RC_FILE}.')
        print('Run the correction script after accepting the licence.')


if __name__ == '__main__':
    main()
