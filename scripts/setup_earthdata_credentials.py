"""
scripts/setup_earthdata_credentials.py
=======================================
Interactive helper to create a NASA Earthdata account and write
credentials to ~/.netrc so earthaccess can authenticate automatically.

Run this ONCE before any NASA data download:

    python scripts/setup_earthdata_credentials.py

What it does
------------
1. Checks whether ~/.netrc already contains Earthdata credentials.
2. If not, prompts for your username and password.
3. Writes the entry to ~/.netrc with permissions set to 0600.
4. Tests authentication via earthaccess.

Prerequisites
-------------
- A NASA Earthdata account (free):
  https://urs.earthdata.nasa.gov/users/new

- For GLDAS (GES DISC) you must also approve the following app in
  your Earthdata profile under "Applications > Authorized Apps":
  NASA GESDISC DATA ARCHIVE
  URL: https://disc.gsfc.nasa.gov/earthdata-login

"""

from __future__ import annotations

import getpass
import sys
import webbrowser
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import check_netrc, setup_logging, write_netrc

log = setup_logging()

EARTHDATA_MACHINE = "urs.earthdata.nasa.gov"
REGISTRATION_URL  = "https://urs.earthdata.nasa.gov/users/new"
GESDISC_APP_URL   = (
    "https://urs.earthdata.nasa.gov/approve_app"
    "?client_id=e2WVk8Pw6weeLUKZYOxvTQ"
)


def main() -> None:
    print("\n" + "=" * 65)
    print("  NASA Earthdata Credentials Setup")
    print("=" * 65)

    # Step 1: Check for existing credentials
    if check_netrc(EARTHDATA_MACHINE):
        print(f"\n✓  Credentials for {EARTHDATA_MACHINE} already found in ~/.netrc.")
        rewrite = input("  Re-enter credentials? [y/N]: ").strip().lower()
        if rewrite != "y":
            _test_auth()
            return

    # Step 2: Registration prompt
    print(
        "\nYou need a free NASA Earthdata account to download GRACE and GLDAS.\n"
        f"  Registration: {REGISTRATION_URL}\n"
    )
    has_account = input("Do you already have a NASA Earthdata account? [y/N]: ").strip().lower()
    if has_account != "y":
        print(f"\nOpening registration page: {REGISTRATION_URL}")
        webbrowser.open(REGISTRATION_URL)
        input(
            "\nComplete registration in your browser, then press ENTER to continue..."
        )

    # Step 3: Collect credentials
    print(f"\nEnter your NASA Earthdata credentials for {EARTHDATA_MACHINE}:")
    username = input("  Username: ").strip()
    password = getpass.getpass("  Password: ")

    if not username or not password:
        print("\nERROR: Username and password cannot be empty.")
        sys.exit(1)

    # Step 4: Write to ~/.netrc
    netrc_path = write_netrc(username, password, EARTHDATA_MACHINE)
    print(f"\n✓  Credentials written to: {netrc_path}")
    print("  File permissions set to 0600 (user read/write only).")

    # Step 5: GES DISC app approval reminder
    print(
        "\n⚠  IMPORTANT — For GLDAS downloads (GES DISC):\n"
        "   You must authorise the GES DISC application in your Earthdata profile.\n"
        f"  → {GESDISC_APP_URL}\n"
    )
    open_gesdisc = input("Open GES DISC authorisation page now? [Y/n]: ").strip().lower()
    if open_gesdisc != "n":
        webbrowser.open(GESDISC_APP_URL)
        input("Complete authorisation, then press ENTER to continue...")

    # Step 6: Test authentication
    _test_auth()


def _test_auth() -> None:
    print("\nTesting NASA Earthdata authentication ...")
    try:
        import earthaccess
        auth = earthaccess.login(strategy="netrc")
        if auth.authenticated:
            print("✓  Authentication successful!")
            print("\nYou are ready to download GRACE and GLDAS data.")
            print("Next step:  python scripts/download_grace_gldas.py")
        else:
            print("✗  Authentication failed. Check your username and password.")
            print(f"  Manage credentials at: https://{EARTHDATA_MACHINE}")
    except ImportError:
        print("  earthaccess not installed. Run: pip install earthaccess")


if __name__ == "__main__":
    main()
