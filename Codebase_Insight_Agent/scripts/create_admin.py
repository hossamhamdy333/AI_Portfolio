"""
Creates the one admin account this site has. There's no public /auth/register
route on purpose - visitors never get accounts at all, only you do, and only
by running this script directly against the database.

Run:
    python scripts/create_admin.py you@example.com
(it will prompt for a password, not take it as a plain command-line argument,
so it doesn't end up sitting in your shell history)
"""

import getpass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database import SessionLocal, init_db
from models import AdminUser
from auth import hash_password


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/create_admin.py you@example.com")
        sys.exit(1)

    email = sys.argv[1]
    password = getpass.getpass("Password (8+ characters): ")
    if len(password) < 8:
        print("Password must be at least 8 characters.")
        sys.exit(1)
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Passwords didn't match.")
        sys.exit(1)

    init_db()
    db = SessionLocal()

    existing = db.query(AdminUser).filter(AdminUser.email == email).first()
    if existing is not None:
        print(f"An admin account for {email} already exists.")
        sys.exit(1)

    admin = AdminUser(email=email, password_hash=hash_password(password))
    db.add(admin)
    db.commit()
    db.close()

    print(f"Admin account created for {email}. Log in at POST /auth/login.")


if __name__ == "__main__":
    main()
