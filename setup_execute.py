import os
import subprocess
import sys
import shutil

def install_requirements():
    """Install the required packages."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_streamlit_app():
    """Run the Streamlit app."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

def main():
    # Step 1: Install requirements
    print("Installing requirements...")
    install_requirements()

    # Step 2: Run the Streamlit app
    print("Running the Streamlit app...")
    run_streamlit_app()

if __name__ == "__main__":
    main()
