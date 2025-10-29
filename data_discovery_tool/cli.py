import os
import sys
import streamlit.web.cli as stcli

def main():
    # Get the path to the app.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app.py")
    
    # Set up Streamlit arguments
    sys.argv = ["streamlit", "run", app_path, "--server.maxUploadSize=1024"]
    
    # Run Streamlit
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()