import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vocapp.app import app

if __name__ == "__main__":
    # host="0.0.0.0" makes it accessible from your phone on the same WiFi
    app.run(debug=True, host="0.0.0.0", port=5000)
