#!/bin/sh
# Build this simulation app from the REAL app + the files in overlay/.
#   - python/ and assets/ are copied from software/ (identical code)
#   - the real python/main.py becomes python/app_main.py, wrapped by overlay/python/main.py
#     (adds the simulation controls)
#   - overlay/assets/ adds the "Simulation controls" page
#   - the dashboard is switched to AI-only (window.AI_ONLY)
# Run after any change in software/ or overlay/.
set -e
cd "$(dirname "$0")"
rm -rf python assets
cp -r ../../software/python python
cp -r ../../software/assets assets
rm -rf python/__pycache__
mv python/main.py python/app_main.py
cp overlay/python/main.py python/main.py
cp overlay/assets/* assets/
sed -i 's|<script src="app.js"></script>|<script>window.AI_ONLY = true;</script>\n  <script src="app.js"></script>\n  <script src="quick.js"></script>|' assets/index.html
echo "synced: real python/ + assets/ from software/, overlay applied, dashboard AI-only"
