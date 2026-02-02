# Steps to run

## Install requirements in a virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```
## Run frontend

```
python3 -m http.server 8002 -d frontend
```

## Run backend
```
python3 backend/app.py
```

If you want to run both fronend and backend in the same directory. add `&` after running frontend, e.g.:
```
python3 -m http.server 8002 -d frontend &
python3 backend/app.py
```