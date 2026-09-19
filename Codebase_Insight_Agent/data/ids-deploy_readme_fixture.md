# IDS Deploy — 2-Stage Cascade NIDS

## Architecture

```
Live traffic → NFStream → Preprocessing → Stage 1: Binary XGBoost
                                                  │
                                          BENIGN  │  ATTACK (conf > threshold)
                                            ↓     │       ↓
                                         logged   Stage 2: Attack-only XGBoost
                                                       ↓
                                               Attack name logged
                                                       ↓
                                          SQLite ← ids-engine
                                             ↓
                                          FastAPI (ids-api :8000)
                                             ↓
                                       Dashboard (ids-dashboard :80)
```

## Folder layout

```
ids-deploy/
├── docker-compose.yml
├── models/
│   ├── artefacts/
│   │   ├── imputer.pkl              ← SimpleImputer (fit on training data)
│   │   ├── corr_dropped_cols.pkl    ← list[str] of corr-dropped columns
│   │   ├── log_features.pkl         ← list[str] of features to log1p
│   │   ├── mi_selected_cols.pkl     ← list[str] of MI-selected column names
│   │   └── scaler.pkl               ← RobustScaler (fit on training data)
│   ├── xgb_binary.pkl               ← Model 1: binary classifier
│   ├── xgb_att_multiclass.pkl       ← Model 3: attack-only multiclass
│   └── le_att_multiclass.pkl        ← LabelEncoder for Model 3
├── engine/
│   ├── engine.py
│   ├── requirements.txt
│   └── Dockerfile
├── api/
│   ├── api.py
│   ├── requirements.txt
│   └── Dockerfile
└── dashboard/
    ├── index.html
    └── Dockerfile
```

## On the IDS server

```bash
# 1. Install Docker (if not already installed)
curl -fsSL https://get.docker.com | sh

# 2. Copy this folder to the server
scp -r ids-deploy/ root@<IDS-IP>:/opt/ids-deploy

# 3. Put your model files in /opt/ids-deploy/models/
#    Copy from Colab/Kaggle output directory

# 4. Check your mirrored interface name
ip link show
# Look for the interface receiving mirrored GNS3 traffic (eth0, ens3, ens4 ...)

# 5. Set the interface in docker-compose.yml if not eth0
#    Change: IFACE=eth0  →  IFACE=ens3

# 6. Build and start
cd /opt/ids-deploy
docker compose up -d --build

# 7. Check engine is running
docker logs -f ids-engine
# You should see: "All artefacts loaded." then "Listening on interface: ethX"
# Attacks appear as: ATTACK [DDoS] conf=0.87 ...

# 8. Open dashboard from ANY machine on the network
http://<IDS-SERVER-IP>           ← dashboard (enter server IP in the input field)
http://<IDS-SERVER-IP>:8000      ← raw API
http://<IDS-SERVER-IP>:8000/docs ← FastAPI Swagger docs
```

## Dashboard access

The dashboard fetches data from `http://<server-ip>:8000`.
- If you open it **from the IDS server itself**, it auto-detects the IP.
- If you open it **from another machine** (laptop, GNS3 router), enter the IDS server IP in the input field and click Connect. The IP is saved in the browser for next time.

## Configuration

| Variable    | Default | Description |
|-------------|---------|-------------|
| IFACE       | eth0    | Network interface to capture from |
| THRESHOLD   | 0.3     | Binary classifier confidence cutoff. Lower = fewer missed attacks, more false positives. 0.3 is recommended for IDS. |
| BATCH_SIZE  | 50      | Flows buffered before each SQLite write (reduces I/O) |
| DB_PATH     | /data/ids.db | SQLite database path (shared volume) |
| MODELS_DIR  | /models | Path to model pkl files |

## 2-Stage cascade logic

- **Stage 1** (`xgb_binary.pkl`): classifies flow as Benign or Attack
  - If `P(attack) < THRESHOLD` → labelled BENIGN, logged, done
  - If `P(attack) >= THRESHOLD` → passed to Stage 2
- **Stage 2** (`xgb_att_multiclass.pkl`): identifies the specific attack type
  - Returns one of the 20 attack class names (DDoS, DoS, XSS, SQLi, ...)
