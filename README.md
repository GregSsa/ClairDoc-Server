# ClairDoc Server

Serveur local de ClairDoc, prévu pour fonctionner sur un PC fixe. Cette première version reçoit des PDF depuis l'application Tauri, les place dans une file d'attente et lance OCRmyPDF/Tesseract sans envoyer les documents vers un service externe.

## Fonctionnalités actuelles

- API HTTP FastAPI documentée automatiquement ;
- authentification par clé partagée (`X-ClairDoc-Key`) ;
- stockage local en fichiers JSON, sans base de données ;
- projets et travaux OCR persistants ;
- file d'attente OCR avec reprise après redémarrage ;
- PDF OCRisé et texte sidecar téléchargeables ;
- limites de taille, validation PDF et journaux locaux ;
- traitement concurrent configurable, limité à un travail par défaut.

## Prérequis

- Linux ou WSL2 avec Ubuntu 22.04 ou une version plus récente ;
- [uv](https://docs.astral.sh/uv/) ;
- OCRmyPDF et Tesseract installés sur la machine ;
- les langues Tesseract `fra` et `eng` si la configuration par défaut est conservée.

Installation sous Ubuntu/WSL :

```bash
sudo apt update
sudo apt install -y ocrmypdf tesseract-ocr-fra tesseract-ocr-eng

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

ocrmypdf --version
tesseract --list-langs
```

## Démarrage

```bash
cp .env.example .env

# Générez une clé différente pour chaque installation, puis placez-la
# dans CLAIRDOC_API_KEY au sein du fichier .env.
openssl rand -hex 32

uv sync
uv run clairdoc-server
```

L'API écoute par défaut uniquement sur `127.0.0.1:8787`. Pour permettre l'accès depuis un autre appareil du réseau local, définissez `CLAIRDOC_HOST=0.0.0.0`, ajoutez une règle de pare-feu limitée au réseau privé et utilisez une clé API forte. N'exposez pas directement ce port sur Internet.

- documentation interactive : `http://127.0.0.1:8787/docs`
- état public : `GET /api/v1/health`
- en-tête protégé : `X-ClairDoc-Key: votre-cle`

### Les deux clés n'ont pas le même rôle

- `CLAIRDOC_API_KEY` est un secret local partagé entre l'application Tauri et ce serveur. Il empêche un autre appareil du réseau d'utiliser l'API ClairDoc. Ce n'est pas une clé OpenAI.
- `OPENAI_API_KEY` sera la clé du compte OpenAI. Elle devra rester uniquement dans le fichier `.env` du serveur lorsque le module LLM sera implémenté. Elle ne doit jamais être placée dans l'application Tauri ni enregistrée dans Git.

Pour cette première version, `OPENAI_API_KEY` n'est pas encore utilisée : le serveur ne fait que l'OCR local.

Exemple d'envoi :

```bash
CLAIRDOC_KEY="remplacez-par-votre-cle"

PROJECT_ID=$(curl --silent --request POST \
  http://127.0.0.1:8787/api/v1/projects \
  --header "X-ClairDoc-Key: $CLAIRDOC_KEY" \
  --header "Content-Type: application/json" \
  --data '{"name":"Archives familiales"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')

curl --request POST \
  "http://127.0.0.1:8787/api/v1/ocr/jobs?project_id=$PROJECT_ID" \
  -H "X-ClairDoc-Key: $CLAIRDOC_KEY" \
  -F "file=@document.pdf;type=application/pdf"
```

## Données locales

Par défaut, tout est conservé dans `./data` :

```text
data/
├── projects/<id>.json
├── jobs/<id>/job.json
├── jobs/<id>/input.pdf
├── jobs/<id>/output.pdf
├── jobs/<id>/output.txt
└── logs/clairdoc-server.log
```

Le dossier `data` et le fichier `.env` sont exclus de Git.

## API v1

| Méthode | Route | Rôle |
|---|---|---|
| `GET` | `/api/v1/health` | État du serveur et disponibilité OCR |
| `GET` | `/api/v1/connection` | Vérifier la clé et la connexion de l'application |
| `POST` | `/api/v1/projects` | Créer un projet local |
| `GET` | `/api/v1/projects/{id}` | Lire un projet |
| `POST` | `/api/v1/ocr/jobs` | Envoyer un PDF et créer un travail |
| `GET` | `/api/v1/ocr/jobs/{id}` | Lire l'état d'un travail |
| `GET` | `/api/v1/ocr/jobs/{id}/document` | Télécharger le PDF OCRisé |
| `GET` | `/api/v1/ocr/jobs/{id}/text` | Télécharger le texte extrait |

## Qualité

```bash
uv run ruff check .
uv run pytest
```

Cette version constitue le socle OCR. L'indexation, les embeddings, la recherche sémantique et les appels LLM seront ajoutés dans des modules séparés afin de ne pas coupler le stockage documentaire au fournisseur d'IA.
