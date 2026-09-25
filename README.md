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

- [uv](https://docs.astral.sh/uv/) ;
- OCRmyPDF et Tesseract installés sur la machine ;
- les langues Tesseract `fra` et `eng` si la configuration par défaut est conservée.

Sous Windows, OCRmyPDF nécessite également ses dépendances système. Vérifiez l'installation avec :

```powershell
ocrmypdf --version
tesseract --list-langs
```

## Démarrage

```powershell
Copy-Item .env.example .env
uv sync
uv run clairdoc-server
```

L'API écoute par défaut uniquement sur `127.0.0.1:8787`. Pour permettre l'accès depuis un autre appareil du réseau local, définissez `CLAIRDOC_HOST=0.0.0.0`, ajoutez une règle de pare-feu limitée au réseau privé et utilisez une clé API forte. N'exposez pas directement ce port sur Internet.

- documentation interactive : `http://127.0.0.1:8787/docs`
- état public : `GET /api/v1/health`
- en-tête protégé : `X-ClairDoc-Key: votre-cle`

Exemple d'envoi :

```powershell
$headers = @{ "X-ClairDoc-Key" = "votre-cle" }
$project = Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8787/api/v1/projects `
  -Headers $headers `
  -ContentType "application/json" `
  -Body '{"name":"Archives familiales"}'

curl.exe -X POST "http://127.0.0.1:8787/api/v1/ocr/jobs?project_id=$($project.id)" `
  -H "X-ClairDoc-Key: votre-cle" `
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
| `POST` | `/api/v1/projects` | Créer un projet local |
| `GET` | `/api/v1/projects/{id}` | Lire un projet |
| `POST` | `/api/v1/ocr/jobs` | Envoyer un PDF et créer un travail |
| `GET` | `/api/v1/ocr/jobs/{id}` | Lire l'état d'un travail |
| `GET` | `/api/v1/ocr/jobs/{id}/document` | Télécharger le PDF OCRisé |
| `GET` | `/api/v1/ocr/jobs/{id}/text` | Télécharger le texte extrait |

## Qualité

```powershell
uv run ruff check .
uv run pytest
```

Cette version constitue le socle OCR. L'indexation, les embeddings, la recherche sémantique et les appels LLM seront ajoutés dans des modules séparés afin de ne pas coupler le stockage documentaire au fournisseur d'IA.

