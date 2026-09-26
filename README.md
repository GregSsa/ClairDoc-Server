# ClairDoc Server

Serveur local de ClairDoc, prévu pour fonctionner sur un PC fixe. Il exécute l'OCR localement, conserve l'index documentaire sur disque et n'envoie à OpenAI que les extraits de texte nécessaires aux embeddings et aux réponses.

## Fonctionnalités actuelles

- API HTTP FastAPI documentée automatiquement ;
- authentification par clé partagée (`X-ClairDoc-Key`) ;
- stockage local en fichiers JSON, sans base de données ;
- projets et travaux OCR persistants ;
- file d'attente OCR avec reprise après redémarrage ;
- paramètres OCR compatibles avec les versions distribuées par Ubuntu (`--skip-text`) ;
- PDF OCRisé et texte sidecar téléchargeables ;
- limites de taille, validation PDF et journaux locaux ;
- traitement concurrent configurable, limité à un travail par défaut.
- liste persistante des projets et travaux OCR ;
- pause, reprise et relance des travaux en échec ;
- détection SHA-256 des PDF identiques dans un même projet ;
- extraction du texte des PDF OCRisés, découpage et embeddings OpenAI ;
- index sémantique JSON local, avec réutilisation des documents inchangés ;
- questions/réponses RAG avec extraits sources.
- ingestion PDF, TXT, Markdown, CSV, DOCX, XLSX, PPTX, EML et images ;
- citations PDF avec numéro de page et métadonnées locales ;
- recherche hybride embeddings + mots-clés ;
- proposition de classement par catégorie et année.
- estimation des tokens et du coût avant indexation ;
- indexation persistante en arrière-plan avec reprise après redémarrage ;
- nouvelles tentatives bornées pour les erreurs OpenAI temporaires ;
- sauvegardes ZIP des métadonnées et version du schéma local ;
- limitation locale du débit HTTP et TLS facultatif.
- bibliothèque documentaire par projet avec catégories, métadonnées et relations calculées localement.
- sélection persistante du modèle de réponse Luna, Terra ou Sol.
- conversations multiples et historique persistant par projet ;
- mémoire projet Markdown actualisée lors de l'indexation ;
- outils IA pour rechercher, catégoriser, relier, copier, déplacer ou mettre à la corbeille ;
- exploration contrôlée du dossier source et lecture bornée des fichiers texte ;
- actions d'écriture limitées au dossier source, soumises à autorisation et journalisées.

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

Pour chiffrer une connexion réseau, renseignez ensemble `CLAIRDOC_TLS_CERTFILE` et `CLAIRDOC_TLS_KEYFILE`, puis utilisez une URL `https://` dans l'application. La limite par défaut est de 600 requêtes par minute et par adresse cliente ; elle se règle avec `CLAIRDOC_API_REQUESTS_PER_MINUTE`.

- documentation interactive : `http://127.0.0.1:8787/docs`
- état public : `GET /api/v1/health`
- en-tête protégé : `X-ClairDoc-Key: votre-cle`

### Les deux clés n'ont pas le même rôle

- `CLAIRDOC_API_KEY` est un secret local partagé entre l'application Tauri et ce serveur. Il empêche un autre appareil du réseau d'utiliser l'API ClairDoc. Ce n'est pas une clé OpenAI.
- `OPENAI_API_KEY` est la clé du compte OpenAI utilisée pour les embeddings et les réponses. Elle reste uniquement dans le fichier `.env` du serveur. Elle ne doit jamais être placée dans l'application Tauri ni enregistrée dans Git.

Par défaut, les embeddings utilisent `text-embedding-3-small` avec 512 dimensions et les réponses utilisent `gpt-6-luna`. Ces valeurs peuvent être changées dans `.env`.

### Embeddings locaux et renommage

Dans les paramètres de l'application, choisissez **Calcul des embeddings → Local sur le serveur**.
FastEmbed utilise `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` sur CPU
(384 dimensions). Le premier calcul télécharge le modèle depuis Hugging Face (environ
220 Mo annoncés), puis le cache est conservé dans `data/models`. Les calculs suivants
ne nécessitent pas d'appel OpenAI. Exécutez `uv sync` après cette mise à jour.
Le choix est persisté dans `data/runtime.json` ; `CLAIRDOC_EMBEDDING_PROVIDER=local`
permet aussi de définir la valeur initiale.

Relancez l'indexation des projets pour utiliser le nouveau fournisseur. Les anciens index
restent interrogeables avec leur propre modèle et fournisseur ; leurs vecteurs ne sont
jamais mélangés avec les nouveaux. Le coût API des embeddings locaux est nul, mais
leur calcul utilise le CPU et prend du temps. Les morceaux locaux sont plus courts
pour limiter la troncature du modèle.

Le LLM de conversation et le renommage automatique utilisent toujours `OPENAI_API_KEY`.
L'option **Autoriser l'IA à proposer de nouveaux noms** est désactivée par défaut.
Quand elle est activée, le serveur transmet au LLM jusqu'à 6 000 caractères de texte
par document, par lots de dix, et prépare uniquement des propositions. Aucun fichier
n'est renommé avant validation du plan. Sans cette option, les noms sont conservés,
avec suffixe en cas de doublon dans le même dossier cible.

L'assistant dispose de `read_project_document` pour lire le texte extrait d'un PDF
par nom, chemin ou identifiant, même si son dossier source n'est pas accessible au serveur.
Une analyse/OCR terminée est nécessaire. `rename_document` conserve l'extension et
le dossier et exige l'autorisation d'écriture. Le renommage effectif nécessite que le
dossier source soit accessible au serveur (même chemin ou montage réseau).

L'estimation préalable utilise approximativement un token pour quatre caractères. Elle sert de garde-fou, pas de facture exacte. Le tarif de référence est configurable avec `CLAIRDOC_EMBEDDING_PRICE_PER_MILLION_USD` afin de pouvoir l'actualiser sans changer le code. `CLAIRDOC_MAX_INDEX_TOKENS` bloque une tâche qui dépasserait la limite choisie.

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
├── schema.json
├── projects/<id>.json
├── jobs/<id>/job.json
├── jobs/<id>/input.pdf
├── jobs/<id>/output.pdf
├── jobs/<id>/output.txt
├── indexes/<projet-id>.json
├── index-tasks/<id>.json
├── conversations/<projet-id>/<conversation-id>.json
├── memories/<projet-id>.md
├── action-logs/<projet-id>/<date>.json
├── backups/clairdoc-metadata-<date>.zip
├── prompts/rag-system.txt
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
| `PATCH` | `/api/v1/projects/{id}` | Renommer le projet ou associer son dossier source |
| `DELETE` | `/api/v1/projects/{id}` | Supprimer les données ClairDoc du projet |
| `GET` | `/api/v1/projects` | Lister les projets existants |
| `GET` | `/api/v1/projects/{id}/documents` | Lister les documents, métadonnées et relations du projet |
| `GET` | `/api/v1/runtime` | Lire les modèles et paramètres non secrets du serveur |
| `PATCH` | `/api/v1/runtime` | Sélectionner le modèle de réponse autorisé |
| `GET` | `/api/v1/projects/{id}/ocr/jobs` | Lister les travaux OCR d'un projet |
| `POST` | `/api/v1/projects/{id}/ocr/pause` | Mettre la file du projet en pause |
| `POST` | `/api/v1/projects/{id}/ocr/resume` | Reprendre la file du projet |
| `POST` | `/api/v1/ocr/jobs` | Envoyer un PDF et créer un travail |
| `GET` | `/api/v1/ocr/jobs/{id}` | Lire l'état d'un travail |
| `POST` | `/api/v1/ocr/jobs/{id}/retry` | Relancer un travail en échec |
| `POST` | `/api/v1/document/jobs` | Envoyer un document pris en charge |
| `GET` | `/api/v1/ocr/jobs/{id}/document` | Télécharger le PDF OCRisé |
| `GET` | `/api/v1/ocr/jobs/{id}/text` | Télécharger le texte extrait |
| `GET` | `/api/v1/projects/{id}/index/estimate` | Estimer les tokens et le coût de l'indexation |
| `POST` | `/api/v1/projects/{id}/index/jobs` | Démarrer une indexation persistante |
| `GET` | `/api/v1/index/jobs/{id}` | Suivre une indexation persistante |
| `POST` | `/api/v1/index/jobs/{id}/retry` | Relancer une indexation en échec |
| `POST` | `/api/v1/projects/{id}/ask` | Poser une question sur l'index du projet |
| `GET/POST` | `/api/v1/projects/{id}/conversations` | Lister ou créer les conversations du projet |
| `GET/DELETE` | `/api/v1/projects/{id}/conversations/{conversation}` | Lire ou supprimer une conversation |
| `POST` | `/api/v1/projects/{id}/conversations/{conversation}/messages` | Envoyer un message à l'assistant du projet |
| `GET` | `/api/v1/projects/{id}/memory` | Lire la mémoire Markdown du projet |
| `POST` | `/api/v1/projects/{id}/organization/plan` | Préparer un plan de classement |
| `POST` | `/api/v1/maintenance/backups` | Créer une sauvegarde ZIP des métadonnées |

Les métadonnées automatiques (catégorie, date, organisme, personnes et montants) sont calculées localement à partir du texte extrait. Elles constituent des propositions à vérifier, pas des données administratives garanties.

L'assistant utilise les appels de fonctions de l'API Responses. L'historique de référence reste
stocké localement ; les réponses OpenAI sont créées avec `store=false`. Les recherches de fichiers
sont en lecture seule. Une modification n'est exécutée que lorsque l'application transmet
`allow_write_actions=true`. Les chemins sont résolus sous le dossier source du projet et une
suppression déplace le fichier dans `.clairdoc/trash` au lieu de l'effacer définitivement.

## Qualité

### Diagnostics d'import

Lors de l'indexation, les anciens imports sont réextraits une fois si nécessaire.
Un PDF sans texte exploitable après le passage normal est repris avec `--redo-ocr`,
sans redressement ni rastérisation forcée. Les tableaux Word sont lus et leurs images
PNG/JPEG/TIFF/BMP/WebP sont analysées par Tesseract localement. Les originaux sont conservés.

Si l'OCR réussit sans produire de texte (photo, dessin, plan…), le document est tout
de même indexé : son embedding est calculé uniquement sur son nom. L'index indique
`indexing_mode=name_only`, l'application affiche **Indexé · nom seul**, et le contexte
de l'assistant précise que le contenu est inconnu. Ce mode ne permet pas une recherche
sur le contenu de l'image. Une erreur technique d'OCR reste un échec explicite.
Le résumé compte les documents en mode nom seul. Les anciennes importations peuvent
être reprises sans réimport en cliquant sur **Mettre à jour l'index** après redémarrage.
Le coût est réestimé après récupération OCR avant calcul des embeddings ; l'estimation
initiale peut être plus basse si les textes n'étaient pas encore récupérés.


Les erreurs sont enregistrées dans `data/jobs/<id>/job.json` (champ `error`) et
affichées dans **Documents en échec** dans l'application. La sortie complète d'OCRmyPDF
est conservée dans `data/jobs/<id>/ocr.log` pour les nouveaux traitements terminés.
Le serveur produit un PDF standard (`--output-type pdf`), sans conversion PDF/A,
pour préserver l'espace colorimétrique et éviter `ColorConversionNeededError`.
Le texte est ensuite extrait de toutes les pages du PDF produit, ainsi que des valeurs
des champs de formulaire ; le sidecar OCR seul ne contient pas le texte des pages sautées.


```bash
uv run ruff check .
uv run pytest
```

Le prompt système est créé dans `data/prompts/rag-system.txt` au premier appel. Il peut être adapté sans modifier le code, puis sera repris lors des questions suivantes.
