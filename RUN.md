# Exécuter les huit exercices

1. Installer `python -m pip install -r requirements.txt` et Poppler (`brew install poppler` sur macOS).
2. Le fichier fourni `CJ.pdf` est inclus. Exécuter le notebook en `MODE='local'` pour l’extraction, le rendu, la recherche et les contrôles locaux.
3. Pour les résumés et les réponses multimodales, renseigner `OPENAI_API_KEY` dans `.env` et choisir `MODE='openai'`. Les modèles du support sont conservés : `gpt-5.6-luna` et `gpt-5.6-terra`, configurables via `SUMMARY_MODEL` et `ANSWER_MODEL` si l’accès API du compte diffère.
4. Le cache de résumés est stocké sous `work/summary-cache`. Les expériences sont exportées sous `results/`.
5. Pour le défi Library of Congress, définir `LCM_PATH` vers `LCM_2020_1112.pdf` : ce fichier n’est pas fourni. Source officielle : https://www.loc.gov/lcm/pdf/LCM_2020_1112.pdf . Le téléchargement automatique a renvoyé HTTP 403 ici ; enregistrer ce fichier manuellement dans le dossier si nécessaire. Il n’a pas été remplacé par un document différent.
6. Pour une expérience distante LangSmith, renseigner `LANGSMITH_API_KEY` et activer `RUN_LANGSMITH`. Ce drapeau crée le dataset du lab s’il n’existe pas et enregistre les exécutions.
7. Tests : `python -m unittest -v test_summary.py`.

Le mode local n’utilise pas de vision et ne génère pas de réponse PDF : les images sont représentées par leur texte extrait. Il ne faut pas interpréter ses résultats comme une ablation des modèles. Le backend TF-IDF remplace Chroma pour rendre la recherche testable localement ; le couple proxy/original et les règles de provenance sont conservés.

Les appels OpenAI peuvent être nombreux : résumés des éléments, dix résumés pour l’ablation des cinq pages, sweep, cinq questions, DPI et évaluateur d’entailment. Le cache évite de refaire les résumés inchangés. Les générations de réponses restent de nouveaux appels. La revue humaine des réponses réelles et la disponibilité des modèles restent à vérifier après activation.
