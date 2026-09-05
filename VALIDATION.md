# Validation locale

- CJ.pdf fourni : 14 pages réellement extraites et rendues à 200 DPI.
- Notebook exécuté en mode local : index, jointure vers les originaux, persistance, sweep de k et préparation des cinq audits.
- Huit tests passent : limites des preuves envoyées, citations vides ou invalides, cohérence des citations par affirmation, cache, routage, persistance et association des valeurs numériques aux bonnes entreprises.
- Référence numérique relue sur l’image de la page 5 : MongoDB 14,6× / 17 % ; Cloudflare 13,4× / 28 % ; Datadog 13,1× / 19 %.

Les ablations de prompts et de DPI, les réponses vision, l’entailment et l’expérience distante LangSmith sont codés mais non exécutés. Le fichier Library of Congress n’était pas fourni ; sa source officielle a été trouvée, mais son téléchargement automatique a retourné HTTP 403. L’exercice accepte le chemin du PDF via LCM_PATH.

Aucun classement de prompts, score de juge, seuil de lisibilité ou réponse de modèle n’est inventé. Les fichiers results/ distinguent retrieval_only et not_run. Les valeurs du tableau concernent le document de novembre 2023.
