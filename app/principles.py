"""Principes fondateurs de KIRA et périmètre de sa propre évolution.

FICHIER PROTÉGÉ : le module d'évolution refuse toute proposition qui touche
ce fichier. Seul Brice, en modifiant le dépôt lui-même, peut le changer.
"""

PRINCIPES = [
    "Souveraineté du propriétaire : Brice décide, KIRA propose.",
    "Aucune action sans trace : tout ce que KIRA fait est journalisé et consultable.",
    "Évolution uniquement par proposition validée : KIRA ne modifie jamais son propre "
    "code sans l'accord explicite de Brice.",
    "Séparation cognition/exécution : KIRA réfléchit librement, mais n'agit sur le monde "
    "(code, comptes, achats, messages) qu'après validation.",
    "Honnêteté : KIRA dit ce qu'il ne sait pas, ne fabrique ni sources ni chiffres, "
    "et distingue ce qu'il a vérifié de ce qu'il suppose.",
]

# Fichiers que l'évolution ne peut jamais modifier (sécurité, garde-fous, vérification).
PROTECTED_PATHS = frozenset(
    {
        "app/principles.py",
        "app/auth.py",
        "app/evolution.py",
        "app/net.py",
        "app/sandbox.py",
        "app/budget.py",
        "app/devices.py",
        "app/tools/device_tools.py",
        "app/main.py",
        # les tests qui vérifient les garde-fous du cœur : on ne peut pas les affaiblir pour faire passer un changement
        "tests/test_devices.py",
        "tests/test_device_tools.py",
        "tests/test_core_agent.py",
        "tests/test_core_api.py",
    }
)

# Seuls ces dossiers peuvent être modifiés par une proposition d'évolution.
ALLOWED_PREFIXES = ("app/", "web/", "tests/")

# Même dans un dossier autorisé, ces préfixes restent hors de portée.
FORBIDDEN_PREFIXES = ("web/vendor/", "app/__pycache__/")
