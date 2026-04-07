# Tutoriel d'installation : Interface d'annotation YOLO

Ce guide explique comment configurer votre machine locale pour lancer l'interface d'annotation du projet **Compost Waste Yolo**. 

Suivez les étapes ci-dessous selon votre système d'exploitation (Windows ou Linux).

---

## 0. Prérequis : Installer Python et Pip
Avant de commencer, vous devez avoir Python **(version 3.8 ou supérieure)** installé sur votre machine. `pip` (le gestionnaire de paquets) est généralement inclus avec l'installation de Python.

### Sous windows
1. Téléchargez l'installateur officiel sur python.org.
2. Important : Lors de l'installation, cochez impérativement la case **"Add Python to PATH"** avant de cliquer sur "Install Now".
3. Vérifiez l'installation dans un terminal :
```bash
python --version
pip --version
```

### Sous Linux
1. Mettez à jour vos dépôts et installez Python :
```bash
sudo apt update
sudo apt install python3 python3-pip
```

2. Vérifiez l'installation :
```bash
python3 --version
pip3 --version
```

## 1. Cloner le dépôt GitHub
Sur GitHub, téléchargez l'archive du projet et décompressez la.

Puis rendez vous dans le dossier décompressé.

Ou sinon si vous voulez télécharger le projet depuis le terminal avec git :


Ouvrez votre terminal (ou l'Invite de commandes / PowerShell sous Windows) et exécutez les commandes suivantes pour télécharger le code et vous rendre dans le dossier du projet :

```bash
git clone https://github.com/TSResearch-hub/Compost_Waste_Yolo.git
cd Compost_Waste_Yolo
```

## 2. Créer et activer l'environnement virtuel

Il est fortement recommandé d'utiliser un environnement virtuel pour ne pas créer de conflits avec les autres installations Python de votre ordinateur.

Ouvrez votre terminal (ou l'Invite de commandes / PowerShell sous Windows) et exécutez les commandes suivantes :

### Sous Linux

**Créer l'environnement :**
```bash
python3 -m venv venv
```
**Activer l'environnement :**
```bash
source venv/bin/activate
```

### Sous Windows
**Créer l'environnement :**
```bash
python -m venv venv
```

**Activer l'environnement :**
```bash
venv\Scripts\activate
```
Une fois l'environnement activé, vous devriez voir (venv) s'afficher au tout début de la ligne de votre terminal.

## 3. Installer les dépendances
Maintenant que vous êtes dans l'environnement virtuel, vous devez installer toutes les bibliothèques nécessaires au fonctionnement du projet (listées dans le fichier `requirements.txt`).
Exécutez ces commandes :
```base
# (Optionnel mais recommandé) Mettre à jour pip
pip install --upgrade pip

# Installer les dépendances du projet
pip install -r requirements.txt
```

## 4. Lancer l'application
Une fois l'installation terminée, il ne vous reste plus qu'à lancer l'interface Streamlit avec la commande suivante :

```bash
streamlit run app.py
```

L'application va se lancer et une page web devrait s'ouvrir automatiquement dans votre navigateur par défaut. Si ce n'est pas le cas, vous pouvez y accéder manuellement en allant à l'adresse suivante : http://localhost:8501