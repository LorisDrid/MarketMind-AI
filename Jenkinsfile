pipeline {
    agent any

    options {
        // Garde seulement les 10 derniers builds pour ne pas saturer le disque
        buildDiscarder(logRotator(numToKeepStr: '10'))
        // Affiche des couleurs dans la console (si plugin installé)
        ansiColor('xterm')
    }

    stages {
        stage('🛠️ Setup Environment') {
            steps {
                echo 'Vérification de l environnement Python...'
                bat 'python --version'
                // On s'assure que les dépendances sont à jour
                bat 'pip install -r requirements.txt --quiet'
            }
        }

        stage('⚡ GPU & Hardware') {
            steps {
                echo 'Validation de la RTX 5070...'
                bat 'nvidia-smi'
                // Test CUDA que nous avons validé ensemble
                bat 'C:\\Users\\loris\\Documents\\Programming\\MarketMind-AI\\.venv\\Scripts\\python.exe -c "import torch; print(\'CUDA OK :\', torch.cuda.is_available())"'
            }
        }

        stage('🧠 Sentiment IA') {
            steps {
                echo 'Lancement du benchmark Llama3 (Ollama)...'
                // Ici on lance ton script qui a donné 92.6% de précision
                bat 'C:\\Users\\loris\\Documents\\Programming\\MarketMind-AI\\.venv\\Scripts\\python.exe scripts/benchmark_sentiment.py'
            }
        }

        stage('⚖️ Risk Management') {
            steps {
                echo 'Vérification des règles de sécurité (Stop-loss, Balance)...'
                // Test des limites financières pour éviter que le bot ne fasse n'importe quoi
                bat 'C:\\Users\\loris\\Documents\\Programming\\MarketMind-AI\\.venv\\Scripts\\python.exe tests/test_risk_manager.py'
            }
        }
    }

    post {
        success {
            echo '✅ Tout est vert ! Le bot est prêt pour le déploiement.'
        }
        failure {
            echo '❌ Échec détecté. Vérifier les logs avant toute opération de trading.'
        }
    }
}