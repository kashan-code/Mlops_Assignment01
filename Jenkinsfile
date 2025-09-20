pipeline {
    agent any
    
    environment {
        DOCKER_HUB_REPO = 'your-dockerhub-username'
        PROJECT_NAME = 'ml-project'
        DOCKER_CREDENTIAL_ID = 'dockerhub-credentials'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Build Model Service') {
            steps {
                script {
                    // Build Docker image for ML model service
                    def modelImage = docker.build("${DOCKER_HUB_REPO}/${PROJECT_NAME}-model:${BUILD_NUMBER}")
                }
            }
        }
        
        stage('Build Flask API') {
            steps {
                script {
                    // Build Docker image for Flask API
                    def apiImage = docker.build("${DOCKER_HUB_REPO}/${PROJECT_NAME}-api:${BUILD_NUMBER}", "-f Dockerfile.api .")
                }
            }
        }
        
        stage('Test Containers') {
            steps {
                script {
                    // Run container tests
                    sh '''
                        docker run --rm ${DOCKER_HUB_REPO}/${PROJECT_NAME}-model:${BUILD_NUMBER} python -m pytest tests/
                        docker run --rm ${DOCKER_HUB_REPO}/${PROJECT_NAME}-api:${BUILD_NUMBER} python -m pytest tests/
                    '''
                }
            }
        }
        
        stage('Push to Docker Hub') {
            steps {
                script {
                    docker.withRegistry('https://registry.hub.docker.com', DOCKER_CREDENTIAL_ID) {
                        // Push model image
                        def modelImage = docker.image("${DOCKER_HUB_REPO}/${PROJECT_NAME}-model:${BUILD_NUMBER}")
                        modelImage.push()
                        modelImage.push("latest")
                        
                        // Push API image
                        def apiImage = docker.image("${DOCKER_HUB_REPO}/${PROJECT_NAME}-api:${BUILD_NUMBER}")
                        apiImage.push()
                        apiImage.push("latest")
                    }
                }
            }
        }
        
        stage('Deploy to Staging') {
            steps {
                script {
                    // Deploy to staging environment
                    sh '''
                        docker-compose -f docker-compose.staging.yml down
                        docker-compose -f docker-compose.staging.yml pull
                        docker-compose -f docker-compose.staging.yml up -d
                    '''
                }
            }
        }
    }
    
    post {
        success {
            emailext (
                to: "${env.ADMIN_EMAIL}",
                subject: "✅ Jenkins Build Successful - ${PROJECT_NAME} #${BUILD_NUMBER}",
                body: """
                <h2>Build Completed Successfully!</h2>
                <p><strong>Project:</strong> ${PROJECT_NAME}</p>
                <p><strong>Build Number:</strong> ${BUILD_NUMBER}</p>
                <p><strong>Git Commit:</strong> ${GIT_COMMIT}</p>
                <p><strong>Branch:</strong> ${GIT_BRANCH}</p>
                
                <h3>Docker Images Pushed:</h3>
                <ul>
                    <li>${DOCKER_HUB_REPO}/${PROJECT_NAME}-model:${BUILD_NUMBER}</li>
                    <li>${DOCKER_HUB_REPO}/${PROJECT_NAME}-api:${BUILD_NUMBER}</li>
                </ul>
                
                <p>Build Details: <a href="${BUILD_URL}">${BUILD_URL}</a></p>
                """,
                mimeType: 'text/html'
            )
        }
        
        failure {
            emailext (
                to: "${env.ADMIN_EMAIL}",
                subject: "❌ Jenkins Build Failed - ${PROJECT_NAME} #${BUILD_NUMBER}",
                body: """
                <h2>Build Failed!</h2>
                <p><strong>Project:</strong> ${PROJECT_NAME}</p>
                <p><strong>Build Number:</strong> ${BUILD_NUMBER}</p>
                <p><strong>Git Commit:</strong> ${GIT_COMMIT}</p>
                <p><strong>Branch:</strong> ${GIT_BRANCH}</p>
                
                <p>Build Details: <a href="${BUILD_URL}">${BUILD_URL}</a></p>
                <p>Console Output: <a href="${BUILD_URL}/console">${BUILD_URL}/console</a></p>
                """,
                mimeType: 'text/html'
            )
        }
        
        always {
            // Clean up Docker images
            sh '''
                docker system prune -f
            '''
        }
    }
}
