# Human Pose Estimation API with CI/CD Pipeline

## 📖 About The Project
This project is a FastAPI-based web service for Human Pose Estimation. Beyond just the machine learning model, this repository demonstrates a modern DevOps workflow by implementing a complete, automated CI/CD pipeline using GitHub Actions, Docker, and a self-hosted runner.

## ✨ Key Features
* **Pose Estimation API:** Detects and estimates human body poses from input data.
* **FastAPI Backend:** Provides a fast, lightweight, and interactive REST API (includes Swagger UI).
* **Automated CI/CD:** Automatically builds, pushes, and deploys the Docker container whenever new code is pushed to the `main` branch.
* **Self-Hosted Runner Deployment:** Utilizes a local environment as a GitHub Actions self-hosted runner for seamless and direct deployment.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Framework:** FastAPI
* **Containerization:** Docker, Docker Hub
* **CI/CD:** GitHub Actions

## 🔄 CI/CD Workflow
1. **Trigger:** Code is pushed to the `main` branch.
2. **Build & Push:** GitHub Actions builds the Docker image and pushes it to Docker Hub using secure secrets.
3. **Deploy:** The self-hosted runner automatically pulls the latest image, stops the old container, and runs the updated service locally.
