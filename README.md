#  GoodFood-Microservice 🤖

<p align="center">
  <img src="https://raw.githubusercontent.com/dquang0504/GoodFood-Microservice/main/assets/GoodFood-MS-cover.png" alt="GoodFood Microservice Banner" width="80%" />
</p>

<h3 align="center">
  **Tagline:** *AI-powered microservice for safe & intelligent e-commerce experiences.*
</h3>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-blue" /></a>
  <a href="https://flask.palletsprojects.com/"><img src="https://img.shields.io/badge/Flask-2.3-lightgrey" /></a>
  <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.x-orange" /></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-ready-blue" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" /></a>
</p>

---

## 📋 Overview

GoodFood-Microservice is a dedicated **AI service** for the [GoodFood](https://github.com/dquang0504/GoodFood-BE) ecosystem.  
It provides intelligent content moderation and analysis to ensure a safe, user-friendly environment for online food reviews and interactions.

Key features include:
- 🛡️ **Toxic speech filtering** – detect and block abusive language  
- 😀 **Sentiment analysis** – classify reviews as positive, negative, neutral, or mixed  
- 🚫 **NSFW content detection** – prevent inappropriate content uploads  
- ⚡ **Decoupled architecture** – exposed via REST APIs for modularity  
- 🐳 **Dockerized deployment** – easy integration into the main system  

---

## 📂 Project Structure
```bash
GoodFood-Microservice/
├── main.py/            # Flask app with routes & controllers
├── configs/            # Configurations & environment settings
├── tests/              # Unit and integration tests
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

* Language: Python

* Framework: Flask

* Machine Learning: TensorFlow, Hugging Face Transformers

* NLP & Moderation: Custom pipelines for toxicity, sentiment, and NSFW detection

* API: REST API

* Deployment: Docker

* Testing: Pytest

---

## 🚀 Deployment

Optimized for containerized environments:
```bash
# Build the Docker image
docker build -t goodfood-microservice .

# Run the container
docker run -p 5000:5000 goodfood-microservice
```

The service will be available at: http://localhost:5000/api/...

---

## 📖 API Documentation

Default Swagger docs available at:
/swagger/ (to be added)

Or test endpoints via Postman collection:
➡️ GoodFood Microservice API Docs (coming soon)

## 🤝 Contributing

We welcome contributions!

Fork the repo

Create a new branch (feature/my-feature)

Commit changes (git commit -m 'Add feature')

Push the branch & open a Pull Request

Please follow best practices for Python + Flask + NLP development.

## 📜 License

Distributed under the MIT License.
See LICENSE
 for more information.
