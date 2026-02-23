<div align="center">

# 🛡️ SafeSight

### _formerly known as **ViolenceSense**_

<br/>

## 🏆 1st Place — IBM SkillsBuild AI Innovation Challenge 2026

**Organized by CSRBOX® at i-Hub Gujarat · Among 35+ shortlisted teams**

<br/>

<img src="https://img.shields.io/badge/🥇_1st_Place-IBM_SkillsBuild_AI_Innovation_Challenge_2026-FFD700?style=for-the-badge" alt="Winner Badge"/>

<br/><br/>

> **AI-powered surveillance system built on a GPU-trained MobileNetV2-LSTM architecture, optimized for real-time CPU deployment — delivering 95%+ violence detection accuracy with confidence scoring.**

<br/>

[![Report](https://img.shields.io/badge/📄_Project_Report-View_PDF-2196F3?style=for-the-badge)](https://drive.google.com/file/d/1qN-nHZitUcRVDV3NqruWSkXukgkPH05f)
[![Dataset](https://img.shields.io/badge/📊_Dataset-Kaggle-20BEFF?style=for-the-badge)](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
[![Colab](https://img.shields.io/badge/🔬_Training_Notebook-Google_Colab-F9AB00?style=for-the-badge)](https://colab.research.google.com/drive/1YkII0c6DETnqfUW-YLNha6ShCfyk2gWE?ts=697f3c76)

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat-square&logo=next.js&logoColor=white)](https://nextjs.org)
[![Node.js](https://img.shields.io/badge/Node.js-18+-339933?style=flat-square&logo=node.js&logoColor=white)](https://nodejs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=flat-square&logo=mongodb&logoColor=white)](https://mongodb.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

<br/>

[The Problem](#-the-problem) •
[Our Approach](#-our-approach) •
[Key Features](#-key-features) •
[Architecture](#️-architecture) •
[Tech Stack](#️-tech-stack) •
[Getting Started](#-getting-started) •
[Model](#-model-details) •
[Team](#-team)

</div>

---

## 🎯 The Problem

Security cameras today are **reactive** — they record incidents but **do not prevent escalation**.

- 📹 Over **99% of surveillance footage is never reviewed**
- 🧑‍💻 Monitoring remains **manual, slow, and inefficient**
- ⏱️ Critical events are identified **hours or days after they occur**
- 💰 Real-time GPU-based solutions are **expensive and hard to scale**

There is a clear need for an intelligent, automated system that can detect violence **as it happens** — not after the fact.

---

## 💡 Our Approach

We trained a **large-scale deep learning model using GPU acceleration** and deployed it in an **optimized CPU environment**. This eliminates dependency on expensive GPU hardware while maintaining real-time violence scoring and high detection accuracy.

```
 GPU-Trained Model  ──►  CPU-Optimized Deployment  ──►  Scalable & Cloud-Ready
    (Training)               (Inference)                  (Production)
```

The result is a **production-ready, full-stack AI system** that bridges the gap between experimental research and real-world deployment — making intelligent surveillance **accessible, scalable, and affordable**.

---

## ✨ Key Features

<table>
<tr>
<td width="50%" valign="top">

### 🎯 95%+ Model Accuracy
State-of-the-art violence detection powered by a fine-tuned **MobileNetV2-LSTM** architecture with GPU-accelerated training.

</td>
<td width="50%" valign="top">

### ⚡ Real-Time Inference Pipeline
Optimized for **CPU deployment** — no GPU required at runtime. Fast, lightweight, and production-ready.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📊 Confidence Scoring & Insights
Every prediction includes **confidence percentages**, probability distributions, and **frame-level analysis** for full transparency.

</td>
<td width="50%" valign="top">

### 🔌 Live RTSP Stream Monitoring
Connect IP cameras and **RTSP streams** for continuous, real-time violence detection with automated alerts and clip saving.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📹 Video Upload & Analysis
Upload videos in **MP4, AVI, MOV, MKV** formats for on-demand violence detection with detailed results and playback.

</td>
<td width="50%" valign="top">

### ☁️ Scalable & Cloud-Ready
Designed for deployment across **institutions, enterprises, and content platforms** — dockerized and horizontally scalable.

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                       SafeSight — System Architecture                          │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   ┌──────────────┐       ┌──────────────┐       ┌──────────────────────┐      │
│   │   Client     │ HTTPS │   Next.js    │       │   ML Service         │      │
│   │   Browser    │◄─────►│   Frontend   │       │   (FastAPI +         │      │
│   └──────────────┘       └──────┬───────┘       │    TensorFlow)       │      │
│                                 │                └──────────┬───────────┘      │
│                                 │ REST API                  │                  │
│                                 ▼                           │ Inference        │
│                          ┌──────────────┐                   │                  │
│                          │   Backend    │◄──────────────────┘                  │
│                          │  (Node.js /  │                                      │
│                          │   Express)   │       ┌──────────────────────┐       │
│                          └──────┬───────┘       │   RTSP Service       │      │
│                                 │               │   (Live Streams)     │      │
│                                 │ Mongoose      └──────────────────────┘      │
│                                 ▼                                              │
│                          ┌──────────────┐                                      │
│                          │   MongoDB    │                                      │
│                          │   Atlas      │                                      │
│                          │   (GridFS)   │                                      │
│                          └──────────────┘                                      │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Core Services

| Service | Description |
|---------|-------------|
| **Frontend** | Next.js 14 dashboard with real-time monitoring, video management, and prediction history |
| **Backend** | Express.js REST API handling video storage, inference orchestration, and data management |
| **ML Service** | FastAPI service running the TensorFlow MobileNetV2-LSTM model for violence inference |
| **RTSP Service** | Live stream processing service for IP camera / RTSP stream monitoring |
| **Database** | MongoDB Atlas with GridFS for video storage and structured prediction data |

---

## 🛠️ Tech Stack

<table>
<tr>
<th>Layer</th>
<th>Technologies</th>
</tr>
<tr>
<td><strong>Frontend</strong></td>
<td>
<img src="https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=next.js" />
<img src="https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white" />
<img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat-square&logo=tailwind-css&logoColor=white" />
<img src="https://img.shields.io/badge/Framer_Motion-0055FF?style=flat-square&logo=framer&logoColor=white" />
</td>
</tr>
<tr>
<td><strong>Backend</strong></td>
<td>
<img src="https://img.shields.io/badge/Node.js-339933?style=flat-square&logo=node.js&logoColor=white" />
<img src="https://img.shields.io/badge/Express-000000?style=flat-square&logo=express&logoColor=white" />
<img src="https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white" />
</td>
</tr>
<tr>
<td><strong>ML / AI</strong></td>
<td>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" />
</td>
</tr>
<tr>
<td><strong>Database</strong></td>
<td>
<img src="https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white" />
<img src="https://img.shields.io/badge/GridFS-47A248?style=flat-square&logo=mongodb&logoColor=white" />
</td>
</tr>
<tr>
<td><strong>Deployment</strong></td>
<td>
<img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" />
<img src="https://img.shields.io/badge/Vercel-000000?style=flat-square&logo=vercel&logoColor=white" />
<img src="https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=white" />
<img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
</td>
</tr>
</table>

---

## 🧠 Model Details

### MobileNetV2-LSTM Architecture

```
Input (16 frames × 224 × 224 × 3)
         │
         ▼
┌─────────────────────┐
│  MobileNetV2 (CNN)  │  ← Pretrained feature extraction
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  TimeDistributed    │  ← Apply CNN to each frame
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│      LSTM (64)      │  ← Temporal sequence learning
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Dense + Dropout   │  ← Classification head
└─────────┬───────────┘
          │
          ▼
    Output (2 classes)
    Violence · Non-Violence
```

| Property | Value |
|----------|-------|
| **Architecture** | MobileNetV2 + LSTM |
| **Training** | GPU-accelerated (Google Colab) |
| **Deployment** | CPU-optimized inference |
| **Accuracy** | 95%+ |
| **Input Shape** | 16 frames × 224 × 224 × 3 |
| **Output** | Binary (Violence / Non-Violence) |
| **Framework** | TensorFlow / Keras |
| **Dataset** | [Real Life Violence Situations](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) — 2,000 videos |

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Node.js | 18+ |
| Python | 3.9+ |
| MongoDB | 6+ (or Atlas) |
| Git | Latest |

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Sudhirkumar6009/ViolenceSense.git
cd ViolenceSense

# Run the setup script
# Windows:
.\setup.bat

# Linux/Mac:
chmod +x setup.sh && ./setup.sh
```

### Docker

```bash
docker-compose up -d        # Start all services
docker-compose logs -f      # View logs
docker-compose down         # Stop services
```

> 📖 For detailed setup instructions, environment variables, and API documentation, see the **[Quick Start Guide](./docs/QUICKSTART.md)**, **[Architecture Docs](./docs/ARCHITECTURE.md)**, and **[API Reference](./docs/API.md)**.

---

## 🌐 Live Demo

| Service | URL | Status |
|---------|-----|--------|
| 🖥️ **Frontend** | [violencesense.vercel.app](https://violencesense.vercel.app) | ![Vercel](https://img.shields.io/badge/Vercel-Live-00C853?style=flat-square) |
| ⚙️ **Backend API** | [violencesense-api.onrender.com](https://violencesense-api.onrender.com) | ![Render](https://img.shields.io/badge/Render-Live-46E3B7?style=flat-square) |
| 🧠 **ML Service** | [huggingface.co/spaces/SudhirKuchara/violencesense-ml](https://huggingface.co/spaces/SudhirKuchara/violencesense-ml) | ![HuggingFace](https://img.shields.io/badge/HuggingFace-Live-FFD21E?style=flat-square) |

---

## 📁 Project Structure

```
SafeSight/
├── frontend/          → Next.js 14 dashboard & UI
├── backend/           → Express.js REST API server
├── ml-service/        → FastAPI + TensorFlow inference service
├── rtsp-service/      → Live RTSP stream processing service
├── docs/              → Architecture, API, and quickstart docs
├── docker-compose.yml → Docker orchestration
├── setup.bat / .sh    → One-click setup scripts
└── README.md          → You are here
```

---

## 👥 Team

This project was built by:

- **[Sudhir Kumar](https://github.com/Sudhirkumar6009)** — Full-stack development, system architecture, and deployment
- **Jay Prajapati** — Model training, accuracy refinement, and presentation

### 🎓 Mentorship

Grateful to our mentors for strategic guidance, technical insights, and pushing us to refine both the model and its real-world applicability.

---

## 📈 Experience & Learning

Building SafeSight was not just about training a model — it was about **engineering a deployable system**. From optimizing the MobileNetV2-LSTM architecture to reducing inference latency on CPU environments, we focused on making the solution **scalable and practical**, not just experimental.

> _🚀 Winning this challenge validates our belief that AI systems should move beyond recording events — and toward **preventing them**._

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

1. **Fork** the repository  
2. **Create** a feature branch (`git checkout -b feature/YourFeature`)  
3. **Commit** your changes (`git commit -m 'Add YourFeature'`)  
4. **Push** to the branch (`git push origin feature/YourFeature`)  
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[IBM SkillsBuild](https://skillsbuild.org/)** & **[CSRBOX®](https://csrbox.org/)** — For organizing the AI Innovation Challenge 2026
- **i-Hub Gujarat** — For hosting the event
- **[Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)** by Mohamed Mustafa
- **[TensorFlow](https://tensorflow.org)**, **[Next.js](https://nextjs.org)**, **[FastAPI](https://fastapi.tiangolo.com)** — Open-source tools that made this possible

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [Sudhir Kumar](https://github.com/Sudhirkumar6009) & Jay Prajapati

[![GitHub](https://img.shields.io/badge/GitHub-Sudhirkumar6009-181717?style=for-the-badge&logo=github)](https://github.com/Sudhirkumar6009)

</div>
