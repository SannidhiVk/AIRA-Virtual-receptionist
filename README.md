# 🎭 AIRA Virtual Receptionist
**An intelligent, voice-enabled virtual receptionist for modern businesses**

> Automate front-desk interactions, handle visitor check-ins, and answer queries in real-time using advanced AI and speech recognition.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=flat&logo=react&logoColor=%2361DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![License](https://img.shields.io/badge/license-MIT-green.svg)

![System Architecture](./images/architecture.svg)

## ✨ Features

### 🎯 **Core Capabilities**
- **🎤 Real-time Voice Interaction** - Natural, dynamic voice interactions powered by state-of-the-art TTS and STT.
- **📅 Automated Appointment Booking** - Effortlessly schedule meetings and manage calendars with AI precision.
- **🌐 Multi-language Support** - Communicate seamlessly with global visitors in their preferred language.
- **👋 Visitor Check-in Handling** - Automate the entire check-in process from greeting to notifying hosts.
- **⚡ Instant Query Resolution** - Provide immediate answers to common front-desk inquiries.

### 🚀 **Advanced Features**
- **🧠 Context-aware Memory** - Remembers visitor context across conversations for a personalized experience.
- **🔊 Custom Wake Word** - Activate the receptionist instantly with a custom wake word (e.g., "Hey jarvi").
- **📊 Dashboard Analytics** - Comprehensive insights into visitor traffic and receptionist performance.
- **📆 Google Calendar Integration** - Native sync with Google Calendar for real-time scheduling.
- **🔔 Seamless Employee Notifications** - Instantly alert team members upon their visitor's arrival.

## 🛠️ **Technology Stack**
- **Backend:** Python, FastAPI, SQLAlchemy
- **Frontend:** Next.js, React, TailwindCSS
- **AI Models:** Groq (LLM), faster-whisper (STT), Kokoro (TTS), openwakeword (Wake Word), silero-vad (VAD), deepface (Face Recognition)
- **Database/Other:** PostgreSQL, Redis

## 📋 **Requirements & Prerequisites**
- **OS:** Windows 11 / Ubuntu 22.04
- **Hardware:** Minimum 8GB RAM, standard microphone/webcam
- **Software:** Python 3.10+, Node.js 18+ (20+ recommended), `pnpm` package manager

## 🚀 **Quick Start / Setup Steps**

**Step 1:** Clone the repository
```bash
git clone https://github.com/SannidhiVk/AIRA-Virtual-receptionist.git
cd AIRA-Virtual-receptionist
```

**Step 2:** Setup configuration
```bash
# Add API keys to your environment files
# See .env.example files in apps/server and apps/client
```

**Step 3:** Setup Monorepo (Frontend & Backend)
```bash
pnpm run monorepo-setup
```

**Step 4:** Run the application
Follow the startup instructions for the individual frontend and backend packages.
```bash
pnpm run dev
```

## 🙏 **Acknowledgments**
- Powered by [Groq](https://groq.com/) and [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M).
- Built with [FastAPI](https://fastapi.tiangolo.com/), and [Next.js](https://nextjs.org/).
- Inspired by modern open-source AI applications and designed for the future of interactive front-desk automation.
