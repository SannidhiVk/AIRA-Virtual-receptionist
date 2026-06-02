# 🎭 AIRA Virtual Receptionist
**An intelligent, voice-enabled virtual receptionist for modern businesses**

Automate front-desk interactions, handle visitor check-ins, and answer queries in real-time using advanced AI, speech recognition, and face verification - all optimized to run efficiently!

## ✨ Features
- **🎤 Real-time Voice Interaction** - Natural, dynamic voice interactions powered by state-of-the-art TTS (Kokoro) and STT (Faster Whisper).
- **👋 Face Verification & Presence Detection** - Autonomously recognize employees and detect new visitors using vision pipelines.
- **📅 Automated Appointments & Slack Integration** - Instant arrival notifications on Slack and automated email workflows.
- **⚡ CPU-Optimized Inference Pipelines** - Runs efficiently without a GPU via quantized Models/ONNX, but natively accelerates on CUDA if available.
- **🌐 Full-Stack Monorepo Ecosystem** - Complete with a 3D avatar Kiosk frontend (Next.js/Three.js), a comprehensive Dashboard (Vite/React), and a robust backend API (FastAPI).

## 🛠️ Technology Stack
- **Backend:** Python, FastAPI, SQLAlchemy, SQLite
- **Client (Kiosk):** Next.js, React, Three.js (3D Avatar)
- **Dashboard:** Vite, React, TailwindCSS
- **AI Models:** Groq (LLM), openwakeword, silero-vad, faster-whisper, Kokoro TTS, DeepFace, MediaPipe.

---

## 🚀 Setup & Installation

### 📋 Prerequisites
- **Node.js**: v18 or higher (v20+ recommended)
- **pnpm**: Package manager for the monorepo (`npm i -g pnpm`)
- **Python**: 3.10+
- **UV**: Python package manager (`pip install uv`) - required for backend setup.

### 💻 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SannidhiVk/AIRA-Virtual-receptionist.git
   cd AIRA-Virtual-receptionist
   ```

2. **Configure your Environment Variables:**
   Create an `.env` file in `apps/server/` (you can duplicate `.env.example` if it exists) and fill in the required keys. 
   *(See the highly detailed guide below on how exactly to obtain these keys).*

3. **Install Dependencies:**
   Automatically setup the Node monorepo and Python virtual environment (via `uv`):
   ```bash
   pnpm run monorepo-setup
   ```

4. **Start the System:**
   Run the Kiosk, Dashboard, and Server simultaneously:
   ```bash
   pnpm run dev
   ```

---

## 🔑 Environment Variables Guide (`apps/server/.env`)

To make the AI Receptionist fully functional, you need to configure API keys for Emails, AI Inference, and Slack Notifications.
Create your `.env` file in the `apps/server` directory with the following variables and fetch them using the steps below.

### 1. Email Bot Credentials (For Sending Invites/Logs)
Used by the system to send emails autonomously using Gmail SMTP.

* **`EMAIL_SENDER`**: Your bot's Gmail address (e.g., `your.receptionist.bot@gmail.com`).
* **`EMAIL_PASSWORD`**: A generated **App Password** (NOT your standard Google Account password).
  * **How to get it:**
    1. Go to your [Google Account > Security tab](https://myaccount.google.com/security).
    2. Ensure **2-Step Verification** is turned ON.
    3. Search for **"App Passwords"** in the settings search bar.
    4. Create a new app password (Name it "AIRA Bot"). It will generate a 16-character string.
    5. Copy and paste it here *(without any spaces)*.

### 2. Groq API Keys (For LLM Intelligence)
Powers the Llama-3 brain of the receptionist for lightning-fast conversations.

* **`GROQ_API_KEY`** and **`GROQ_API_KEY_2`**: 
  * **How to get it:**
    1. Go to the [GroqCloud Console](https://console.groq.com/keys).
    2. Sign in or create an account.
    3. Click on **Create API Key**.
    4. You can make two distinct keys or duplicate the same one if load balancing is unnecessary for your use case.

### 3. Slack Integration (For Employee Notifications)
Tells the host via Slack that their visitor has arrived at the reception.

* **`SLACK_SIGNING_SECRET`** & **`SLACK_BOT_TOKEN`**:
  * **How to get them:**
    1. Go to the [Slack API Dashboard](https://api.slack.com/apps).
    2. Click **Create New App** → **From scratch**. Name it and select your workspace.
    3. In the default **Basic Information** page, scroll down to **App Credentials** to find your **Signing Secret** (`SLACK_SIGNING_SECRET`).
    4. On the left sidebar, click **OAuth & Permissions**.
    5. Scroll down to **Scopes** → **Bot Token Scopes** and add the following required scopes:
       - `channels:history`
       - `chat:write`
       - `groups:history`
       - `im:history`
       - `im:read`
       - `im:write`
       - `incoming-webhook`
       - `users:read`
       - `users:read.email`
    6. Scroll up and click **Install to Workspace**.
    7. Once installed, it will generate a **Bot User OAuth Token** starting with `xoxb-`. Use this string for your `SLACK_BOT_TOKEN`.
* **`SLACK_CHANNEL_ID`**: The specific channel where the bot drops visitor alerts.
  * **How to get it:**
    1. Inside the Slack application, create or open a channel for front-desk notifications (e.g., `#reception`).
    2. Right-click the channel name in the sidebar and select **View channel details**.
    3. Scroll to the very bottom to find the **Channel ID** (usually starts with a `C`).
    4. *⚠️ Important:* Make sure to invite the bot to this channel using `/invite @YourBotName` inside Slack!

### 4. Face Verification Tuning (Optional)
Configurations used by the DeepFace library for identity recognition. These are usually safe to leave defaults.

* **`FACE_VERIFY_DETECTOR=ssd`**
* **`FACE_VERIFY_MODEL=ArcFace`**
* **`FACE_VERIFY_THRESHOLD=0.68`**

---

## 🏗️ Project Structure
* `/apps/server` - The FastAPI backend routing and AI inference logic.
* `/apps/client` - Next.js Kiosk system providing the real-time 3D avatar view.
* `/apps/dashboard` - Admin interface to manage employees, view active logs, and monitor system states.
