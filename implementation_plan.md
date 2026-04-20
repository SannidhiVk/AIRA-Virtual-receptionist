# Visual Face Recognition — Employee Arrival Verification

---

## Part 1 — Face Library Comparison (Tailored to AIRA)

> Your project: Python **3.10**, **Windows**, CPU-compatible, FastAPI backend, pip managed via `uv`.


| Criterion              | `face_recognition` (dlib)                                                                                                                       | `**DeepFace`** ✅ Recommended                                                        | `OpenCV` (LBPH/Haar)                                                                                        |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Install on Windows** | ❌ Requires `cmake` + Visual C++ Build Tools. Dlib builds frequently fail on Python 3.10/Windows without a pre-built wheel. Most painful option. | ✅ Pure `pip install deepface` — no compilers, no build tools.                       | ✅ Already installed as a transitive dep. Zero effort.                                                       |
| **Accuracy**           | ⭐⭐⭐⭐⭐ 99.38% on LFW dataset. Best raw accuracy.                                                                                                 | ⭐⭐⭐⭐ 97–98% with Facenet/ArcFace. Excellent for a controlled reception environment. | ⭐⭐ ~85–90%. LBPH is outdated. Sensitive to lighting and face angle. Will produce false positives/negatives. |
| **CPU Speed**          | Fast (~0.1–0.3 s/comparison)                                                                                                                    | Moderate (~0.5–1.5 s/comparison with Facenet)                                       | Very fast (~0.02 s)                                                                                         |
| **First-run setup**    | None after build                                                                                                                                | Downloads model weights once (~90 MB for Facenet)                                   | None                                                                                                        |
| **Code complexity**    | Very simple (2-line API)                                                                                                                        | Simple (1-line verify call)                                                         | Complex (manual pipeline)                                                                                   |
| **Python 3.10 compat** | ⚠️ May need pre-built wheel                                                                                                                     | ✅ Full support                                                                      | ✅ Full support                                                                                              |
| **Maintenance**        | Stable but slow to update                                                                                                                       | Actively maintained, multiple backends                                              | Stable                                                                                                      |
| **Fit for reception**  | ✅ Great — if you can get it installed                                                                                                           | **✅ Best overall fit for your project**                                             | ❌ Not reliable enough for identity verification                                                             |


### ✅ Verdict: Use **DeepFace** with the `Facenet` model

**Why not `face_recognition`?**  
Building dlib on Windows + Python 3.10 without a pre-built wheel is a well-known pain point. It can take hours of debugging cmake/MSVC issues. Your project is already complex — we don't want a fragile build dependency.

**Why not `OpenCV`?**  
LBPH (Local Binary Pattern Histogram) is a shallow, non-deep-learning method. In a real-world reception scenario (different lighting, slight angle variation, glasses), it will produce too many false positives/negatives for identity verification to be trustworthy.

**Why `DeepFace` with `Facenet`?**

- `pip install deepface` — done. No cmake, no MSVC.
- `Facenet` model is ~90 MB, downloaded once, then cached.
- One-line verification: `DeepFace.verify(img1, img2, model_name='Facenet')`
- Accuracy is more than sufficient for a controlled office reception (good lighting, frontal face).
- CPU inference is acceptable (~1 second), which fits a conversational flow where a 1 second delay is imperceptible.

---

## Part 2 — Finalized Decisions


| Decision       | Choice                                                                                                   |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| Face library   | **DeepFace + Facenet model**                                                                             |
| Trigger moment | **Option A** — When LLM identifies the employee name from audio                                          |
| On mismatch    | **Jarvis verbal challenge** — *"I see someone different in the camera — can you confirm your identity?"* |
| Scope          | **Employees only**                                                                                       |


---

## Part 3 — System Architecture

```
Employee arrives at reception
          │
          ▼
[CameraStream.tsx] — live webcam running in background
          │
          ▼
[Audio Pipeline]
Mic → VAD → Wake Word "Hey Jarvis" → Whisper STT → query_router
                                                          │
                                              Employee identifies themselves
                                              e.g. "I'm John" → LLM returns
                                              confirmation + extracts name
                                                          │
                                         ┌────────────────┘
                                         │  Trigger: name identified
                                         ▼
                         [Frontend captures frame]
                         CameraStream.captureFrame() → base64 JPEG
                                         │
                         WebSocket sends: { type: "verify_face",
                                           audio_name: "John Doe",
                                           image_b64: "..." }
                                         │
                                         ▼
                         [Server: face_recognition_service.py]
                         DeepFace.verify(captured_frame, stored_photo)
                                         │
                              ┌──────────┴──────────┐
                            MATCH                MISMATCH
                              │                      │
                    Jarvis: "Welcome          Jarvis: "I see someone
                     back, John!"            different — can you
                   ✅ UI badge               confirm your identity?"
                                            ❌ UI badge + log event
```

---

## Part 4 — Files to Create / Modify

### Component A — Server: Face Recognition Service

#### [NEW] `apps/server/services/face_recognition_service.py`

Responsibilities:

- Accept a base64 JPEG image (captured frame from frontend)
- Look up the employee by name in the DB, get their `photo_path`
- Run `DeepFace.verify(img1_path, img2_bytes, model_name='Facenet', enforce_detection=False)`
- Return `{ verified: bool, distance: float, matched_name: str }`
- Cache face encodings in memory to avoid re-encoding on every call

**Key design notes:**

- `enforce_detection=False` — graceful fallback if face isn't perfectly centred (important for a wide webcam angle)
- Use `Facenet` (not VGG-Face) — smaller download, faster on CPU
- Store photos in `apps/server/receptionist/photos/employees/` (relative to the DB)

---

### Component B — Server: Photo Upload API

#### [NEW] `apps/server/routes/employee_routes.py`

Endpoints:

```
GET  /employees/              → List all employees (id, name, department, photo_path)
POST /employees/{id}/photo   → Upload photo (multipart), save to disk, update DB photo_path
GET  /employees/{id}/photo   → Serve stored photo as image response
```

The photo is saved to `receptionist/photos/employees/{employee_id}.jpg`.

---

### Component C — Server: WebSocket face verify message type

#### [MODIFY] `apps/server/routes/websocket_routes.py`

Add handling for a new JSON message type inside the `listener()` coroutine:

```python
if msg.get("type") == "verify_face":
    audio_name = msg["audio_name"]
    image_b64  = msg["image_b64"]
    result = await run_in_executor(verify_employee_face, audio_name, image_b64)
    await websocket.send_text(json.dumps({
        "type": "face_verification_result",
        "verified": result["verified"],
        "distance": result["distance"],
        "audio_name": audio_name,
        "message": result["message"]
    }))
```

If **verified = False**, the `brain()` coroutine queues the challenge phrase to TTS:

> *"I see someone different in the camera — can you confirm your identity?"*

---

### Component D — Server: `main.py` / router registration

#### [MODIFY] `apps/server/main.py`

Include the new `employee_routes` router:

```python
from routes.employee_routes import router as employee_router
app.include_router(employee_router, prefix="/api")
```

---

### Component E — Frontend: Frame Capture in CameraStream

#### [MODIFY] `apps/client/src/components/CameraStream.tsx`

Add:

1. A hidden `<canvas>` element (ref: `canvasRef`) sized to video dimensions
2. A `captureFrame(): string | null` function exposed via `useImperativeHandle`
  - Draws the current video frame onto the canvas
  - Returns `canvas.toDataURL('image/jpeg', 0.8)` (base64 JPEG, 80% quality)
3. A new prop `ref` (`forwardRef`) so the parent can call `cameraRef.current.captureFrame()`

---

### Component F — Frontend: Face Verification Hook

#### [NEW] `apps/client/src/hooks/useFaceVerification.ts`

```typescript
useFaceVerification(cameraRef, websocket)
  → verifyFace(audioName: string): void
  → result: { verified, distance, message } | null
  → isVerifying: boolean
```

When `verifyFace(name)` is called:

1. Calls `cameraRef.current.captureFrame()` to grab the base64 image
2. Sends `{ type: "verify_face", audio_name: name, image_b64 }` over the WebSocket
3. Listens for `{ type: "face_verification_result" }` message
4. Updates state with the result

---

### Component G — Frontend: Verification Result UI Badge

#### [MODIFY] Main page / conversation component (wherever `CameraStream` is used)

Add a subtle overlay badge near the camera widget:

- ✅ Green badge: *"Identity Confirmed — John Doe"*
- ❌ Red badge: *"Identity Mismatch — please confirm"* + pulsing warning
- ⏳ Spinner: *"Verifying..."* during the DeepFace call

The badge auto-dismisses after 8 seconds.

---

### Component H — Admin: Employee Photo Upload UI

#### [NEW] `apps/client/src/app/admin/employees/page.tsx`

A clean admin page (accessible at `/admin/employees`) that:

- Lists all employees from `GET /api/employees/`
- Shows their stored photo thumbnail (or a placeholder if no photo yet)
- Has a file upload button per employee → `POST /api/employees/{id}/photo`
- Shows a ✅ / ⚠️ badge indicating whether a photo is set

---

## Part 5 — Install Command

```bash
# Inside apps/server with the .venv active:
pip install deepface
# DeepFace will auto-download Facenet weights (~90MB) on first use
```

Add to `pyproject.toml`:

```toml
"deepface>=0.0.93",
"tf-keras",   # required by some DeepFace backends on newer TF
```

> [!NOTE]
> `deepface` pulls in `tensorflow` as a dependency. On CPU-only systems, use `tensorflow-cpu` to keep the install light. We will pin this in `pyproject.toml`.

---

## Part 6 — Verification Plan

### Step 1 — Seed a test photo

- Upload a photo for a seeded employee via `POST /api/employees/{id}/photo`
- Confirm `photo_path` is updated in the DB

### Step 2 — Happy path

- Start AIRA, trigger wake word
- Say the employee's name
- When LLM confirms the name, frontend should auto-capture frame + verify
- Confirm ✅ badge appears and Jarvis does NOT challenge

### Step 3 — Mismatch path

- Say a **different employee's name** while your face is in the camera
- Confirm ❌ badge appears
- Confirm Jarvis says *"I see someone different — can you confirm your identity?"*

### Step 4 — No photo fallback

- Employee has no `photo_path` → skip verification, log a warning, continue normally (no false blocks)

### Step 5 — No face detected fallback

- Cover camera / bad lighting → `enforce_detection=False` means DeepFace returns a low-confidence result, not a crash
- System gracefully skips the challenge if confidence is below threshold

## 1. What is `CameraStream.tsx` and what does it do today?

`CameraStream.tsx` is a **floating, draggable camera widget** in the AIRA frontend.  
It is part of the **AI Virtual Receptionist** (`SannidhiVk/AIRA-Virtual-receptionist`) — a CPU-compatible voice assistant built with:


| Layer           | Technology                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------ |
| Frontend        | Next.js / React (TSX)                                                                            |
| Real-time comms | WebSocket (`/ws/{client_id}`)                                                                    |
| Audio pipeline  | Microphone → VAD (Silero) → Wake-word (OpenWakeWord) → STT (Whisper) → LLM (Groq) → TTS (Kokoro) |
| Database        | SQLite via SQLAlchemy (`receptionist/` module)                                                   |


**What `CameraStream.tsx` currently does:**

- Opens the webcam via `getUserMedia`.
- Renders a live `<video>` element in a floating, resizable, draggable panel.
- Exposes a `onStreamChange(stream)` callback to give a parent component access to the `MediaStream`.
- Has a companion `CameraToggleButton` for toggling the widget.

> **It is currently purely a display widget** — the camera feed is shown but never processed or sent anywhere. That is exactly what we will change.

---

## 2. Proposed Feature: Visual Employee Verification

### Goal

When an employee says their name (audio path), the system will **simultaneously capture a photo from the camera**, send it to the server, compare it to the stored employee photo in the DB, and report whether the face and name match.

### Flow Diagram

```
Employee arrives
     │
     ▼
[CameraStream] streams live video
     │
     ▼
[Audio pipeline] hears name (Whisper → query_router)
     │
     ├──► Server extracts name, looks up DB → audio_name_match
     │
     └──► Server receives captured photo frame
               │
               ▼
         [face_recognition service]
         Loads stored photo_path from Employee DB row
         Compares encodings (face_recognition / DeepFace CPU)
               │
               ▼
         Returns: { visual_match: bool, confidence: float, matched_employee: str }
               │
               ▼
         Cross-validate: audio_name == matched_employee?
               │
         ┌────┴────┐
         YES       NO
         │         │
    "Identity   "Identity mismatch!
     confirmed"  You said [X] but camera
                 shows [Y]."
```

---

## 3. Proposed Changes

### Component A — Database (no schema change needed)

The `employees` table **already has** `photo_path = Column(String)`.  
We just need a way to **upload and store employee photos**.

---

#### [NEW] `apps/server/routes/employee_routes.py`

- `POST /employees/{id}/photo` — uploads a photo, saves it to `photos/employees/{id}.jpg`, updates `photo_path` in DB.
- `GET /employees/{id}/photo` — serves the stored photo.
- `GET /employees/` — list all employees (for admin UI).

---

### Component B — Server: Face Recognition Service

> [!IMPORTANT]
> We will use `**face_recognition`** (dlib-based, CPU-friendly) or `**DeepFace**` (also CPU-capable with `VGG-Face` or `Facenet`). Both work without GPU. We will default to `face_recognition` as it is simpler.

#### [NEW] `apps/server/services/face_recognition_service.py`

```
Functions:
  - encode_face(image_bytes) → numpy encoding
  - compare_faces(known_encoding, candidate_image_bytes) → (match: bool, distance: float)
  - find_matching_employee(candidate_image_bytes, db_session) → Employee | None
```

This service:

1. Loads the captured frame (JPEG bytes from frontend).
2. Computes a face encoding.
3. Iterates over all employees with a stored `photo_path`, computes their encodings (cached in memory after first load).
4. Returns the closest match below a tolerance threshold.

---

### Component C — Server: New WebSocket / REST endpoint for frame submission

#### [MODIFY] `apps/server/routes/api_routes.py`

Add:

```
POST /verify-face
Body: { client_id, image_b64, audio_name }
Response: { verified: bool, visual_match: str, audio_name: str, message: str, confidence: float }
```

This endpoint:

1. Decodes the base64 image.
2. Calls `find_matching_employee()`.
3. Compares result name to `audio_name` (fuzzy string match).
4. Returns a structured result.

Alternatively (preferred for real-time): add a JSON message type `{ type: "verify_face", image_b64, audio_name }` over the **existing WebSocket**, so the client needs no second connection.

---

### Component D — Frontend: Capture frame and send on trigger

#### [MODIFY] `apps/client/src/components/CameraStream.tsx`

Add a `captureFrame(): string` method (exposed via `useImperativeHandle`) that:

1. Draws the current `videoRef` frame onto a hidden `<canvas>`.
2. Returns the image as a base64 JPEG string.

Also add a new prop `onFrameCapture?: (b64: string) => void` that parents can listen to.

#### [NEW] `apps/client/src/hooks/useFaceVerification.ts`

A hook that:

1. Accepts a `cameraRef` and WebSocket ref.
2. Exposes `verifyFace(audioName: string)`.
3. Captures a frame from `CameraStream`, sends `{ type: "verify_face", ... }` over WS.
4. Receives the result and returns `{ verified, message, confidence }`.

#### [MODIFY] Main page / conversation component

- After the audio pipeline resolves an employee name (i.e., the LLM replies with an identity), trigger `verifyFace(name)`.
- Display result visually: ✅ "Identity Confirmed — John Doe" or ❌ "Mismatch — camera sees Jane, you said John."

---

### Component E — Admin: Employee photo upload UI

#### [NEW] `apps/client/src/app/admin/employees/page.tsx`

A simple admin page to:

- List all employees from `GET /employees/`.
- Allow uploading a photo per employee via `POST /employees/{id}/photo`.
- Show the stored photo thumbnail.

---

## 4. Open Questions

> [!IMPORTANT]
> **Q1 — Face recognition library choice:**  
> `face_recognition` requires `cmake` + `dlib` to build (can be slow to install on Windows).  
> `DeepFace` is pip-installable and has more model options but is heavier.  
> **Which would you prefer?** Or should we use OpenCV template matching as a lighter fallback?

> [!IMPORTANT]
> **Q2 — Trigger point:**  
> When exactly should the face capture be triggered?  
> Option A: When the employee **says their name** (LLM identifies them by audio).  
> Option B: When the **wake word fires** (capture immediately on approach).  
> Option C: **Continuously in the background** and only cross-check when name is spoken.  
> (Recommended: Option A — simplest to implement correctly.)

> [!IMPORTANT]
> **Q3 — What happens on mismatch?**  
> Should Jarvis verbally challenge ("I see someone different — can you confirm your identity?"), or just log it silently, or refuse access?

> [!NOTE]
> **Q4 — Visitor photos:**  
> The `visitors` table also has `id_photo_path`. Should visitor face-verification be included in this scope, or only employees for now?

---

## 5. Verification Plan

### Automated

- Unit test `face_recognition_service.py` with a known photo pair (match) and a different pair (no match).
- Test the `/verify-face` REST endpoint with Postman/httpie.

### Manual

- Upload an employee photo via the admin UI.
- Trigger the wake word, speak the employee's name.
- Observe the ✅/❌ verification badge in the UI.
- Test with a **wrong face** speaking the correct name to confirm mismatch detection.  
  
  


6 . **Full Trigger Flow (End-to-End)**  
[Employee walks in]

       │

       ▼

"Hey Jarvis, I'm John Doe"

       │

       ├─ Audio → Whisper → "I am John Doe"

       │

       ├─ query_[router.py](http://router.py) → finds Employee "John Doe" in DB

       │      │

       │      └─ LLM reply: "Welcome back John! How can I help?"

       │             │

       │             └─ Backend sends to frontend:

       │                { type: "employee_identified", name: "John Doe" }

       │

       ▼

Frontend receives "employee_identified"

       │

       ├─ useFaceVerification hook fires

       ├─ captureFrame() grabs 1 frame from camera video

       ├─ Sends over WS: { type: "verify_face", audio_name: "John Doe", image_b64: "..." }

       │

       ▼

Backend face_recognition_[service.py](http://service.py)

       │

       ├─ Loads John Doe's stored photo from disk

       ├─ DeepFace.verify(stored_photo, captured_frame)

       │

       ├─ MATCH → { verified: true }

       │     └─ Frontend shows ✅ "Identity Confirmed"

       │        Jarvis already greeted, no extra challenge

       │

       └─ MISMATCH → { verified: false }

             └─ Frontend shows ❌ badge

                Backend queues TTS:

                "I see someone different in the camera —

                 can you confirm your identity?"

