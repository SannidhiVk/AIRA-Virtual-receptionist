# Face Verification Implementation Tasks

## Server Side
- `[x]` **A** — `face_recognition_service.py` (DeepFace core logic)
- `[x]` **B** — `employee_routes.py` (photo upload/serve API)
- `[x]` **C** — `websocket_routes.py` — add `verify_face` message handler
- `[x]` **D** — `main.py` — register employee router

## Frontend
- `[x]` **E** — `CameraStream.tsx` — add `captureFrame()` via `forwardRef`
- `[x]` **F** — `useFaceVerification.ts` hook
- `[x]` **G** — `page.tsx` — wire hook + show verification badge
- `[x]` **H** — `admin/employees/page.tsx` — photo upload admin UI

## Config
- `[x]` **I** — `pyproject.toml` — add `deepface` dependency
