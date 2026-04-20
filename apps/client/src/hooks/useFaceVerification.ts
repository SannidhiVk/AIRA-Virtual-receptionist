// ─── Face Verification Feature Implementation ──────────────────────────────────
// This file implements the `useFaceVerification` hook which ties together the CameraStream
// frame capture capability and the WebSocket communication with the backend for
// face recognition.

import { useState, useCallback, useRef, useEffect } from 'react';
import { CameraStreamHandle } from '@/components/CameraStream';
import { useWebSocketContext } from '@/contexts/WebSocketContext';

export interface FaceVerificationResult {
  verified: boolean;
  distance: number;
  message?: string;
  audioName: string;
  hasPhoto: boolean;
}

export function useFaceVerification(
  cameraRef: React.RefObject<CameraStreamHandle | null>
) {
  const [isVerifying, setIsVerifying] = useState(false);
  const [result, setResult] = useState<FaceVerificationResult | null>(null);
  const clearTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { onVerificationResult, sendFaceVerificationRequest, onEmployeeIdentified } =
    useWebSocketContext();

  // Listen for verification results from the backend
  useEffect(() => {
    // We expect the backend to send: { type: "face_verification_result", verified, distance, audio_name, has_photo }
    if (onVerificationResult) {
      onVerificationResult((data) => {
        setIsVerifying(false);
        setResult({
          verified: data.verified,
          distance: data.distance,
          audioName: data.audio_name,
          hasPhoto: data.has_photo,
          message: data.message
        });

        // Auto-clear the result after 8 seconds
        if (clearTimerRef.current) {
          clearTimeout(clearTimerRef.current);
        }
        clearTimerRef.current = setTimeout(() => setResult(null), 8000);
      });
    }
    return () => {
      if (clearTimerRef.current) {
        clearTimeout(clearTimerRef.current);
      }
    };
  }, [onVerificationResult]);


  const verifyFace = useCallback(
    (audioName: string) => {
      if (!cameraRef.current || !sendFaceVerificationRequest) {
        console.warn("Face verification skipped: camera or websocket not ready.");
        return;
      }

      console.log(`Verifying face for employee: ${audioName}`);
      
      // 1. Capture the current frame from the camera as base64 JPEG
      const base64Image = cameraRef.current.captureFrame();
      
      if (!base64Image) {
        console.warn("Face verification failed: could not capture frame from camera.");
        return;
      }

      // 2. Clear previous result and set loading state
      setResult(null);
      setIsVerifying(true);

      // 3. Send the request via WebSocket
      sendFaceVerificationRequest(audioName, base64Image);
    },
    [cameraRef, sendFaceVerificationRequest]
  );

  // Auto-trigger verification when backend resolves employee identity from speech.
  useEffect(() => {
    if (!onEmployeeIdentified) {
      return;
    }
    onEmployeeIdentified((employeeName) => {
      verifyFace(employeeName);
    });
  }, [onEmployeeIdentified, verifyFace]);

  return { verifyFace, isVerifying, result };
}
