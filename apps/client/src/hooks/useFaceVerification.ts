// ─── Face Verification Feature Implementation ──────────────────────────────────
// This file implements the `useFaceVerification` hook which ties together the CameraStream
// frame capture capability and the WebSocket communication with the backend for
// face recognition.

import { useState, useCallback, useRef, useEffect } from 'react';
import { CameraStreamHandle } from '@/components/CameraStream';
import {
  FaceVerificationRequestOptions,
  useWebSocketContext
} from '@/contexts/WebSocketContext';

export interface FaceVerificationResult {
  verified: boolean;
  distance: number;
  message?: string;
  audioName: string;
  hasPhoto: boolean;
  personType?: 'employee' | 'visitor';
  sessionAction?: 'capture_reference' | 'compare_reference';
  referenceCaptured?: boolean;
}

interface FaceVerificationOptions {
  ensureCameraReady?: () => Promise<boolean>;
}

export function useFaceVerification(
  cameraRef: React.RefObject<CameraStreamHandle | null>,
  options?: FaceVerificationOptions
) {
  const [isVerifying, setIsVerifying] = useState(false);
  const [result, setResult] = useState<FaceVerificationResult | null>(null);
  const [cameraStartupError, setCameraStartupError] = useState<string | null>(
    null
  );
  const clearTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const clearCameraErrorRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  );
  const {
    onVerificationResult,
    sendFaceVerificationRequest,
    onEmployeeIdentified
  } = useWebSocketContext();

  // Listen for verification results from the backend
  useEffect(() => {
    // We expect the backend to send: { type: "face_verification_result", verified, distance, audio_name, has_photo }
    if (onVerificationResult) {
      onVerificationResult((data) => {
        setIsVerifying(false);
        setResult({
          verified: data.verified ?? false,
          distance: data.distance ?? -1,
          audioName: data.audio_name ?? '',
          hasPhoto: data.has_photo ?? false,
          message: data.message,
          personType: data.person_type,
          sessionAction: data.session_action,
          referenceCaptured: data.reference_captured
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
      if (clearCameraErrorRef.current) {
        clearTimeout(clearCameraErrorRef.current);
      }
    };
  }, [onVerificationResult]);

  const verifyFace = useCallback(
    (audioName: string, requestOptions?: FaceVerificationRequestOptions) => {
      if (!cameraRef.current || !sendFaceVerificationRequest) {
        console.warn(
          'Face verification skipped: camera or websocket not ready.'
        );
        return;
      }

      const personType = requestOptions?.personType ?? 'employee';
      const sessionAction =
        requestOptions?.sessionAction ??
        (personType === 'visitor' ? 'compare_reference' : undefined);

      console.log(
        `Verifying face for ${personType}: ${audioName || 'visitor'}${
          sessionAction ? ` (${sessionAction})` : ''
        }`
      );

      // 1. Capture the current frame from the camera as base64 JPEG
      const base64Image = cameraRef.current.captureFrame();

      if (!base64Image) {
        console.warn(
          'Face verification failed: could not capture frame from camera.'
        );
        return;
      }

      // 2. Clear previous result and set loading state
      setResult(null);
      setIsVerifying(true);

      // 3. Send the request via WebSocket
      sendFaceVerificationRequest(audioName, base64Image, requestOptions);
    },
    [cameraRef, sendFaceVerificationRequest]
  );

  // Auto-trigger verification when backend resolves employee identity from speech.
  useEffect(() => {
    if (!onEmployeeIdentified) {
      return;
    }
    onEmployeeIdentified((employeeName) => {
      void (async () => {
        if (options?.ensureCameraReady) {
          const isReady = await options.ensureCameraReady();
          if (!isReady) {
            const message =
              'Could not start camera automatically. Please allow camera access and try again.';
            setCameraStartupError(message);
            console.warn(
              'Face verification skipped: camera could not be started automatically.',
              { employeeName }
            );
            if (clearCameraErrorRef.current) {
              clearTimeout(clearCameraErrorRef.current);
            }
            clearCameraErrorRef.current = setTimeout(
              () => setCameraStartupError(null),
              6000
            );
            return;
          }
        }
        setCameraStartupError(null);
        verifyFace(employeeName);
      })();
    });
  }, [onEmployeeIdentified, options, verifyFace]);

  return { verifyFace, isVerifying, result, cameraStartupError };
}
