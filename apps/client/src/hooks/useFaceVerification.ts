// ─── Face Verification Feature Implementation ──────────────────────────────────

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

// Extend the options to include our background flag so we don't flicker the UI
interface ExtendedVerificationOptions extends FaceVerificationRequestOptions {
  isBackground?: boolean;
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

  // Holds the running setInterval so we can stop it at any time
  const backgroundFaceCheckRef = useRef<ReturnType<typeof setInterval> | null>(
    null
  );

  const {
    onVerificationResult,
    sendFaceVerificationRequest,
    onEmployeeIdentified,
    onRequestFaceFrame,
    onStateChange
  } = useWebSocketContext();

  // ── Core verify function ────────────────────────────────────────────────────
  const verifyFace = useCallback(
    (audioName: string, requestOptions?: ExtendedVerificationOptions) => {
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

      // Capture the current frame from the camera as base64 JPEG
      const base64Image = cameraRef.current.captureFrame();

      if (!base64Image) {
        console.warn(
          '⚠️ captureFrame() returned null — camera may not be streaming yet.'
        );
        return;
      }

      // Only show loading UI for active (manual) checks, not silent background ones
      if (!requestOptions?.isBackground) {
        console.log(
          `Verifying face for ${personType}: ${audioName || 'visitor'} (${sessionAction})`
        );
        setResult(null);
        setIsVerifying(true);
      }

      sendFaceVerificationRequest(audioName, base64Image, requestOptions);
    },
    [cameraRef, sendFaceVerificationRequest]
  );

  // ── Stop background loop helper ─────────────────────────────────────────────
  const stopBackgroundLoop = useCallback((reason: string) => {
    if (backgroundFaceCheckRef.current) {
      console.log(`🛑 Stopping background camera loop. Reason: ${reason}`);
      clearInterval(backgroundFaceCheckRef.current);
      backgroundFaceCheckRef.current = null;
    }
  }, []);

  // ── Listen for verification results ────────────────────────────────────────
  useEffect(() => {
    if (!onVerificationResult) return;

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

      // ── Start background loop once visitor reference is captured ────────────
      if (data.reference_captured === true) {
        console.log(
          '✅ Reference captured! Starting 3-second background loop...'
        );

        // Clear any existing loop before starting a fresh one
        if (backgroundFaceCheckRef.current)
          clearInterval(backgroundFaceCheckRef.current);

        backgroundFaceCheckRef.current = setInterval(() => {
          console.log('📸 Snapping background photo and sending to backend...');
          verifyFace(data.audio_name ?? 'visitor', {
            personType: 'visitor',
            sessionAction: 'compare_reference',
            isBackground: true // Silent — won't trigger "Verifying..." UI
          });
        }, 3000);
      }

      // ── Stop loop if verification fails (visitor mismatch / kicked out) ─────
      if (data.verified === false && data.person_type === 'visitor') {
        stopBackgroundLoop('Visitor verification failed');
      }

      // Auto-clear result card after 8 seconds
      if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
      clearTimerRef.current = setTimeout(() => setResult(null), 8000);
    });

    return () => {
      if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
      if (clearCameraErrorRef.current)
        clearTimeout(clearCameraErrorRef.current);
      stopBackgroundLoop('component unmounted');
    };
  }, [onVerificationResult, verifyFace, stopBackgroundLoop]);

  // ── Backend-requested frame (active speaker verification) ───────────────────
  useEffect(() => {
    if (!onRequestFaceFrame) return;

    onRequestFaceFrame((data) => {
      console.log(
        '📸 Backend requested active speaker verification. Snapping frame...'
      );
      verifyFace(data.audio_name ?? '', {
        personType: data.person_type,
        sessionAction: data.session_action,
        isBackground: true // Silent
      });
    });
  }, [onRequestFaceFrame, verifyFace]);

  // ── Stop loop when session goes passive ─────────────────────────────────────
  useEffect(() => {
    if (!onStateChange) return;

    onStateChange((state) => {
      if (state === 'passive') {
        stopBackgroundLoop('session ended (passive)');
      }
    });
  }, [onStateChange, stopBackgroundLoop]);

  // ── Auto-trigger verification when backend identifies an employee ───────────
  useEffect(() => {
    if (!onEmployeeIdentified) return;

    onEmployeeIdentified((employeeName) => {
      void (async () => {
        if (options?.ensureCameraReady) {
          const isReady = await options.ensureCameraReady();
          if (!isReady) {
            setCameraStartupError(
              'Could not start camera automatically. Please allow camera access and try again.'
            );
            if (clearCameraErrorRef.current)
              clearTimeout(clearCameraErrorRef.current);
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

  return {
    verifyFace,
    isVerifying,
    result,
    cameraStartupError,
    stopBackgroundLoop
  };
}
