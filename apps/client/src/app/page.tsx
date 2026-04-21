'use client';

import { useRef } from 'react';
import VoiceActivityDetector from '@/components/VoiceActivityDetector';
import TalkingHead from '@/components/TalkingHead';
import {
  CameraToggleButton,
  CameraStreamHandle
} from '@/components/CameraStream';
import { useFaceVerification } from '@/hooks/useFaceVerification';

export default function Home() {
  const cameraRef = useRef<CameraStreamHandle | null>(null);
  const { result, isVerifying } = useFaceVerification(cameraRef);

  return (
    <main className="relative min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="mb-2 text-4xl font-bold text-gray-900">
            AlmostHuman AI
          </h1>
          <p className="text-lg text-gray-600">
            An AI-Based Virtual Receptionist System
          </p>
        </div>

        {/* Main Content Layout */}
        <div className="mb-8 grid grid-cols-1 gap-8 xl:grid-cols-2">
          {/* TalkingHead Component */}
          <div className="order-1">
            <div className="rounded-lg bg-white p-6 shadow-lg">
              <TalkingHead />
            </div>
          </div>

          {/* Voice Activity Detector */}
          <div className="order-2">
            <VoiceActivityDetector />
          </div>
        </div>
      </div>

      {/* Floating Camera Component */}
      <CameraToggleButton cameraRef={cameraRef} />

      {/* Face verification badge */}
      {(isVerifying || result) && (
        <div className="fixed right-6 bottom-24 z-40 max-w-sm rounded-lg border bg-white p-3 shadow-xl">
          {isVerifying && (
            <div className="flex items-center gap-2 text-sm text-gray-700">
              <span className="h-2 w-2 animate-pulse rounded-full bg-amber-500" />
              Verifying identity...
            </div>
          )}

          {!isVerifying && result && (
            <div
              className={`text-sm font-medium ${
                result.verified ? 'text-green-700' : 'text-red-700'
              }`}
            >
              {result.verified
                ? `Identity Confirmed - ${result.audioName}`
                : 'Identity Mismatch - please confirm'}
            </div>
          )}
        </div>
      )}
    </main>
  );
}
