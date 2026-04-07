'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Mic } from 'lucide-react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';

import { useWebSocketContext } from '@/contexts/WebSocketContext';

// ─── Types ────────────────────────────────────────────────────────────────────

type BackendState = 'passive' | 'listening' | 'processing' | 'speaking';

// ─── Status Orb ───────────────────────────────────────────────────────────────

const StatusOrb: React.FC<{ state: BackendState }> = ({ state }) => {
  const config = {
    passive: {
      outer: 'bg-gray-200',
      inner: 'bg-gray-400',
      ring: '',
      label: "Say \"Hey Jarvis\"",
      labelClass: 'text-gray-500'
    },
    listening: {
      outer: 'bg-blue-100 animate-pulse',
      inner: 'bg-blue-500',
      ring: 'ring-4 ring-blue-300',
      label: 'Listening...',
      labelClass: 'text-blue-600 font-semibold'
    },
    processing: {
      outer: 'bg-yellow-100',
      inner: 'bg-yellow-400 animate-spin',
      ring: 'ring-4 ring-yellow-300',
      label: 'Thinking...',
      labelClass: 'text-yellow-600 font-semibold'
    },
    speaking: {
      outer: 'bg-green-100 animate-pulse',
      inner: 'bg-green-500',
      ring: 'ring-4 ring-green-300',
      label: 'Speaking...',
      labelClass: 'text-green-600 font-semibold'
    }
  } satisfies Record<BackendState, object>;
  const c = config[state] || config.passive;

  return (
    <div className="flex flex-col items-center gap-4">
      <div className={`flex h-36 w-36 items-center justify-center rounded-full transition-all duration-300 ${c.outer} ${c.ring}`}>
        <div className={`flex h-20 w-20 items-center justify-center rounded-full transition-all duration-300 ${c.inner}`}>
          <Mic size={28} className="text-white drop-shadow" />
        </div>
      </div>
      <p className={`text-base transition-all duration-300 ${c.labelClass}`}>
        {c.label}
      </p>
    </div>
  );
};

// ─── Indicator Dot ────────────────────────────────────────────────────────────

const Dot: React.FC<{ active: boolean; color: string; label: string }> = ({
  active,
  color,
  label
}) => (
  <div className="text-center">
    <div
      className={`inline-block h-3 w-3 rounded-full transition-colors duration-300 ${
        active ? color : 'bg-gray-300'
      }`}
    />
    <p className="text-muted-foreground mt-1 text-sm">{label}</p>
  </div>
);

// ─── Main Component ───────────────────────────────────────────────────────────

const VoiceActivityDetector: React.FC = () => {
  const [backendState, setBackendState] = useState<BackendState>('passive');
  const [micActive, setMicActive] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);

  const { sendAudioSegment, onServerState, isConnected } = useWebSocketContext();

  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const pcmCarryRef = useRef<number[]>([]);

  // 1. Listen for State Changes from Backend
  useEffect(() => {
    onServerState((state) => setBackendState(state));
  }, [onServerState]);

  useEffect(() => {
    if (!isConnected) {
      setBackendState('passive');
    }
  }, [isConnected]);

  // 2. Start Microphone IMMEDIATELY on mount
  useEffect(() => {
    let isMounted = true;

    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        if (!isMounted) return;

        streamRef.current = stream;
        setMicActive(true);

        const audioContext = new window.AudioContext({ sampleRate: 16_000 });
        audioContextRef.current = audioContext;
        void audioContext.resume().catch(() => {
          // Browser policies may still block autoplay; stream starts when resumed.
        });

        const source = audioContext.createMediaStreamSource(stream);
        sourceRef.current = source;

        const processor = audioContext.createScriptProcessor(2048, 1, 1);
        processorRef.current = processor;

        const gainNode = audioContext.createGain();
        gainNode.gain.value = 0;

        source.connect(processor);
        processor.connect(gainNode);
        gainNode.connect(audioContext.destination);

        const targetSampleRate = 16_000;
        const samplesPerChunk = 320; // 20ms @ 16kHz
        let firstPacketSent = false;

        const downsampleTo16k = (input: Float32Array, inSampleRate: number): Int16Array => {
          if (inSampleRate === targetSampleRate) {
            const out = new Int16Array(input.length);
            for (let i = 0; i < input.length; i++) {
              const s = Math.max(-1, Math.min(1, input[i]));
              out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
            }
            return out;
          }

          const ratio = inSampleRate / targetSampleRate;
          const outLen = Math.floor(input.length / ratio);
          const out = new Int16Array(outLen);
          let offsetResult = 0;
          let offsetBuffer = 0;

          while (offsetResult < outLen) {
            const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
            let accum = 0;
            let count = 0;
            for (let i = offsetBuffer; i < nextOffsetBuffer && i < input.length; i++) {
              accum += input[i];
              count += 1;
            }
            const sample = count > 0 ? accum / count : 0;
            const clamped = Math.max(-1, Math.min(1, sample));
            out[offsetResult] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
            offsetResult += 1;
            offsetBuffer = nextOffsetBuffer;
          }

          return out;
        };

        processor.onaudioprocess = (e) => {
          const float32 = e.inputBuffer.getChannelData(0);
          const pcm16 = downsampleTo16k(float32, audioContext.sampleRate);
          const carry = pcmCarryRef.current;

          for (let i = 0; i < pcm16.length; i++) {
            carry.push(pcm16[i]);
          }

          while (carry.length >= samplesPerChunk) {
            const chunk = new Int16Array(carry.splice(0, samplesPerChunk));
            sendAudioSegment(chunk.buffer);
          }

          if (!firstPacketSent) {
            console.log('📤 First audio packet successfully sent to backend!');
            firstPacketSent = true;
          }
        };

        console.log('🎤 Continuous microphone streaming initialized');
      })
      .catch((err: DOMException) => {
        setMicError('Microphone permission denied or device not found.');
        console.error('Microphone error:', err);
      });

    return () => {
      isMounted = false;
      pcmCarryRef.current = [];
      processorRef.current?.disconnect();
      sourceRef.current?.disconnect();
      audioContextRef.current?.close();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [sendAudioSegment]);

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader className="text-center">
        <CardTitle>Receptionist AI</CardTitle>
        <CardDescription>Continuous voice assistant — always listening</CardDescription>
      </CardHeader>

      <CardContent className="flex flex-col items-center gap-6">
        {/* Main status orb */}
        <StatusOrb state={backendState} />

        {/* Indicator dots row */}
        <div className="flex gap-6">
          <Dot
            active={isConnected}
            color="bg-green-500"
            label="Connected"
          />
          <Dot
            active={micActive}
            color="bg-blue-500"
            label="Mic Active"
          />
          <Dot
            active={backendState === 'listening'}
            color="bg-blue-400"
            label="Listening"
          />
          <Dot
            active={backendState === 'processing'}
            color="bg-yellow-400"
            label="Processing"
          />
          <Dot
            active={backendState === 'speaking'}
            color="bg-green-400"
            label="Speaking"
          />
        </div>

        {/* Error alert */}
        {micError && (
          <Alert variant="destructive" className="w-full">
            <AlertDescription>{micError}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default VoiceActivityDetector;