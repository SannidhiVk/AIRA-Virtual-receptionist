'use client';

import React, {
  createContext,
  useContext,
  useRef,
  useCallback,
  useState,
  useMemo,
  ReactNode
} from 'react';

interface WordTiming {
  word: string;
  start_time: number;
  end_time: number;
}

type TimingPayload = unknown;

interface WebSocketMessage {
  status?: string;
  client_id?: string;
  interrupt?: boolean;
  audio?: string;
  word_timings?: WordTiming[];
  sample_rate?: number;
  method?: string;
  audio_complete?: boolean;
  error?: string;
  type?: string;
  state?: string;
  name?: string;
  verified?: boolean;
  distance?: number;
  audio_name?: string;
  has_photo?: boolean;
  message?: string;
  person_type?: 'employee' | 'visitor';
  session_action?: 'capture_reference' | 'compare_reference';
  reference_captured?: boolean;
}

export interface FaceVerificationRequestOptions {
  personType?: 'employee' | 'visitor';
  sessionAction?: 'capture_reference' | 'compare_reference';
}

interface WebSocketContextType {
  isConnected: boolean;
  isConnecting: boolean;
  connect: () => Promise<void>;
  disconnect: () => void;
  sendAudioSegment: (audioData: ArrayBuffer) => void;
  sendImage: (imageData: string) => void;
  sendAudioWithImage: (audioData: ArrayBuffer, imageData: string) => void;
  sendWakeWord: () => void;
  onAudioReceived: (
    callback: (
      audioData: string,
      timingData?: TimingPayload,
      sampleRate?: number,
      method?: string
    ) => void
  ) => void;
  onInterrupt: (callback: () => void) => void;
  onError: (callback: (error: string) => void) => void;
  onStatusChange: (
    callback: (status: 'connected' | 'disconnected' | 'connecting') => void
  ) => void;
  onServerState: (
    callback: (
      state: 'passive' | 'listening' | 'processing' | 'speaking'
    ) => void
  ) => void;
  sendFaceVerificationRequest?: (
    audioName: string,
    imageB64: string,
    options?: FaceVerificationRequestOptions
  ) => void;
  onVerificationResult?: (callback: (data: WebSocketMessage) => void) => void;
  onEmployeeIdentified?: (callback: (employeeName: string) => void) => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

export const useWebSocketContext = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error(
      'useWebSocketContext must be used within a WebSocketProvider'
    );
  }
  return context;
};

interface WebSocketProviderProps {
  children: ReactNode;
  serverUrl?: string;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({
  children,
  serverUrl = 'ws://127.0.0.1:8000/ws'
}) => {
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);

  // Generate a unique client ID for this browser session
  const clientId = useMemo(
    () => 'user_' + Math.random().toString(36).substring(7),
    []
  );

  // Create the full URL by combining the base serverUrl and the clientId
  const fullWsUrl = `${serverUrl}/${clientId}`;

  // Callback refs
  const audioReceivedCallbackRef = useRef<
    | ((
        audioData: string,
        timingData?: TimingPayload,
        sampleRate?: number,
        method?: string
      ) => void)
    | null
  >(null);
  const interruptCallbackRef = useRef<(() => void) | null>(null);
  const errorCallbackRef = useRef<((error: string) => void) | null>(null);
  const statusChangeCallbackRef = useRef<
    ((status: 'connected' | 'disconnected' | 'connecting') => void) | null
  >(null);
  const serverStateCallbackRef = useRef<
    | ((state: 'passive' | 'listening' | 'processing' | 'speaking') => void)
    | null
  >(null);
  const faceVerificationResultCallbackRef = useRef<
    ((data: WebSocketMessage) => void) | null
  >(null);
  const employeeIdentifiedCallbackRef = useRef<
    ((employeeName: string) => void) | null
  >(null);

  const connect = useCallback(async () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      setIsConnecting(true);
      statusChangeCallbackRef.current?.('connecting');

      console.log('Attempting to connect to WebSocket at:', fullWsUrl);
      wsRef.current = new WebSocket(fullWsUrl);

      wsRef.current.onopen = () => {
        setIsConnected(true);
        setIsConnecting(false);
        statusChangeCallbackRef.current?.('connected');
        console.log('WebSocket connected');
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);
          console.log('WebSocket message received:', data);

          if (data.status === 'connected') {
            console.log(
              `Server confirmed connection. Client ID: ${data.client_id}`
            );
          }

          if (data.status === 'connected') {
            console.log(
              `Server confirmed connection. Client ID: ${data.client_id}`
            );
          }

          if (data.type === 'face_verification_result') {
            console.log('Received face verification result:', data);
            faceVerificationResultCallbackRef.current?.(data);
          }

          if (
            data.type === 'employee_identified' &&
            typeof data.name === 'string'
          ) {
            employeeIdentifiedCallbackRef.current?.(data.name);
          }

          if (data.state) {
            serverStateCallbackRef.current?.(
              data.state as 'passive' | 'listening' | 'processing' | 'speaking'
            );
          }

          if (data.interrupt) {
            console.log('Received interrupt signal');
            interruptCallbackRef.current?.();
          }

          if (data.audio) {
            // Handle audio with native timing
            let timingData = null;

            if (data.word_timings) {
              // TalkingHead expects wtimes/wdurations in milliseconds.
              timingData = {
                words: data.word_timings.map((wt) => wt.word),
                word_times: data.word_timings.map((wt) =>
                  Number(wt.start_time)
                ),
                word_durations: data.word_timings.map(
                  (wt) => Number(wt.end_time) - Number(wt.start_time)
                )
              };
              console.log('Converted timing data:', timingData);
            }

            const normalizedAudio = data.audio.startsWith('data:')
              ? data.audio.split(',')[1] || ''
              : data.audio;

            console.log('Calling audioReceivedCallback with:', {
              audioLength: normalizedAudio.length,
              timingData,
              sampleRate: data.sample_rate || 24000,
              method: data.method || 'unknown'
            });

            audioReceivedCallbackRef.current?.(
              normalizedAudio,
              timingData,
              data.sample_rate || 24000,
              data.method || 'unknown'
            );
          }

          if (data.audio_complete) {
            console.log('Audio processing complete');
          }

          if (data.error) {
            errorCallbackRef.current?.(data.error);
          }

          if (data.type === 'ping') {
            // Keepalive ping - no action needed
          }
        } catch {
          console.log('Non-JSON message:', event.data);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        errorCallbackRef.current?.('WebSocket connection error');
      };

      wsRef.current.onclose = () => {
        setIsConnected(false);
        setIsConnecting(false);
        statusChangeCallbackRef.current?.('disconnected');
        console.log('WebSocket disconnected');
      };
    } catch {
      setIsConnecting(false);
      errorCallbackRef.current?.('Failed to connect to WebSocket server');
    }
  }, [fullWsUrl]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const sendAudioSegment = useCallback((audioData: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(audioData);
    }
  }, []);

  const sendWakeWord = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ wake_word_detected: true }));
      console.log('Wake word event sent to server');
    }
  }, []);

  const sendImage = useCallback((imageData: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const message = {
        image: imageData
      };

      wsRef.current.send(JSON.stringify(message));
      console.log('Sent image to server');
    }
  }, []);

  const sendAudioWithImage = useCallback(
    (audioData: ArrayBuffer, imageData: string) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        // Convert ArrayBuffer to base64
        const bytes = new Uint8Array(audioData);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        const base64Audio = btoa(binary);

        const message = {
          audio_segment: base64Audio,
          image: imageData
        };

        wsRef.current.send(JSON.stringify(message));
        console.log(`Sent audio + image: ${audioData.byteLength} bytes audio`);
      }
    },
    []
  );

  const sendFaceVerificationRequest = useCallback(
    (
      audioName: string,
      imageB64: string,
      options?: FaceVerificationRequestOptions
    ) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const personType = options?.personType ?? 'employee';
        const sessionAction =
          options?.sessionAction ??
          (personType === 'visitor' ? 'compare_reference' : undefined);
        wsRef.current.send(
          JSON.stringify({
            type: 'verify_face',
            audio_name: audioName,
            image_b64: imageB64,
            person_type: personType,
            ...(sessionAction ? { session_action: sessionAction } : {})
          })
        );
        console.log(
          `Sent face verification request for: ${audioName} (${personType}${sessionAction ? `:${sessionAction}` : ''})`
        );
      }
    },
    []
  );

  // Callback registration methods
  const onVerificationResult = useCallback(
    (callback: (data: WebSocketMessage) => void) => {
      faceVerificationResultCallbackRef.current = callback;
    },
    []
  );

  const onEmployeeIdentified = useCallback(
    (callback: (employeeName: string) => void) => {
      employeeIdentifiedCallbackRef.current = callback;
    },
    []
  );

  const onAudioReceived = useCallback(
    (
      callback: (
        audioData: string,
        timingData?: TimingPayload,
        sampleRate?: number,
        method?: string
      ) => void
    ) => {
      audioReceivedCallbackRef.current = callback;
    },
    []
  );

  const onInterrupt = useCallback((callback: () => void) => {
    interruptCallbackRef.current = callback;
  }, []);

  const onError = useCallback((callback: (error: string) => void) => {
    errorCallbackRef.current = callback;
  }, []);

  const onStatusChange = useCallback(
    (
      callback: (status: 'connected' | 'disconnected' | 'connecting') => void
    ) => {
      statusChangeCallbackRef.current = callback;
    },
    []
  );

  const onServerState = useCallback(
    (
      callback: (
        state: 'passive' | 'listening' | 'processing' | 'speaking'
      ) => void
    ) => {
      serverStateCallbackRef.current = callback;
    },
    []
  );

  const value: WebSocketContextType = {
    isConnected,
    isConnecting,
    connect,
    disconnect,
    sendAudioSegment,
    sendImage,
    sendAudioWithImage,
    sendWakeWord,
    onAudioReceived,
    onInterrupt,
    onError,
    onStatusChange,
    onServerState,
    sendFaceVerificationRequest,
    onVerificationResult,
    onEmployeeIdentified
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};
