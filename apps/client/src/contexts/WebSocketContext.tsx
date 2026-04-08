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
      timingData?: any,
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
        timingData?: any,
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
              const usesMilliseconds = data.word_timings.some(
                (wt) => wt.start_time > 20 || wt.end_time > 20
              );
              const toSeconds = (value: number) =>
                usesMilliseconds ? value / 1000 : value;

              // Convert to TalkingHead format
              timingData = {
                words: data.word_timings.map((wt) => wt.word),
                word_times: data.word_timings.map((wt) =>
                  toSeconds(wt.start_time)
                ),
                word_durations: data.word_timings.map(
                  (wt) => toSeconds(wt.end_time) - toSeconds(wt.start_time)
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
        } catch (e) {
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
    } catch (error) {
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

  // Callback registration methods
  const onAudioReceived = useCallback(
    (
      callback: (
        audioData: string,
        timingData?: any,
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
    onServerState
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};
