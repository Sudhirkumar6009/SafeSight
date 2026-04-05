# Mediasoup WebRTC Live Player

This frontend includes a low-latency WebRTC player at:

- /streams/live

It uses mediasoup-client and secure WebSocket signaling to consume an existing producer stream.

## 1) Install dependency

If not already installed:

```bash
npm install mediasoup-client
```

## 2) Configure frontend env

Add these variables in frontend/.env.local:

```env
NEXT_PUBLIC_MEDIASOUP_SIGNALING_URL=wss://your-mediasoup-signaling.example/ws
NEXT_PUBLIC_MEDIASOUP_ROOM_ID=default-room
NEXT_PUBLIC_MEDIASOUP_PRODUCER_ID=
NEXT_PUBLIC_MEDIASOUP_STUN_SERVERS=stun:stun.l.google.com:19302,stun:stun1.l.google.com:19302
```

Notes:

- Use wss:// in production.
- If the app is served over HTTPS and WS URL is ws://, the client upgrades it to wss://.

## 3) Signaling message contract

Client requests include:

```json
{
  "requestId": "uuid",
  "action": "getRouterRtpCapabilities",
  "type": "getRouterRtpCapabilities",
  "data": { "roomId": "default-room" }
}
```

The player accepts any of these response envelope styles:

```json
{ "requestId": "uuid", "ok": true, "data": { "...": "..." } }
{ "responseTo": "uuid", "success": true, "data": { "...": "..." } }
{ "requestId": "uuid", "error": "message" }
```

Required actions (or compatible aliases server-side):

1. getRouterRtpCapabilities
2. createConsumerTransport
3. connectConsumerTransport
4. consume
5. resumeConsumer

### Expected response payloads

Router RTP capabilities:

```json
{ "rtpCapabilities": { "codecs": [], "headerExtensions": [] } }
```

Consumer transport:

```json
{
  "id": "transport-id",
  "iceParameters": {
    "usernameFragment": "...",
    "password": "...",
    "iceLite": true
  },
  "iceCandidates": [
    {
      "foundation": "...",
      "ip": "...",
      "protocol": "udp",
      "port": 40000,
      "type": "host"
    }
  ],
  "dtlsParameters": {
    "role": "auto",
    "fingerprints": [{ "algorithm": "sha-256", "value": "..." }]
  }
}
```

Consume response:

```json
{
  "id": "consumer-id",
  "producerId": "producer-id",
  "kind": "video",
  "rtpParameters": { "codecs": [], "encodings": [], "headerExtensions": [] }
}
```

## 4) Browser compatibility and latency

The player is built for modern browsers (Chrome/Edge/mobile Chromium + Safari with WebRTC support).

Low-latency behavior comes from:

- WebRTC transport (DTLS + SRTP)
- No MSE/HLS segment buffering path
- Immediate track attachment to HTML5 video
- Automatic reconnect on transport/signaling drops

## 5) How to test end-to-end

1. Ensure mediasoup backend is running and a producer already exists (FFmpeg/RTSP -> mediasoup).
2. Start frontend and open /streams/live.
3. Confirm status changes:
   - Connecting -> Connected
4. If stream drops, verify status transitions to Reconnecting and returns to Connected.
