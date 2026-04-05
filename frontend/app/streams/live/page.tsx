import { Navbar } from "@/components";
import MediasoupLivePlayer from "@/components/MediasoupLivePlayer";

export default function LiveMediasoupPlayerPage() {
  return (
    <div className="min-h-screen bg-gray-950">
      <Navbar />

      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-white">WebRTC Live Player</h1>
          <p className="mt-2 max-w-3xl text-gray-400">
            Ultra-low-latency live playback using mediasoup-client consumer flow
            over secure WebSocket signaling.
          </p>
        </header>

        <MediasoupLivePlayer />

        <section className="mt-6 rounded-xl border border-slate-800 bg-slate-900/70 p-4 text-sm text-slate-300">
          <h2 className="mb-2 text-base font-semibold text-slate-100">
            Expected signaling actions
          </h2>
          <p className="mb-2">
            Server should support these request actions from the browser:
          </p>
          <ul className="list-disc space-y-1 pl-6 text-slate-400">
            <li>getRouterRtpCapabilities</li>
            <li>createConsumerTransport</li>
            <li>connectConsumerTransport</li>
            <li>consume</li>
            <li>resumeConsumer</li>
          </ul>
        </section>
      </main>
    </div>
  );
}
