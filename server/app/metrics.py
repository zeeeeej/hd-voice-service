"""Prometheus 指标。"""
from prometheus_client import Counter, Gauge, Histogram

asr_sessions_active = Gauge("voice_asr_sessions_active", "active ASR WS sessions")
tts_sessions_active = Gauge("voice_tts_sessions_active", "active TTS WS sessions")
asr_sessions_total = Counter("voice_asr_sessions_total", "ASR sessions accepted")
tts_requests_total = Counter("voice_tts_requests_total", "TTS requests (ws+rest)")
rest_requests_total = Counter("voice_rest_requests_total", "REST requests", ["path", "status"])
rejected_total = Counter("voice_rejected_total", "429/1013 rejections", ["kind"])
errors_total = Counter("voice_errors_total", "5xx errors", ["path"])

denoise_chunk_ms = Histogram("voice_denoise_chunk_ms", "denoise chunk latency ms", buckets=(1, 5, 10, 25, 50, 100, 250))
asr_chunk_ms = Histogram("voice_asr_chunk_ms", "asr feed+decode latency ms", buckets=(5, 10, 25, 50, 100, 250, 500, 1000))
asr_first_partial_ms = Histogram("voice_asr_first_partial_ms", "speech_start -> first partial ms", buckets=(50, 100, 200, 300, 500, 800, 1500))
asr_final_ms = Histogram("voice_asr_final_ms", "speech_end -> final emitted ms", buckets=(10, 50, 100, 250, 500, 1000))
tts_first_chunk_ms = Histogram("voice_tts_first_chunk_ms", "sentence -> first audio chunk ms", buckets=(50, 100, 200, 400, 800, 1500, 3000))
refine_ms = Histogram("voice_refine_ms", "sense-voice refine latency ms", buckets=(50, 100, 250, 500, 1000, 2000, 4000))
punc_ms = Histogram("voice_punc_ms", "punctuation latency ms", buckets=(10, 50, 100, 250, 500, 1000))
refine_total = Counter("voice_refine_total", "refine attempts", ["result"])  # ok|fallback
tts_sentence_ms = Histogram("voice_tts_sentence_ms", "sentence synthesis total ms", buckets=(100, 250, 500, 1000, 2000, 4000))
