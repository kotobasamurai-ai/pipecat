# エラー注入テスト手順書

## 概要

Soniox STT / Fish Audio TTS の failover・reconnect 動作を検証するためのエラー注入機能。
環境変数で制御し、デフォルトは全て無効（`0`）。

## 環境変数一覧

### Soniox STT

| 環境変数 | 説明 | 値の範囲 |
|---|---|---|
| `SONIOX_STT_CONNECT_ERROR_INJECT_RATE` | WebSocket 接続時に失敗させる確率 | 0.0〜1.0 |
| `SONIOX_STT_STREAM_ERROR_INJECT_RATE` | ストリーミング中に切断する確率 | 0.0〜1.0 |
| `SONIOX_STT_STREAM_ERROR_AFTER_MSGS` | ↑ 何メッセージ受信後にエラーを起こすか | 整数 (デフォルト: 10) |
| `SONIOX_STT_SEND_ERROR_INJECT_RATE` | 音声送信時に失敗させる確率 | 0.0〜1.0 |

### Fish Audio TTS

| 環境変数 | 説明 | 値の範囲 |
|---|---|---|
| `FISH_TTS_ERROR_INJECT_RATE` | TTS 合成時にエラーを注入する確率 | 0.0〜1.0 |

---

## テスト手順

### 前提

```bash
cd /home/yuki/ai-contact/kotoba-agent/server
```

pipecat の custom ブランチが最新であること:
```bash
uv lock --upgrade-package pipecat-ai && uv sync
```

---

### テスト 1: STT 接続失敗 → ServiceSwitcher failover

```bash
SONIOX_STT_CONNECT_ERROR_INJECT_RATE=1.0 uv run bot.py --flow agents/definitions/configs/UME_サポート_team.yaml
```

**期待動作:**
- Soniox 接続が 100% 失敗
- ServiceSwitcher が Speechmatics に切り替え
- 挨拶が再生される
- 発話 → Speechmatics で認識 → LLM 返答 → TTS 再生

**確認ログ:**
```
[DEBUG] Injecting synthetic connection error (rate=1.0)
LazySTT(speechmatics/ja) lazily initializing STT service
SpeechmaticsSTTService#0 connected
```

---

### テスト 2: STT ストリーム中切断 → reconnect

```bash
SONIOX_STT_STREAM_ERROR_INJECT_RATE=1.0 SONIOX_STT_STREAM_ERROR_AFTER_MSGS=5 uv run bot.py --flow agents/definitions/configs/UME_サポート_team.yaml
```

**期待動作:**
- Soniox 接続は成功
- 5 メッセージ受信後に切断 → 自動 reconnect → 成功
- reconnect のたびに再度 5 メッセージ後に切断（rate=1.0 のため）
- 音声認識は断続的だが、パイプラインはクラッシュしない

**確認ログ:**
```
[DEBUG] Injecting synthetic stream error after 5 messages (rate=1.0)
SonioxSTTService#0 error receiving messages: ...
SonioxSTTService#0 reconnecting, attempt 1
SonioxSTTService#0 reconnected successfully on attempt 1
```

**備考:** reconnect は pipecat の `WebsocketService` base class が実装。最大 3 回リトライ（exponential backoff）。全失敗時は fatal ErrorFrame → ServiceSwitcher failover。

---

### テスト 3: TTS エラー → retry → 成功

```bash
FISH_TTS_ERROR_INJECT_RATE=0.3 uv run bot.py --flow agents/definitions/configs/UME_サポート_team.yaml
```

**期待動作:**
- 30% の確率で TTS 合成が失敗 → retry で再生成
- retry 成功時は音声が聞こえる（順序がずれる場合あり）

**確認ログ:**
```
[DEBUG] Injecting synthetic TTS error (rate=0.3)
[tts_retry] TTS error on CachedFishAudioTTSService#0, retrying (1/3)
[tts_retry] ✅ Retry SUCCESS
```

---

### テスト 4: TTS エラー → retry 全失敗 → Azure failover

```bash
FISH_TTS_ERROR_INJECT_RATE=1.0 uv run bot.py --flow agents/definitions/configs/UME_サポート_team.yaml
```

**期待動作:**
- Fish Audio が 100% 失敗 → retry 3 回全て失敗 → Azure TTS に切り替え
- Azure の声で挨拶〜返答が再生される

**確認ログ:**
```
[DEBUG] Injecting synthetic TTS error (rate=1.0)
[tts_retry] retrying (1/3) ... (2/3) ... (3/3)
Retry exhausted ... failing over
AzureTTSService#0 ...
```

---

### テスト 5: STT + TTS 同時エラー（最悪ケース）

```bash
SONIOX_STT_CONNECT_ERROR_INJECT_RATE=1.0 FISH_TTS_ERROR_INJECT_RATE=1.0 uv run bot.py --flow agents/definitions/configs/UME_サポート_team.yaml
```

**期待動作:**
- Soniox → Speechmatics failover
- Fish Audio → Azure failover
- 全てバックアップサービスで会話が成立する

---

## 既知の問題

### STT 接続失敗時の挨拶タイムアウト

`SONIOX_STT_CONNECT_ERROR_INJECT_RATE=1.0` の場合、STT failover の初期化遅延により Fish Audio のレスポンスが audio context の 3 秒タイムアウトに間に合わず、挨拶が再生されないことがある。

### ServiceSwitcher の ServiceMetadataFrame フィルタリング

STT failover 後、LazySTTProxy 経由で初期化された Speechmatics の `STTMetadataFrame` が ServiceSwitcher の名前チェック（`service_switcher.py:311-314`）でドロップされる。Smart Turn の `_stt_timeout` が更新されない可能性がある。

### TTS retry 時の文順序

エラー注入で失敗した文は retry 後に再生されるが、後続の文が先に再生される場合がある（文順序の逆転）。

---

## ブランチ構成

| ブランチ | ベース | 内容 |
|---|---|---|
| `fish-tts-error-injection` | main | Fish TTS エラー注入 |
| `soniox-stt-error-injection` | main | Soniox STT エラー注入 |
| `custom` | - | ↑ 両方をマージ済み + プロジェクト固有の変更 |

`ai-contact` の `pyproject.toml` は `pipecat-ai @ git+...@custom` で参照。
変更後は `uv lock --upgrade-package pipecat-ai && uv sync` で反映。
