# FastWAM StarVLA Placement Decisions

## Goal

FastWAM を StarVLA 上で論文設定に近い形でフルスクラッチ学習できるようにする。
方針は「StarVLA の既存 module に吸収できるものは吸収し、吸収すると既存責務や挙動を壊すものだけ公式 FastWAM ファイルを近い module 配下へ置く」です。

## Placement Rule

- 既存 StarVLA 実装が同じ責務を満たし、FastWAM の必要 interface を無理なく追加できる場合は既存実装を再利用する。
- 既存実装が wrapper で、FastWAM が内部 attention や token 化を直接制御する必要がある場合は、既存ファイルへ吸収しない。
- 公式 FastWAM ファイルを残す場合も、新しい大きなフォルダを作らず StarVLA の既存 module 階層へ置く。
- 既存の `WanGR00T`、`WanPI`、`WanOFT`、`WanFastWAM` の挙動を変えない名前と registry で追加する。

## Current Mapping

| FastWAM source | StarVLA placement | Decision | Reason |
| --- | --- | --- | --- |
| `action_dit.py` | `starVLA/model/modules/action_model/FastWAM_ActionDiT.py` | 公式実装を action module に配置 | Action denoising expert なので StarVLA の `action_model` 責務に一致する。既存 action head へ無理に混ぜると diffusion action expert の独自構造が崩れる。 |
| `wan_video_dit.py` | `starVLA/model/modules/world_model/FastWAM_WanVideoDiT.py` | 公式実装を world model module に配置 | FastWAM は video DiT の q/k/v、self-attention mask、patchify/unpatchify、action-conditioned context を直接制御する。既存 `Wan2.py` は diffusers wrapper で内部 attention へ十分アクセスできない。 |
| `mot.py` | `starVLA/model/framework/WM4A/FastWAM.py` に吸収 | 公式実装を framework entry 内へ統合 | StarVLA の `framework` は原則として world model と action head の統合層であり、中核 network module を別ファイルとして `framework` 配下へ置くのは既存設計とずれる。MoT は FastWAM 専用で、現状ほかの framework から参照されないため、`FastWAM.py` に局所化して新規 framework module ファイルを減らす。 |
| `fastwam.py` | `starVLA/model/framework/WM4A/FastWAM.py` | StarVLA framework entry として新規作成 | StarVLA の registry、training loop、config に接続する入口。公式 `FastWAM` の責務を StarVLA の `baseframework` interface に合わせる場所。 |
| `scheduler_continuous.py` | `starVLA/model/framework/WM4A/FastWAM.py` | 小さい scheduler class を吸収 | video/action training loss に密接で、単独 module にする必要が薄い。新規ファイル増加を避けつつ公式ロジックを保持できる。 |
| `wan_video_vae.py`, `wan_video_text_encoder.py` | 公式ファイルは未配置。`FastWAM.py` に diffusers encoder adapter を吸収 | 条件付き再利用 | StarVLA 既存 `Wan2.py` は `AutoencoderKLWan` と `UMT5EncoderModel` のロード経路を持つが、`_Wan2_Interface` をそのまま使うと transformer もロードする。FastWAM 学習 loss には VAE/text encoder だけ必要なので、同じ diffusers 経路を `FastWAM.py` に遅延ロード adapter として吸収する。 |
| `helpers/loader.py`, `helpers/io.py`, `helpers/state_dict_converters.py` | 未配置 | 条件付き | 公式重み形式の読み込みや変換が必要になった場合だけ、最小限を framework または world_model 配下へ追加する。diffusers/HF 経路だけで足りるなら置かない。 |

## Why Not Absorb `WanVideoDiT` Into `Wan2.py`

`Wan2.py` は StarVLA の既存 world-model wrapper で、diffusers の `WanTransformer3DModel` をロードして hidden states を取り出す役割を持つ。
一方で FastWAM の `WanVideoDiT` は DiT 本体であり、以下を内部で制御する。

- latent の patchify/unpatchify
- 3D RoPE
- video-to-video self-attention mask
- action-conditioned context token
- MoT から各 layer の self-attention q/k/v を受け渡す経路

これらを `Wan2.py` に入れると、既存の `WanGR00T`、`WanPI`、`WanOFT` が使う wrapper の責務が変わり、diffusers 互換の推論経路にも影響する。
そのため `Wan2.py` は壊さず、FastWAM 用 DiT は `world_model/FastWAM_WanVideoDiT.py` として分離する。

## How This Prevents Breakage

- 既存 `Wan2.py` を変更しないため、既存 Wan 系 framework のロード、生成、hidden-state 抽出に影響しない。
- full FastWAM は registry 名 `FastWAM` で追加し、既存 prototype の `WanFastWAM` と分ける。
- 公式 FastWAM の大きな責務単位を保持するため、MoT と DiT の interface を途中で崩さない。
- scheduler は小さく framework 専用なので `FastWAM.py` に吸収し、新規ファイル数を増やさない。
- `py_compile` と StarVLA framework registry の auto discovery で import 破損を確認する。

## MoT Integration Decision

当初は FastWAM 公式 `mot.py` を `starVLA/model/framework/WM4A/FastWAM_MoT.py` として配置した。
これは MoT が video expert と action expert を層ごとに混ぜる中核実装であり、`FastWAM.py` から分けた方が読みやすかったため。

ただし StarVLA 既存設計を見ると、`framework/WM4A` には基本的に以下の責務が置かれている。

- world model と action head の組み立て
- StarVLA trainer から呼ばれる `forward()` / `predict_action()` の入口
- framework registry 登録
- config の解釈

一方で、Transformer block や action head などの中核 network module は `model/modules/...` 配下に置かれている。
この観点では、MoT だけを `framework/WM4A/FastWAM_MoT.py` として独立させると、StarVLA の既存配置規則から外れやすい。

選択肢は以下だった。

1. `FastWAM_MoT.py` を維持する。
2. MoT を `model/modules` 配下へ移す。
3. MoT を `FastWAM.py` に吸収する。

現時点では 3 を選んだ。
理由は、MoT が現在 `FastWAM.py` 専用であり、他の framework から再利用されていないため。
`model/modules` 配下へ置く選択もあり得るが、MoT は video module 単体でも action module 単体でもなく、FastWAM 専用の統合機構である。
そのため、別の module 階層を新設するより `FastWAM.py` に局所化する方が、現在の「新規ファイルをできるだけ増やさない」方針に合う。

### Risk

MoT を `FastWAM.py` に吸収するリスクは以下。

- `FastWAM.py` が大きくなる。
- 公式 FastWAM の `mot.py` との差分追跡が少し難しくなる。
- 将来 `FastWAMJoint` / `FastWAMIDM` など複数 variant が同じ MoT を共有する場合、再度分離したくなる可能性がある。
- MoT の単体テストを追加する場合、import path が `FastWAM.MoT` になる。

一方、壊れる可能性が低い理由は以下。

- `FastWAM_MoT.py` を参照していた実コードは `FastWAM.py` の import のみだった。
- MoT class 本体はそのまま移動し、内部ロジックは変更していない。
- `flash_attention`、`modulate`、`rope_apply` への依存は `FastWAM.py` から直接 import する形にした。
- `MoT` は registry 登録対象ではなく、外部設定からファイル名指定される module でもない。
- smoke test で `build_framework()`、training loss、action inference 経路を確認できる。

### Mitigation

統合後の保守リスクを抑えるため、以下を守る。

- MoT class は `FastWAM.py` 内でも独立 class として残し、`FastWAMFramework` にメソッドとして混ぜ込まない。
- MoT の内部処理は、公式 `mot.py` 由来のまとまりを崩さない。
- `FastWAM.py` の framework entry 部分と MoT class 部分を分けて読めるよう、class 境界を維持する。
- 将来 MoT を複数 FastWAM variant で共有する必要が出た場合は、`model/modules` 配下への再分離を検討する。

検証済み:

```bash
.venv-py311/bin/python -m py_compile starVLA/model/framework/WM4A/FastWAM.py starVLA/model/modules/world_model/FastWAM_WanVideoDiT.py starVLA/model/modules/action_model/FastWAM_ActionDiT.py
.venv-py311/bin/python -m unittest tests.test_fastwam_smoke
```

結果:

```text
Ran 11 tests in 1.288s
OK (skipped=1)
```

## Current Progress

- `FastWAM_ActionDiT.py`: 配置済み。
- `FastWAM_WanVideoDiT.py`: 配置済み。StarVLA logger/import/checkpoint helper に合わせた最小変更のみ。
- `FastWAM_MoT.py`: 削除済み。MoT class は `FastWAM.py` に吸収。
- `FastWAM.py`: registry entry、expert 構築、MoT class、scheduler 吸収、precomputed latent training loss、Wan2 diffusers encoder adapter 入口、precomputed latent action inference まで完了。
- `starVLA/config/training/starvla_fastwam_encoder_smoke.yaml`: 追加済み。Wan2 VAE/text encoder を実ロードするための小型 FastWAM smoke config。
- `tests/test_fastwam_smoke.py`: 追加済み。tiny config で scheduler、framework build、video/action expert と MoT の token-level forward、precomputed latent forward/backward、raw examples adapter routing、encoder smoke config 配線、`predict_action()` の shape を確認する。

## Smoke Test Scope

この smoke test は、実データや Wan2.2 本番重みを使わず、以下の破損を早く検出する目的で置く。

- `FastWAM` が StarVLA の `build_framework()` 経由で構築できるか。
- `WanContinuousFlowMatchScheduler` の timestep/noise/weight shape が training loss 用に使えるか。
- `WanVideoDiT.pre_dit()` と `ActionDiT.pre_dit()` が MoT へ渡す token/freq/context/timestep modulation の shape 契約を満たすか。
- MoT が video expert と action expert を同じ layer interface で扱えるか。
- `input_latents/context/context_mask/action` が事前計算済みの場合に `FastWAM.forward()` が `action_loss` を返し、backward できるか。
- StarVLA trainer 互換の `compute_loss("vla", batch)` 経由でも `action_loss` を返せるか。
- StarVLA examples 形式の raw `image`/`video` + `lang` が encoder adapter を通って loss 経路に入るか。
- `framework.encoder.load_wan2_encoders=false` のまま raw examples が来た場合、precomputed latent が必要だと明確に失敗するか。
- `framework.encoder.load_wan2_encoders=true` の smoke config が `FastWAM`、Wan2 latent dim `48`、text dim `4096` に揃っているか。
- `first_frame_latents/context/context_mask` が事前計算済みの場合に `predict_action()` が StarVLA 互換の `normalized_actions` を返せるか。
- action-only 推論が `video_attention_mask_mode='first_frame_causal'` を要求し、誤設定なら明確に失敗するか。

この段階では実 dataloader batch はまだ確認対象外。
実物の VAE/text encoder ロードは optional test として追加済みだが、現在のローカル環境には Wan2 diffusers モデル実体が見つからないため通常テストでは skip する。

## Training Loss Integration

公式 FastWAM の `training_loss()` は `build_inputs()` で video を VAE latent 化し、text encoder で context を作る前提だった。
StarVLA 側では VAE/text encoder の再利用方針をまだ確定していないため、先に以下の precomputed sample 経路を実装した。

- `input_latents`: `[B, C, T, H, W]`
- `context`: `[B, L, D]`
- `context_mask`: `[B, L]`
- `action`: `[B, T_action, action_dim]`
- optional `image_is_pad`: latent step に揃えた `[B, T_latent]`
- optional `action_is_pad`: `[B, T_action]`

この形なら StarVLA trainer が `compute_loss("vla", batch)` から `FastWAM.forward()` を呼び、`{"action_loss": loss}` を受け取れる。
画像列から `input_latents/context` を作る入口も追加したが、実物の Wan2 encoder ロードは次に smoke config で確認する。

## Encoder Adapter Decision

`Wan2.py` の `_Wan2_Interface` は text encoder、VAE、transformer をまとめてロードする world-model wrapper である。
FastWAM の `training_loss()` では transformer は `FastWAM_WanVideoDiT.py` 側を使うため、`_Wan2_Interface` を直接保持すると不要な diffusers transformer までメモリに載る。

そのため、公式 FastWAM の `wan_video_vae.py` / `wan_video_text_encoder.py` はまだ追加せず、`FastWAM.py` に以下だけを吸収した。

- `framework.encoder.load_wan2_encoders=true` のときだけ遅延ロードする。
- VAE は diffusers `AutoencoderKLWan` を使う。
- text encoder は diffusers 版 Wan2 と同じ `T5TokenizerFast` + `UMT5EncoderModel` を使う。
- raw StarVLA examples は `image`/`video` + `lang`/`prompt` + `action` から `input_latents/context/context_mask/action` に変換する。
- デフォルトは `load_wan2_encoders=false` とし、重い encoder を意図せずロードしない。

この判断により、新しい FastWAM 公式 VAE/text encoder ファイルを追加せず、既存 `Wan2.py` と同じ diffusers 経路で学習 adapter を進められる。

## Action Inference Integration

公式 FastWAM の `infer_action()` は、first-frame video token を一度 MoT に通して layer-wise K/V cache を作り、action expert だけを flow scheduler で反復 denoise する。
StarVLA 側では以下の形に移植した。

- `predict_action()` は `{"normalized_actions": np.ndarray[B, T, action_dim]}` を返す。
- 事前計算済み `first_frame_latents/context/context_mask`、または `input_latents/context/context_mask` から推論できる。
- raw `image`/`video` + `lang` も encoder adapter が有効なら `first_frame_latents/context/context_mask` に変換できる。
- action-only 推論は `video_attention_mask_mode='first_frame_causal'` を要求する。

この段階では実物 Wan2 encoder から raw image を latent 化する推論 smoke は未実行。precomputed latent 経路は tiny test 済み。

## Inference Readiness Detail

現状の FastWAM 構成では、推論経路は大きく 2 種類に分かれる。

1. precomputed latent/context を直接渡す推論
2. raw image/video + text instruction から Wan2 encoder を通して latent/context を作る推論

### 1. precomputed latent/context 推論

この経路は現在動作確認済み。
入力として以下を直接渡す。

- `first_frame_latents`: `[B, C, 1, H, W]`
- `context`: `[B, L, D]`
- `context_mask`: `[B, L]`

この場合、Wan2 VAE や text encoder をロードしなくても `FastWAM.predict_action()` を実行できる。
ただし、ここで得られる action は現在の重みに依存するため、ランダム初期化のままでは値そのものに意味はない。
この確認の目的は、以下の構造が壊れていないことを確認することにある。

- `FastWAM.py` が StarVLA framework registry から build できる。
- `FastWAM_WanVideoDiT.py` が first-frame latent を video token に変換できる。
- `FastWAM_ActionDiT.py` が noisy action token を作れる。
- `FastWAM.py` に吸収した `MoT` class が video token の K/V cache を作り、action token を video token に attend させられる。
- flow scheduler による action denoise loop が最後まで走る。
- StarVLA 互換形式で `{"normalized_actions": np.ndarray[B, T, action_dim]}` を返せる。

確認済みコマンド:

```bash
.venv-py311/bin/python -m unittest tests.test_fastwam_smoke.FastWAMSmokeTest.test_predict_action_returns_normalized_actions_for_precomputed_inputs
```

結果:

```text
OK
```

この smoke test では tiny config を使っている。
つまり本番サイズの Wan2.2 / FastWAM ではなく、1 layer の小さい video expert、action expert、MoT で構造だけを検証している。

### 2. raw image/video + text instruction 推論

この経路は、現在のローカル環境では未実行。
理由は Wan2 diffusers モデル実体が見つからないため。

必要なローカルパスとして現在 config が参照している場所:

```text
/home/anpan/WM/starVLA/playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

確認済みコマンド:

```bash
FASTWAM_RUN_ENCODER_SMOKE=1 .venv-py311/bin/python -m unittest -v tests.test_fastwam_smoke.FastWAMSmokeTest.test_optional_wan2_encoder_load_smoke
```

結果:

```text
skipped 'Wan2 diffusers model path does not exist: /home/anpan/WM/starVLA/playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers'
```

raw image/video 推論に必要なもの:

- Wan2 diffusers 形式の `tokenizer`
- Wan2 diffusers 形式の `text_encoder`
- Wan2 diffusers 形式の `vae`

`FastWAM.py` では `framework.encoder.load_wan2_encoders=true` の場合だけ、これらを遅延ロードする。
これは、precomputed latent/context の実験では重い encoder を不要にロードしないため。

### Meaningful Inference Requirement

構造上の推論と、意味のある推論は別物として扱う。

構造上の推論:

- ランダム初期化重みでも実行できる。
- shape、interface、MoT 接続、scheduler loop の確認が目的。
- 出力 action の数値には意味がない。

意味のある推論:

- 学習済み、またはこれからフルスクラッチ学習した FastWAM checkpoint が必要。
- 少なくとも `video_expert`、`action_expert`、`mot` の重みが必要。
- raw image/video を入力する場合は Wan2 VAE/text encoder も必要。

今回の実装目標は FastWAM を StarVLA 上でフルスクラッチ学習することなので、最終的には以下の順で進めるのが自然。

1. precomputed latent/context で framework 構造を維持する。
2. Wan2 VAE/text encoder を用意して raw image/video adapter を通す。
3. StarVLA dataloader から実 batch を流し、`training_loss()` を確認する。
4. フルスクラッチ学習を開始する。
5. 保存 checkpoint から `predict_action()` を実行する。

### MoT Action Inference Flow

action-only 推論では、video を毎 step 生成するのではなく、first-frame latent を条件として action を denoise する。
処理の流れは以下。

1. first-frame latent を `FastWAM_WanVideoDiT.pre_dit()` に入れる。
2. video token、video RoPE、video timestep modulation、text context を作る。
3. MoT が video 側の各 layer K/V cache を作る。
4. action はランダムノイズから開始する。
5. 各 action denoise step で `FastWAM_ActionDiT.pre_dit()` が action token を作る。
6. MoT の `forward_action_with_video_cache()` が、action token を video K/V cache に attend させる。
7. `FastWAM_ActionDiT.post_dit()` が action noise/velocity を出す。
8. flow scheduler が action latent を 1 step 更新する。
9. step を繰り返し、最後の action latent を `normalized_actions` として返す。

中学生向けに言い換えると、MoT action inference は以下のような処理。

- 最初の画像を見て「場面のメモ」を作る。
- 行動は最初はでたらめなノイズから始める。
- 行動担当は、毎回その「場面のメモ」を見ながら、自分の答えを少しずつ直す。
- 何回か直した最後の答えを、ロボットの行動として出す。

このため、`predict_action()` には `video_attention_mask_mode='first_frame_causal'` が必要。
これは action が first-frame video token を条件として参照する前提で K/V cache を作るためである。

## Next Work

1. 実際の StarVLA dataloader examples で raw adapter を小さい batch から確認する。
2. Wan2 diffusers モデル実体のローカルパスを用意し、`FASTWAM_RUN_ENCODER_SMOKE=1` で optional encoder load test を実行する。
3. 公式 FastWAM checkpoint と dataset stats を `checkpoints/fastwam_release/` へ置く。
4. 公式 checkpoint を StarVLA FastWAM へロードし、shape/key対応を確認する。
5. 公式 checkpoint に含まれる `proprio_encoder` を StarVLA側へ吸収し、state/proprioをcontextへ追加する経路を実装する。
6. raw image・language・proprio/state から `predict_action()` までの実物 encoder smoke を実行する。
7. StarVLA dataloader の実 batch で `training_loss()` を確認する。
8. FastWAM をフルスクラッチ学習する。

2026-08-02更新:

- 公式READMEのcheckpoint 4ファイルは、StarVLAローカルの `checkpoints/fastwam_release/` に配置済み。
- `checkpoints/` は `.gitignore` に追加済み。合計約23GBの外部成果物であり、gitには入れない。
- `libero_uncond_2cam224.pt` は `mot` 1649 keys、`action_dim=7`、`proprio_dim=8`。
- `robotwin_uncond_3cam_384.pt` は `mot` 1649 keys、`action_dim=14`、`proprio_dim=14`。
- 両checkpointとも top-level keys は `mot`, `proprio_encoder`, `step`, `torch_dtype`。
- StarVLA側の `load_checkpoint()` は `payload["mot"]` に対応済みだが、現状は `proprio_encoder` を無視する。そのため公式checkpointを使った意味のあるAction推論には、公式 `FastWAM._append_proprio_to_context()` 相当のproprio経路をStarVLA側へ吸収する必要がある。

2026-08-04更新:

- StarVLA側へ `proprio_encoder` を実装済み。
- `framework.action_model.proprio_dim` を追加した。通常は `None` で無効。
- 公式checkpointに `proprio_encoder` が含まれる場合は、checkpointのweight shapeから `proprio_dim` を自動判定して `nn.Linear(proprio_dim, text_dim)` を構築する。
- `payload["mot"]` に加えて `payload["proprio_encoder"]` をロードする。
- trainingでは `sample["proprio"]` または `sample["state"]` を受け取り、公式実装と同じくcontext末尾へ1トークン追加する。
- `predict_action()` でも `proprio` または `state` を受け取り、context末尾へ1トークン追加する。
- ABCIで公式checkpointを確認するため、`starvla_fastwam_libero_official_infer.yaml` と `starvla_fastwam_robotwin_official_infer.yaml` を追加した。
- `tests.test_fastwam_smoke` は17件通過。proprioのcontext追加とcheckpointロードを小型構成で確認済み。
- 次の未完了点は、ABCI上で公式サイズのモデルを構築して実checkpointをロードし、shape mismatchが残っていないかを確認すること。
