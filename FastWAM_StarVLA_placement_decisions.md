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
| `mot.py` | `starVLA/model/framework/WM4A/FastWAM_MoT.py` | 公式実装を WM4A framework 配下に配置 | MoT は video expert と action expert を層ごとに結合する orchestration。単なる world/action module ではなく framework-level の混合制御なので `framework/WM4A` が自然。 |
| `fastwam.py` | `starVLA/model/framework/WM4A/FastWAM.py` | StarVLA framework entry として新規作成 | StarVLA の registry、training loop、config に接続する入口。公式 `FastWAM` の責務を StarVLA の `baseframework` interface に合わせる場所。 |
| `scheduler_continuous.py` | `starVLA/model/framework/WM4A/FastWAM.py` | 小さい scheduler class を吸収 | video/action training loss に密接で、単独 module にする必要が薄い。新規ファイル増加を避けつつ公式ロジックを保持できる。 |
| `wan_video_vae.py`, `wan_video_text_encoder.py` | 未配置。まず既存 `Wan2.py`/diffusers 経路を再利用予定 | 条件付き再利用 | StarVLA 既存 `Wan2.py` は `AutoencoderKLWan` と `UMT5EncoderModel` のロード経路を持つため、公式 FastWAM loader をそのまま増やす前に再利用可否を確認する。 |
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

## Current Progress

- `FastWAM_ActionDiT.py`: 配置済み。
- `FastWAM_WanVideoDiT.py`: 配置済み。StarVLA logger/import/checkpoint helper に合わせた最小変更のみ。
- `FastWAM_MoT.py`: 配置済み。StarVLA の module import に合わせた最小変更のみ。
- `FastWAM.py`: registry entry、expert 構築、MoT 接続、scheduler 吸収まで完了。
- `tests/test_fastwam_smoke.py`: 追加済み。tiny config で scheduler、framework build、video/action expert と MoT の token-level forward を確認する。

## Smoke Test Scope

この smoke test は、実データや Wan2.2 本番重みを使わず、以下の破損を早く検出する目的で置く。

- `FastWAM` が StarVLA の `build_framework()` 経由で構築できるか。
- `WanContinuousFlowMatchScheduler` の timestep/noise/weight shape が training loss 用に使えるか。
- `WanVideoDiT.pre_dit()` と `ActionDiT.pre_dit()` が MoT へ渡す token/freq/context/timestep modulation の shape 契約を満たすか。
- MoT が video expert と action expert を同じ layer interface で扱えるか。

この段階では VAE/text encoder、dataset adapter、`training_loss()` 本体はまだ確認対象外。

## Next Work

1. `FastWAM.training_loss()` を `FastWAMFramework.forward()` に移植する。
2. StarVLA batch から FastWAM が必要とする `video/action/proprio/prompt` sample 形式への adapter を決める。
3. VAE/text encoder はまず StarVLA 既存 `Wan2.py`/diffusers 経路を再利用できるか確認する。
4. 再利用できない場合だけ、公式 FastWAM の loader/helper を既存 module 配下へ最小配置する。
