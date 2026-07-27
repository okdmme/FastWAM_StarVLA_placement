# Wan2 / FastWAM Reuse Plan

このメモは、`starVLA/starVLA/model/modules/world_model/Wan2.py` と、
FastWAM 由来で一時配置した `world_model` ファイル群のうち、
どこが既存 `Wan2` から再利用できるかを整理したものです。

対象:
- `starVLA/starVLA/model/modules/world_model/Wan2.py`
- `starVLA/starVLA/model/modules/world_model/FastWAM_WanVideoDiT.py`
- `starVLA/starVLA/model/modules/world_model/FastWAM_WanVideoTextEncoder.py`
- `starVLA/starVLA/model/modules/world_model/FastWAM_WanVideoVAE.py`
- `starVLA/starVLA/model/modules/world_model/FastWAM_scheduler_continuous.py`
- `starVLA/starVLA/model/modules/world_model/FastWAM_gradient.py`
- `starVLA/starVLA/model/modules/world_model/FastWAM_io.py`
- `starVLA/starVLA/model/modules/world_model/FastWAM_loader.py`
- `starVLA/starVLA/model/modules/world_model/FastWAM_state_dict_converters.py`

## 前提整理

`Wan2.py` はすでに以下を持っています。

- `UMT5EncoderModel` ベースの text encoder ロード
- `AutoencoderKLWan` ベースの VAE ロード
- `WanTransformer3DModel` ベースの DiT ロード
- `VideoProcessor` による前処理
- `build_inputs()`
- `forward()`
- `generate()`
- 中間層 hook による hidden states 抽出

つまり、FastWAM 側 `world_model` ファイルの多くは、既存 `Wan2.py` と責務が重なっています。

## Step 1. `Wan2.py` と `FastWAM_WanVideoDiT.py` の機能差分一覧

ここでは「何が違うか」だけを整理します。  
まだ「どれを採用するか」は決めません。

### A. 実装のレイヤ

`Wan2.py`
- diffusers の `WanTransformer3DModel` をそのまま利用する wrapper
- 目的は world-model backend の統一 interface を提供すること
- `build_inputs -> transformer forward -> hidden_states 抽出` に集中している

`FastWAM_WanVideoDiT.py`
- `WanVideoDiT` というローカル実装の DiT 本体
- patchify / unpatchify / RoPE / self-attn mask / action-conditioned context を自前で持つ
- 目的は video denoising を直接制御すること

差分の本質:
- `Wan2.py` は wrapper
- `FastWAM_WanVideoDiT.py` は backbone 実装

### B. 入力表現と前処理

`Wan2.py`
- 画像列を `VideoProcessor` と `AutoencoderKLWan` で latent 化
- `build_inputs()` が model input dict を返す
- timestep は token-wise zero tensor を組むが、その先の token 化は `WanTransformer3DModel` 側に委譲

`FastWAM_WanVideoDiT.py`
- `pre_dit()` 内で latent tensor を自前 patchify
- `patch_embedding` によって `[B, C, T, H, W] -> token sequence`
- `precompute_freqs_cis_3d()` を使って 3D RoPE を自前で適用

差分:
- `Wan2.py` は diffusers モデル内部へ処理を委譲
- `FastWAM_WanVideoDiT.py` は tokenization と positional encoding を外から制御

### C. timestep の扱い

`Wan2.py`
- `build_inputs()` で token-wise timestep tensor を作る
- feature extraction 時はゼロ timestep を使う
- scheduler は `UniPCMultistepScheduler`

`FastWAM_WanVideoDiT.py`
- `seperated_timestep` と `fuse_vae_embedding_in_latents` を前提に、
  各 token ごとに timestep embedding を生成
- 最初の latent frame だけ timestep を 0 に固定するロジックが `pre_dit()` に組み込まれている
- scheduler 本体は別ファイル `FastWAM_scheduler_continuous.py`

差分:
- `Wan2.py` は wrapper 側で timestep tensor を作る
- `FastWAM_WanVideoDiT.py` は backbone 側で token-wise modulation まで持つ

### D. attention mask 制御

`Wan2.py`
- `WanTransformer3DModel` の標準挙動を利用
- wrapper 側には video-video attention mask 制御 API がない

`FastWAM_WanVideoDiT.py`
- `build_video_to_video_mask()` を持つ
- `video_attention_mask_mode` を切り替え可能
  - `bidirectional`
  - `per_frame_causal`
  - `first_frame_causal`

差分:
- FastWAM 側には明示的な video self-attention 制御機構がある
- これは FastWAM の action-only 推論や first-frame 固定と強く関係する

### E. action-conditioned context

`Wan2.py`
- text instruction のみを `encoder_hidden_states` として使う
- action を world model context に混ぜる構造はない

`FastWAM_WanVideoDiT.py`
- `action_conditioned=True` のとき action を context token に変換して追加できる
- `action_group_causal_mask_mode` により、
  どの latent frame がどの action token 群を参照できるかを制御する

差分:
- これは FastWAM 独自の重要機能
- StarVLA の現行 `Wan2.py` には存在しない

### F. forward 出力の意味

`Wan2.py`
- world-model wrapper として hidden states を返す
- action head が使う中間表現抽出が主目的

`FastWAM_WanVideoDiT.py`
- 直接 video latent の denoising 出力を返す
- `post_dit()` で unpatchify まで行う

差分:
- `Wan2.py` は feature extraction 寄り
- `FastWAM_WanVideoDiT.py` は generative denoising 本体寄り

## Step 1 の結論

`Wan2.py` と `FastWAM_WanVideoDiT.py` は、
「同じ Wan2.2 を使っている」ものの、責務はかなり違います。

一言でいうと:

- `Wan2.py` = StarVLA 用の world-model interface wrapper
- `FastWAM_WanVideoDiT.py` = FastWAM 用の制御しやすい独自 DiT 実装

したがって、
`FastWAM_WanVideoDiT.py` をそのまま `Wan2.py` の代替として置き換えるのは不適切です。

## ここまでの再利用方針

### そのまま再利用すべきではないもの

- `FastWAM_WanVideoTextEncoder.py`
  - `Wan2.py` がすでに `UMT5EncoderModel` をロードしているため重複
- `FastWAM_WanVideoVAE.py`
  - `Wan2.py` がすでに `AutoencoderKLWan` をロードしているため重複
- `FastWAM_loader.py`
  - `Wan2.py` は diffusers/HF の標準ロード経路を使っており、独自 loader は不要
- `FastWAM_io.py`
- `FastWAM_state_dict_converters.py`
  - 独自重み形式を読む必要がある場合だけ必要

### 条件付きで検討するもの

- `FastWAM_WanVideoDiT.py`
  - 全移植ではなく、必要機能だけを抽出するかを判断する
- `FastWAM_scheduler_continuous.py`
  - flow-matching 学習/推論を world-model 側に導入するなら候補
- `FastWAM_gradient.py`
  - gradient checkpoint helper として一部再利用可能

## 次のステップ

## Step 2. `FastWAM_WanVideoDiT.py` から残す機能を選別する

このステップの目的は、
`FastWAM_WanVideoDiT.py` 全体を移すことではなく、
`Wan2.py` に不足している制御機能だけを抽出することです。

判断基準:
- StarVLA の既存 `Wan2.py` で再現できないか
- FastWAM の上位オーケストレーションなしでも意味を持つか
- `world_model` の責務として自然か
- `action_model` や上位 training loop に押し出した方がよくないか

### Step 2-A. 採用候補

#### 1. `video_attention_mask_mode`

元実装:
- `FastWAM_WanVideoDiT.py` の `video_attention_mask_mode`
- `build_video_to_video_mask()`

必要性:
- StarVLA の `Wan2.py` には video self-attention の振る舞いを明示的に切り替える API がない
- first-frame 固定や frame-causal な制約を試したい場合、この制御点は有用

推奨:
- 採用候補
- ただし `Wan2.py` 全体を書き換えるのではなく、
  `build_inputs()` か `forward()` に渡す設定として最小追加する

最低限必要なこと:
- `wm_cfg.video_attention_mask_mode` のような config 追加
- `Wan2.py` 側で `bidirectional / per_frame_causal / first_frame_causal` を解釈する補助関数追加
- diffusers の `WanTransformer3DModel` に attention mask を渡せるか事前確認

#### 1-A. 実現可能性の確認結果

確認ソース:
- [Wan2.py](/home/anpan/WM/starVLA/starVLA/model/modules/world_model/Wan2.py:84)
- Hugging Face Diffusers docs: `WanTransformer3DModel.forward`

確認結果:
- `Wan2.py` は `self.transformer(...)` に
  - `hidden_states`
  - `timestep`
  - `encoder_hidden_states`
  だけを渡している
- Diffusers の `WanTransformer3DModel.forward` 公開シグネチャには、
  `attention_mask` ではなく `attention_kwargs` はある
- つまり、FastWAM の `build_video_to_video_mask()` のような
  **video self-attention mask をそのまま wrapper から渡せることは確認できていない**

この時点の判断:
- `video_attention_mask_mode` は「すぐ移植」ではなく「要調査」
- まずは `attention_kwargs` で内部 attention processor まで mask を流せるか確認が必要
- それが無理なら、`WanTransformer3DModel` backend のままでは実装困難

必要な追加作業:
1. 実際の diffusers 実装で `attention_kwargs` が Wan の self-attention まで届くか確認
2. 届かない場合、`Wan2.py` wrapper ではなく backend 変更が必要と判断
3. backend 変更が必要なら、この項目は Step 3 ではなく保留に回す

#### 2. token-wise separated timestep modulation

元実装:
- `FastWAM_WanVideoDiT.py` の `seperated_timestep`
- `fuse_vae_embedding_in_latents`
- `pre_dit()` 内の frame 先頭 timestep=0 固定ロジック

必要性:
- `Wan2.py` も token-wise timestep tensor 自体は作っている
- ただし、FastWAM 側は「最初の latent frame だけ 0 にする」という制御を backbone の前処理ロジックに強く埋め込んでいる
- first-frame conditioning を明示化したいなら候補になる

推奨:
- 条件付き採用候補
- `Wan2.py` ですでに token-wise timestep を作っているため、
  まずは `build_inputs()` 側の timestep 構成だけで再現できるかを確認する

最低限必要なこと:
- `Wan2.py` の `build_inputs()` が作る timestep を、
  `all zero` から `first frame zero + others configurable` に拡張できるか確認
- backbone を差し替えずに済むなら、`FastWAM_WanVideoDiT.py` からはロジックだけ借りる

#### 2-A. 実現可能性の確認結果

確認結果:
- `Wan2.py` はすでに `build_inputs()` で token-wise timestep tensor を作っている
- したがって、FastWAM 側の `seperated_timestep` のうち
  「token ごとに timestep を与える」という発想自体はすでに近い
- 差分は、
  `FastWAM_WanVideoDiT.py` が `pre_dit()` の中で
  「先頭 frame の token timestep だけ 0 にする」ことを強く前提にしている点

この時点の判断:
- これは `Wan2.py` wrapper 側で最も取り込みやすい候補
- `build_inputs()` の timestep 構成ロジックだけを拡張すれば済む可能性が高い

必要な追加作業:
1. `Wan2.py:build_inputs()` で生成している `timestep` の shape と token 順序を固定的に整理
2. `first frame -> zero`, `other frames -> configurable` の規則に書き換える案を作る
3. その変更で hidden-state 抽出用途に副作用がないか確認する

#### 3. `pre_dit()` / `post_dit()` 的な外部制御性

元実装:
- `FastWAM_WanVideoDiT.py` の `pre_dit()`
- `FastWAM_WanVideoDiT.py` の `post_dit()`

必要性:
- FastWAM は `MoT` と接続するため、video token を中間表現として明示的に取り出す必要があった
- StarVLA 側でも、将来的に world-model token を action 側へ細かく渡したいなら、この分解は便利

推奨:
- いきなり全面採用はしない
- まずは `Wan2.py` の wrapper レベルで
  「どの hidden states を返すか」を明示化するだけで足りる可能性が高い

最低限必要なこと:
- `Wan2.py` で hidden states の抽出位置を config 化
- `pre_dit/post_dit` 相当の API を wrapper に設ける必要が本当にあるか確認

#### 3-A. 実現可能性の確認結果

確認結果:
- `Wan2.py` はすでに hook を使って複数層の hidden states を抽出できる
- `extract_layers` も config で切り替えられる
- したがって、FastWAM の `pre_dit/post_dit` をそのまま持ち込まなくても、
  StarVLA 側の「中間表現を action head に渡す」目的のかなりの部分は達成済み

この時点の判断:
- この項目は「追加 API」ではなく「既存 hook ベース抽出で十分かを確認する」方向が先
- Step 2 の時点では優先度は低い

### Step 2-B. 現時点では採用しない候補

#### 4. `action_conditioned` context

元実装:
- `FastWAM_WanVideoDiT.py` の `action_conditioned`
- `action_group_causal_mask_mode`

見送り理由:
- これは FastWAM の video/action joint 学習設計に強く依存する
- StarVLA の現行 `Wan2.py` は world-model backend であり、
  action を直接 world-model context に混ぜる責務はまだ持っていない
- 先に `Wan2.py` を world-model として安定させるべき

結論:
- Step 2 では採用しない
- 必要になったら Step 3 以降で別機能として検討する

#### 5. `patchify()` / `unpatchify()` を含む独自 backbone 全体

見送り理由:
- これは `WanTransformer3DModel` を使う既存 `Wan2.py` を、独自実装 backbone に置き換える方向になる
- 変更量が大きく、Step 2 の目的を超える

結論:
- Step 2 では採用しない

### Step 2 の実施順

1. `Wan2.py` に追加したい制御点を 3 つに絞る
   - `video_attention_mask_mode`
   - timestep 構成の拡張
   - hidden state 抽出位置の明示化
2. その 3 つが diffusers `WanTransformer3DModel` の wrapper のまま実現可能か確認する
3. wrapper のまま実現できるものだけ `Wan2.py` に寄せる
4. wrapper のまま無理なものは「backbone 差し替えが必要」と明示して保留する

### Step 2 のゴール

この時点で目指す状態は以下です。

- `Wan2.py` を残したまま FastWAM 由来の有用な制御機能を一部吸収する
- `FastWAM_WanVideoDiT.py` を参照元として残すかどうか判断できる
- world-model 側の重複ファイルをこれ以上増やさない

## Step 2 の中間結論

現時点での優先順位はこうなる。

1. 最優先で検討する
   - token-wise timestep 構成の拡張

2. 調査が必要
   - `video_attention_mask_mode`

3. 今は後回しでよい
   - `pre_dit/post_dit` 的な外部制御 API

つまり、次にやるべき具体作業は
`Wan2.py` の `build_inputs()` における timestep 構成を、
FastWAM 的 first-frame 固定に寄せられるかを設計することです。

### Step 3 でやること

Step 2 で残した機能だけを、
`Wan2.py` にどう移植するかの具体案に落とす。

### Step 4 でやること

不要と判断した FastWAM 由来 `world_model` ファイルを整理し、
残すファイルを最小化する。
