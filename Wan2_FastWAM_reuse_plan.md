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

## Step 2-1. `Wan2.py` の timestep 構成を FastWAM 的 first-frame 固定に寄せる設計

確認ソース:
- [Wan2.py:256](/home/anpan/WM/starVLA/starVLA/model/modules/world_model/Wan2.py:256)
- [Wan2.py:282](/home/anpan/WM/starVLA/starVLA/model/modules/world_model/Wan2.py:282)
- [FastWAM_WanVideoDiT.py:509](/home/anpan/WM/starVLA/starVLA/model/modules/world_model/FastWAM_WanVideoDiT.py:509)

### 現状の `Wan2.py`

`Wan2.py` は `build_inputs()` で以下を行っている。

1. `latents` の shape から `T, H, W` を得る
2. `patch_size = (p_t, p_h, p_w)` を読む
3. `seq_len = (T // p_t) * (H // p_h) * (W // p_w)` を計算する
4. `timestep = zeros([B, seq_len])` を作る

つまり現状は、全 token が同じ timestep 0 で固定されている。

### FastWAM 側の前提

`FastWAM_WanVideoDiT.py` は `pre_dit()` で:

1. `tokens_per_frame = (H // p_h) * (W // p_w)` を計算する
2. `token_timesteps` を `[B, num_latent_frames, tokens_per_frame]` で構成する
3. `token_timesteps[:, 0, :] = 0` として first frame を clean に固定する
4. 最後に `reshape(B, -1)` して token sequence に揃える

ここで重要なのは token 順序で、FastWAM 側は

```text
(frame, h, w) -> flatten
```

すなわち frame-major の順で token を並べている。

### `Wan2.py` でも同じ token 順序を仮定できる理由

`Wan2.py` の `seq_len` 計算は

```text
(T // p_t) * (H // p_h) * (W // p_w)
```

となっており、Wan の patch 化後トークン列も
時間軸を先頭に持つ frame-major flatten を前提にしているとみなすのが自然。

この前提のもとでは、各 frame に属する token 数は

```text
tokens_per_frame = (H // p_h) * (W // p_w)
num_temporal_groups = T // p_t
```

で定まる。

### 変更案

`Wan2.py` の `build_inputs()` で、現在の

```python
timestep = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)
```

を、次のような構成に変える。

```text
base_timestep: [B] または scalar 的な制御値
token_timesteps: [B, num_temporal_groups, tokens_per_frame]
token_timesteps[:, 0, :] = 0
token_timesteps[:, 1:, :] = base_value
flatten -> [B, seq_len]
```

### config として追加したい項目

`wm_cfg` に次のような設定を追加する案が妥当。

- `token_timestep_mode`
  - `all_zero`
  - `first_frame_zero`
- `non_first_frame_timestep`
  - 既定値は `0`
  - 将来的に非ゼロも試せるようにする

これにより:
- 現行互換は `all_zero`
- FastWAM 寄り挙動は `first_frame_zero`

と切り替えられる。

### この変更の利点

- `WanTransformer3DModel` 自体は差し替えない
- `Wan2.py` の wrapper だけで試せる
- `FastWAM_WanVideoDiT.py` の全移植より影響範囲が小さい

### 先に確認すべきこと

この設計で実装する前に、次の確認が必要。

1. `timestep` の dtype と期待レンジ
   - 現在は `torch.long`
   - 非ゼロ値を入れる場合、diffusers 側で問題なく受けられるか

2. `T // p_t` が latent frame 数として常に妥当か
   - Wan2.2 の patch size が `(1, 2, 2)` 前提なら問題は小さい
   - ただし config 依存なので実装では一般式にする

3. hidden-state 抽出用途への影響
   - `all_zero` 前提で取っていた中間表現が変質しないか
   - まずは `non_first_frame_timestep = 0` のままで API だけ追加するのが安全

### Step 2-1 の実装方針

最初の実装は最小限にする。

1. `wm_cfg.token_timestep_mode` を読む
2. デフォルトは現行互換 `all_zero`
3. `first_frame_zero` のときだけ `token_timesteps` を frame 単位で組み立てる
4. `non_first_frame_timestep` の既定値は `0` にして、挙動差を最初は出さない

この形なら、まずは API と token grouping だけを安全に入れられる。

### Step 2-1 の実装結果

実装ファイル:
- [Wan2.py](/home/anpan/WM/starVLA/starVLA/model/modules/world_model/Wan2.py:62)
- [Wan2.py](/home/anpan/WM/starVLA/starVLA/model/modules/world_model/Wan2.py:260)

追加した設定:
- `wm_cfg.token_timestep_mode`
  - 既定値: `all_zero`
  - 対応値: `all_zero`, `first_frame_zero`
- `wm_cfg.non_first_frame_timestep`
  - 既定値: `0`

追加した実装:
- `_Wan2_Interface.__init__()` で timestep mode を読み込む
- `_build_token_timesteps()` を追加
- `build_inputs()` の timestep 生成を `_build_token_timesteps()` に切り出す

現行互換:
- config を追加しない場合は `all_zero` なので、既存と同じく全 token timestep は `0`

FastWAM 寄り挙動:
- `token_timestep_mode = first_frame_zero` のとき、
  `[B, temporal_groups, tokens_per_frame]` で timestep を組み、
  最初の temporal group の token を `0` に固定する
- それ以外の temporal group は `non_first_frame_timestep` で埋める

確認済み:
- `python3 -m py_compile starVLA/model/modules/world_model/Wan2.py` は成功

残る確認:
- 実際の `diffusers.WanTransformer3DModel` 実行環境で、
  `non_first_frame_timestep != 0` が期待どおり受けられるか
- 現在の環境には `diffusers` が入っていないため、実 forward の確認は未実施

## Step 2-2. `video_attention_mask_mode` の実現可能性確認

確認対象:
- `FastWAM_WanVideoDiT.py` の `build_video_to_video_mask()`
- diffusers `WanTransformer3DModel`
- 既存 `Wan2.py` wrapper

### 確認結果

diffusers の `WanTransformer3DModel.forward()` は公開 API として
`attention_kwargs` を受け取る。

ただし、公式実装を見る限り、Wan の transformer block 内では
self-attention 呼び出し時に attention mask が渡されていない。

具体的には、`WanTransformerBlock.forward()` では self-attention が概ね次の形で呼ばれている。

```python
attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
```

ここで第3引数にあたる attention mask は `None` で固定されている。

また、`WanTransformer3DModel.forward()` 側も各 block を

```python
hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
```

の形で呼んでおり、video self-attention mask を block に渡す経路がない。

### 判断

`Wan2.py` の wrapper だけで `video_attention_mask_mode` を実現するのは難しい。

理由:
- `Wan2.py` から `attention_kwargs` を渡しても、Wan block の self-attention mask 引数に接続されていない
- `FastWAM_WanVideoDiT.py` のような `build_video_to_video_mask()` を作っても、diffusers backend に自然に差し込む場所がない
- 実装するには、diffusers の `WanTransformerBlock.forward()` 相当を変更するか、独自 backend に差し替える必要がある

### 結論

`video_attention_mask_mode` は Step 2 では採用しない。

理由:
- 既存 `Wan2.py` を使う方針から外れ、diffusers backend の内部改変に近くなる
- 「StarVLA の既存 Wan2 を用いる」という説明が弱くなる
- FastWAM の `MoT` / action-only 推論設計に近い機能であり、現段階の world-model wrapper 改修としては大きすぎる

### 今後必要になった場合の選択肢

1. diffusers `WanTransformer3DModel` を subclass / wrapper 化して block forward に mask を通す
2. `FastWAM_WanVideoDiT.py` の独自 backbone を正式採用する
3. `video_attention_mask_mode` は world model ではなく、将来の FastWAM 統合層で扱う

現時点の推奨は 3。

つまり、今は `Wan2.py` を維持し、
`video_attention_mask_mode` は保留扱いにする。

## Step 2-3. `FastWAM_ActionDiT.py` の依存整理

対象:
- `starVLA/starVLA/model/modules/action_model/FastWAM_ActionDiT.py`
- `starVLA/starVLA/model/modules/world_model/FastWAM_WanVideoDiT.py`
- `starVLA/starVLA/model/modules/world_model/FastWAM_gradient.py`
- `starVLA/starVLA/model/modules/world_model/FastWAM_scheduler_continuous.py`

### 確認結果

`FastWAM_ActionDiT.py` は、もともと `FastWAM_WanVideoDiT.py` から以下を import していた。

- `DiTBlock`
- `sinusoidal_embedding_1d`
- `precompute_freqs_cis`

また、`FastWAM_gradient.py` から以下を import していた。

- `gradient_checkpoint_forward`

StarVLA 既存の `action_model/DiT_modules/models.py` にも `DiTBlock` はあるが、
構造が異なる。

FastWAM の `ActionDiT` が必要とする block は:
- RoPE self-attention
- text context cross-attention
- timestep modulation
- gate 付き residual

一方、StarVLA 既存 `DiT_modules.models.DiTBlock` は通常の DiT block であり、
FastWAM `ActionDiT` の block と互換ではない。

### 判断

`FastWAM_ActionDiT.py` は action model として残す価値がある。

ただし、`world_model/FastWAM_WanVideoDiT.py` に依存させるのは不自然。

理由:
- `FastWAM_WanVideoDiT.py` は world model 側の独自 video backbone
- `ActionDiT` が必要としているのは、その中の小さな transformer 部品だけ
- world model 側に独自 backbone 全体を残すと、既存 `Wan2.py` 再利用方針と矛盾する

### 実施内容

`FastWAM_ActionDiT.py` を自己完結させた。

具体的には、以下を `FastWAM_ActionDiT.py` 内に移した。

- `gradient_checkpoint_forward`
- `flash_attention`
- `modulate`
- `sinusoidal_embedding_1d`
- `precompute_freqs_cis`
- `rope_apply`
- `RMSNorm`
- `SelfAttention`
- `CrossAttention`
- `DiTBlock`

また、FastWAM 固有 logger 依存を削除し、
StarVLA 既存の `initialize_overwatch` に置き換えた。

### 削除した未追跡ファイル

以下はコード上の参照がなくなったため削除した。

- `starVLA/starVLA/model/modules/world_model/FastWAM_WanVideoDiT.py`
- `starVLA/starVLA/model/modules/world_model/FastWAM_gradient.py`
- `starVLA/starVLA/model/modules/world_model/FastWAM_scheduler_continuous.py`

### 現在残すファイル

- `starVLA/starVLA/model/modules/action_model/FastWAM_ActionDiT.py`

このファイルは、FastWAM 側の action expert 候補として `action_model` 配下に置く。

### 追加した StarVLA 入口

`FastWAM_ActionDiT.py` に `FastWAMActionDiTHead` と `get_action_model(config=None)` を追加した。

目的:
- StarVLA 既存 action head と同じ factory 形式に寄せる
- `config.framework.action_model` から StarVLA 互換 action head を構築できるようにする
- 外側からは `forward(vl_embs, actions, state)` と `predict_action(vl_embs, state)` を呼べるようにする
- pretrained payload が指定された場合だけ `ActionDiT.from_pretrained()` を使う
- pretrained 指定がない場合は通常の `ActionDiT(**config)` で構築し、呼び出し側の device/dtype 管理を邪魔しない

config の読み方:
- `action_dit_config` があれば、その mapping を優先する
- なければ `action_hidden_dim`, `action_dim`, `ffn_dim`, `text_dim`, `freq_dim`, `num_heads`, `attn_head_dim`, `num_layers`, `eps`, `use_gradient_checkpointing` から組み立てる

`FastWAMActionDiTHead` の役割:
- StarVLA framework が渡す world-model feature `vl_embs` を `ActionDiT` の `context` として使う
- action chunk に flow-matching 風の noise/time を加えて MSE loss を返す
- inference では Euler update で action chunk を生成する
- `state_dim` が設定されている場合は state を 1 token として context 末尾に追加する

### 確認済み

- `python3 -m py_compile starVLA/model/modules/action_model/FastWAM_ActionDiT.py`
- `python3 -m py_compile starVLA/model/modules/world_model/Wan2.py`

どちらも成功。

未確認:
- この作業環境には `torch` import 可能な実行環境が無いため、dummy forward/predict の runtime 検証は未実施
- PyTorch が入った StarVLA 実行環境で shape test が必要

### 次の作業

`FastWAM_ActionDiT.py` を StarVLA の action model routing からどう呼ぶかを検討する。

候補:
- 既存 `action_model/__init__.py` または model builder がある場合、そこに `FastWAM_ActionDiT` を登録する
- framework config 側に `action_model_type` または `action_model_name` で選択できる設定を追加する
- routing 追加前に、既存の action model import 経路を確認する

最小方針は、既存の action model 選択機構を壊さず、
`FastWAM_ActionDiT.get_action_model(config)` を選べる分岐だけを追加すること。

ただし現在の StarVLA は中央 action-model registry を持たず、
各 framework が個別に action head を import している。
そのため、次に変更すべき対象は以下のどちらかを選ぶ必要がある。

- `WanGR00T.py` を FastWAM action head に切り替えられるようにする
- `WanFastWAM.py` のような専用 framework を追加し、既存 `WanGR00T/WanPI` は触らない

### 実施した routing 方針

専用 framework 方針を採用した。

追加ファイル:
- `starVLA/starVLA/model/framework/WM4A/WanFastWAM.py`

登録名:
- `WanFastWAM`

理由:
- 既存 `WanGR00T.py` / `WanPI.py` の挙動を変えない
- FastWAM action head の検証を独立した framework 名で行える
- 問題が出た場合も `framework.name: WanFastWAM` を使っている設定だけに影響を限定できる

構成:
- world model は既存 `get_world_model(config)` を利用
- Wan hidden state は `wm_projector` で `ActionDiT` の `text_dim` に射影
- action head は `FastWAM_ActionDiT.get_action_model(config)` から取得
- training は `forward(vl_embs, actions, state)` 互換 wrapper 経由で action loss を返す
- inference は `predict_action(vl_embs, state)` 互換 wrapper 経由で normalized action を返す

### ブランチ方針

作成したブランチ:
- `develop`: `upstream/starVLA_dev` と同じ位置。公式実装に何も足していない基準ブランチ
- `fastwam-starvla-integration`: FastWAM 統合作業用ブランチ

remote:
- `origin`: `https://github.com/okdmme/FastWAM-and-StarVLA.git`
- `upstream`: `https://github.com/starVLA/starVLA.git`
- `upstream` push URL は `DISABLED`

この時点では push は行っていない。

### 実行環境で確認するコマンド

この作業環境では当初 `torch` import ができなかったため、Python 3.11 の検証用 venv を作って確認した。

環境確認結果:
- `nvidia-smi`: command not found
- `lspci`: command not found
- `nvcc`: command not found
- `.venv-py311` の `torch.cuda.is_available()`: `False`
- `.venv-py311` の `torch.__version__`: `2.6.0+cpu`

この環境からは NVIDIA GPU / CUDA driver は見えていない。

実行した検証:

```bash
cd /home/anpan/WM/starVLA
python3 - <<'PY'
from types import SimpleNamespace
import torch
from starVLA.model.modules.action_model.FastWAM_ActionDiT import get_action_model

cfg = SimpleNamespace(framework=SimpleNamespace(action_model=SimpleNamespace(
    action_dit_config={
        "action_dim": 3,
        "hidden_dim": 16,
        "ffn_dim": 32,
        "num_heads": 2,
        "attn_head_dim": 4,
        "num_layers": 1,
        "text_dim": 8,
        "freq_dim": 8,
        "eps": 1e-6,
        "use_gradient_checkpointing": False,
    },
    action_horizon=4,
    num_inference_timesteps=2,
    num_timestep_buckets=10,
    noise_beta_alpha=1.5,
    noise_beta_beta=1.0,
    noise_s=0.999,
)))
model = get_action_model(cfg)
vl = torch.randn(2, 5, 8)
actions = torch.randn(2, 4, 3)
loss = model(vl, actions)
pred = model.predict_action(vl)
print(type(model).__name__, tuple(loss.shape), tuple(pred.shape))
PY
```

確認結果:
- `FastWAMActionDiTHead () (2, 4, 3) True`
- loss は scalar tensor なので shape は `()`
- predict output は `[2, 4, 3]`

補足:
- 最初の dummy test では wrapper が `ActionDiT` に discrete long timestep を渡しており dtype mismatch になった
- `ActionDiT` は continuous float timestep を前提にしているため、wrapper 側を float timestep 渡しに修正した

### Step 3 でやること

Step 2 で残した機能だけを、
`Wan2.py` にどう移植するかの具体案に落とす。

### Step 4 でやること

不要と判断した FastWAM 由来 `world_model` ファイルを整理し、
残すファイルを最小化する。

## 方針修正: フル FastWAM を StarVLA 既存 module に載せる

ここまでの作業は、StarVLA 既存 `Wan2.py` を維持したうえで
FastWAM ActionDiT を action head として試す「最小統合」に寄っていた。

しかし最終目標は次の通り。

- FastWAM を StarVLA 上の既存 module 構造にできるだけ吸収する
- 新しい folder/file は最小限にする
- FastWAM 論文相当の video/action joint model をフルスクラッチ学習できるようにする
- 単なる `Wan2 hidden states -> ActionDiT` ではなく、FastWAM の `MoT` 経由 training を再現する

したがって、今後の優先順位は修正する。

### 現状の到達点

現在 StarVLA に入っている FastWAM 関連実装:

- `starVLA/model/modules/action_model/FastWAM_ActionDiT.py`
  - FastWAM の `action_dit.py` を StarVLA action model 配下に移したもの
  - `ActionDiT`, `FastWAMActionDiTHead`, `get_action_model()` を持つ
  - StarVLA 既存 action head と同じ factory 形式では使える
  - ただし FastWAM 本来の `MoT` mixed-attention training 経路にはまだ接続されていない

- `starVLA/model/framework/WM4A/WanFastWAM.py`
  - `framework.name: WanFastWAM` として registry には乗っている
  - 現状は `Wan2.py` の hidden states を `FastWAM_ActionDiT` に渡す薄い wrapper
  - FastWAM 本体の `training_loss()` / `MoT` / video loss はまだ入っていない

- `starVLA/model/modules/world_model/Wan2.py`
  - FastWAM 由来の token-wise timestep 構成を一部追加済み
  - ただし diffusers `WanTransformer3DModel` wrapper のまま
  - `pre_dit()`, `post_dit()`, `build_video_to_video_mask()`, action-conditioned context は未実装

### FastWAM ファイルごとの StarVLA 対応先

#### `action_dit.py`

対応先:
- `starVLA/model/modules/action_model/FastWAM_ActionDiT.py`

判断:
- ここに置く方針は正しい
- StarVLA 既存 `DiT_modules/models.py` の `DiTBlock` とは構造が違うため、既存 DiT に置き換えない
- 今後は wrapper 用の `FastWAMActionDiTHead` だけでなく、MoT から直接 `ActionDiT.pre_dit/post_dit` を使う経路を残す必要がある

#### `wan_video_dit.py`

対応先:
- 第一候補: `starVLA/model/modules/world_model/Wan2.py`

判断:
- FastWAM の video expert は world model 配下の責務なので、対応先は `Wan2.py`
- ただし現在の `Wan2.py` は diffusers transformer を一括 forward する wrapper
- FastWAM の `WanVideoDiT` は block ごとの `pre_dit/post_dit` と q/k/v mixed-attention を前提にする
- そのため、`Wan2.py` へ「設定だけ追加」では MoT 連携を再現できない

必要な未吸収機能:
- patchify / unpatchify
- `pre_dit()` / `post_dit()`
- `build_video_to_video_mask()`
- token-wise timestep modulation
- action-conditioned context
- RoPE helper / DiTBlock helper
- FastWAM 形式の denoising output

配置方針:
- 新しい subfolder は作らない
- まず `Wan2.py` に FastWAM backend mode を追加できるか検討する
- もし `Wan2.py` が大きくなりすぎる場合でも、追加ファイルは `world_model` 直下の最小数に留める

#### `mot.py`

対応先:
- 第一候補: `starVLA/model/framework/WM4A/WanFastWAM.py`

判断:
- MoT は単独 world model ではなく、video expert と action expert を layer-wise に混ぜる orchestration
- StarVLA の既存構造では framework が action/world model を組み合わせる責務を持つ
- そのため、まずは `WanFastWAM.py` に吸収するのが最小変更

注意:
- `MoT` は `wan_video_dit.py` の helper (`flash_attention`, `modulate`, `rope_apply`) に依存する
- これらはすでに `FastWAM_ActionDiT.py` にも重複して移されている
- 重複を避けるなら、最終的には StarVLA 既存 module 内で共有位置を考える必要がある
- ただし新規フォルダを増やさない方針なので、まずは既存ファイル内への吸収を優先する

#### `fastwam.py`

対応先:
- `starVLA/model/framework/WM4A/WanFastWAM.py`

判断:
- `FastWAM.training_loss()` は StarVLA では framework の `forward()` / `compute_loss()` に対応する
- 現在の `WanFastWAM.forward()` は action loss のみ
- フル FastWAM では video loss と action loss の両方を返す必要がある

必要な未吸収機能:
- `build_inputs(sample)`
- VAE encode/decode helper
- text/context handling
- proprio context append
- video/action scheduler
- `training_loss()`
- `_predict_joint_noise()`
- `infer_joint()`
- `infer_action()`

#### `fastwam_joint.py`

対応先:
- `starVLA/model/framework/WM4A/WanFastWAM.py`

判断:
- `FastWAMJoint` は attention mask policy の variant
- まず `WanFastWAM.py` の config option として吸収するのがよい
- 追加 framework ファイルを作るのは後回し

#### `fastwam_idm.py`

対応先:
- `starVLA/model/framework/WM4A/WanFastWAM.py`

判断:
- IDM は training objective / attention mask variant
- 最初から別ファイル化せず、`WanFastWAM.py` 内の mode として扱えるか検討する

#### `scheduler_continuous.py`

対応先:
- 第一候補: `starVLA/model/framework/WM4A/WanFastWAM.py`
- 第二候補: `starVLA/model/modules/world_model/Wan2.py`

判断:
- FastWAM では video/action 両方で同じ scheduler class を使う
- StarVLA 既存 action heads に flow matching scheduler 類はあるが、FastWAM の `WanContinuousFlowMatchScheduler` と同一ではない
- まずは `WanFastWAM.py` に吸収し、重複が明らかになったら既存 action_model 側へ寄せる

#### `wan_video_vae.py`

対応先:
- `starVLA/model/modules/world_model/Wan2.py`

判断:
- StarVLA の `Wan2.py` は diffusers `AutoencoderKLWan` を使っている
- FastWAM の `WanVideoVAE38` は独自 API (`encode(video, device, tiled, ...)`) を持つ
- フルスクラッチ学習で diffusers VAE をそのまま使えるなら新規移植は不要
- ただし `FastWAM.training_loss()` の VAE API と shape/normalization が合うように adapter が必要

#### `wan_video_text_encoder.py`

対応先:
- `starVLA/model/modules/world_model/Wan2.py`

判断:
- StarVLA `Wan2.py` は `UMT5EncoderModel` + `T5TokenizerFast` を既に使う
- FastWAM 独自 text encoder/tokenizer を丸ごと追加する優先度は低い
- ただし FastWAM の `context/context_mask` 前提と StarVLA dataloader の instruction encoding を揃える必要がある

#### `helpers/loader.py`, `helpers/io.py`, `helpers/state_dict_converters.py`

対応先:
- `starVLA/model/modules/world_model/Wan2.py`

判断:
- diffusers/HF 形式の pretrained を使うだけなら既存 `Wan2.py` の loader で代替可能
- FastWAM 独自 checkpoint または converted Wan weights を読むなら必要部分だけ `Wan2.py` に吸収する
- フルスクラッチ学習開始だけなら優先度は中程度

#### `helpers/gradient.py`

対応先:
- 既に `FastWAM_ActionDiT.py` に一部吸収済み
- `Wan2.py` / `WanFastWAM.py` 側にも必要なら同様に局所吸収

### StarVLA 既存ファイルでそのまま使えるもの

- framework registry / build:
  - `starVLA/model/framework/base_framework.py`
  - `framework.name: WanFastWAM` は registry に乗る

- world model entry:
  - `starVLA/model/modules/world_model/__init__.py`
  - 既存 `get_world_model(config)` は使えるが、FastWAM 本体では `Wan2.py` の通常 forward だけでは不足

- Wan2 VAE/text loading:
  - `starVLA/model/modules/world_model/Wan2.py`
  - VAE/text encoder/tokenizer は原則再利用候補

- action model location:
  - `starVLA/model/modules/action_model/FastWAM_ActionDiT.py`
  - 既存 action_model 配下に置く判断は正しい

- trainer loss routing:
  - `baseframework.compute_loss()` は `forward()` が返す tensor dict を扱える
  - `WanFastWAM.forward()` が `video_loss`, `action_loss`, `loss` などを返す形にすれば既存 trainer に乗せやすい

### StarVLA 既存ファイルだけでは不足しているもの

- MoT mixed attention
- video expert の `pre_dit/post_dit`
- video denoising loss
- action/video joint noise schedule
- FastWAM 用 sample builder
  - `video [B,3,T,H,W]`
  - `action [B,T,A]`
  - `context/context_mask`
  - `action_is_pad`
  - `image_is_pad`
- FastWAM inference
  - `infer_joint`
  - `infer_action`
  - video KV cache action-only inference

### 修正後の優先順位

1. `Wan2.py` に FastWAM video expert 相当を吸収できるかを設計する
   - `pre_dit/post_dit`
   - `build_video_to_video_mask`
   - FastWAM denoising output
   - diffusers `WanTransformer3DModel` を使い続けられるか、独自 block 実装が必要かを最終判断する

2. `WanFastWAM.py` を現在の薄い wrapper から FastWAM 本体へ近づける
   - `FastWAM.training_loss()` を StarVLA `forward()` に移す
   - `MoT` をまず `WanFastWAM.py` 内へ吸収する方針で検討する
   - video/action loss を返す

3. dataloader bridge を確認する
   - StarVLA の LeRobot batch が FastWAM `sample` 形式に変換できるか確認する
   - 既存 dataloader に手を入れるか、`WanFastWAM.forward()` 内で変換するか判断する

4. `fastwam_joint.py` / `fastwam_idm.py` は variant として後から吸収する
   - 最初から新規 framework file を増やさない
   - config mode として扱えるかを優先する

5. loader / state_dict converter は最後に判断する
   - フルスクラッチ学習に不要なら移植しない
   - pretrained initialization が必要になったら `Wan2.py` の loader 経路へ必要部分だけ吸収する

### 現時点の重要な判断

`Wan2.py` の diffusers wrapper だけでは FastWAM の MoT training を再現しにくい。

理由:
- MoT は各 layer で video/action expert の q/k/v を取り出して mixed attention する
- diffusers `WanTransformer3DModel` はその内部 q/k/v を StarVLA wrapper から自然には取り出せない
- FastWAM の `WanVideoDiT` はそのために `pre_dit/post_dit` と block-level API を持っている

したがって次の設計判断が必要。

1. `Wan2.py` に FastWAM 独自 block-level backend を吸収する
2. diffusers `WanTransformer3DModel` を改造/subclass 化して MoT に必要な q/k/v 経路を開く

新規フォルダを増やさない方針では、まず 1 を `Wan2.py` 内でどこまで可能か確認する。

## Step 5. `Wan2.py` への FastWAM video expert 吸収可否

方針:
- まず StarVLA 既存 file へ吸収できるかを判断する
- ただし役割を満たせない場合は、StarVLA の適切な既存 module 配下に FastWAM 公式ファイル相当を置く
- 新しい folder は作らない

### 確認対象

- StarVLA:
  - `starVLA/model/modules/world_model/Wan2.py`
  - diffusers `WanTransformer3DModel`
  - diffusers `WanTransformerBlock`

- FastWAM:
  - `src/fastwam/models/wan22/wan_video_dit.py`
  - `src/fastwam/models/wan22/mot.py`

### diffusers Wan の確認結果

実行環境の diffusers:

```text
diffusers 0.39.0
WanTransformer3DModel.forward(
    hidden_states,
    timestep,
    encoder_hidden_states,
    encoder_hidden_states_image=None,
    return_dict=True,
    attention_kwargs=None,
)
WanTransformerBlock.forward(
    hidden_states,
    encoder_hidden_states,
    temb,
    rotary_emb,
)
```

`WanTransformer3DModel.forward()` は内部で以下を行う。

1. `hidden_states` を `patch_embedding`
2. timestep / text embedding を作る
3. `for block in self.blocks: hidden_states = block(...)`
4. `norm_out + proj_out`
5. unpatchify して denoising output を返す

`WanTransformerBlock.forward()` は self-attention を次のように内部完結させる。

```python
norm_hidden_states = ...
attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
hidden_states = hidden_states + attn_output * gate_msa
```

ここでは:
- self-attention mask は常に `None`
- q/k/v は block 内部で作られる
- MoT 側が q/k/v を concat して mixed attention する入口がない
- block の post-attention/cross-attention/FFN 部分だけを再利用する公開 API もない

`WanAttention.forward()` 自体は `attention_mask` を受け取るが、
`WanTransformerBlock.forward()` が self-attention 呼び出し時に `None` を渡しているため、
`Wan2.py` wrapper から video self-attention mask を自然に流せない。

### FastWAM MoT が必要とする API

FastWAM の `MoT` は各 layer で以下を行う。

1. video expert block から q/k/v を作る
2. action expert block から q/k/v を作る
3. video/action q/k/v を concat
4. mixed attention mask を使って attention
5. attention output を expert ごとに split
6. 各 expert の cross-attention / FFN / gate residual を適用

このため、video expert は単なる `forward()` では足りず、最低限以下が必要。

- `pre_dit()`
- `post_dit()`
- block ごとの q/k/v 構築に必要な module 構造
- `build_video_to_video_mask()`
- token-wise timestep modulation
- action-conditioned context mask

### 判断

`Wan2.py` の既存 diffusers backend に FastWAM video expert を「吸収」するのは不適切。

理由:
- MoT の中核である q/k/v mixed-attention を diffusers `WanTransformer3DModel` の public wrapper から実現できない
- `WanTransformerBlock.forward()` を大きく改造する必要がある
- その改造は StarVLA 既存 `Wan2.py` の world-model feature-extraction wrapper と責務が大きく違う
- `Wan2.py` に無理に埋め込むと、既存 `WanGR00T/WanPI/WanOFT` の backend と FastWAM 独自 backend が混在して保守しにくい

したがって、ここは吸収ではなく、FastWAM 公式 `wan_video_dit.py` 相当を
StarVLA 既存 module 配下に置くべき。

### 配置判断

新しい folder は作らず、既存 `world_model` module 直下に置く。

推奨配置:

- `starVLA/model/modules/world_model/FastWAM_WanVideoDiT.py`

理由:
- 役割は明確に world model の video expert
- `Wan2.py` は StarVLA 既存 diffusers Wan wrapper として残せる
- FastWAM の `pre_dit/post_dit/build_video_to_video_mask` を公式実装に近い形で維持できる
- MoT から video expert として直接使える
- 新規 folder は増えない

### 関連 helper の扱い

`FastWAM_WanVideoDiT.py` は次を持つ。

- `flash_attention`
- `modulate`
- `sinusoidal_embedding_1d`
- `precompute_freqs_cis_3d`
- `precompute_freqs_cis`
- `rope_apply`
- `DiTBlock`
- `RMSNorm`
- `SelfAttention`
- `CrossAttention`

これらは `FastWAM_ActionDiT.py` にも一部吸収済みで重複する。
ただし、まずは FastWAM 公式構造を崩さず `FastWAM_WanVideoDiT.py` 側にも保持する。

重複整理は後工程。

理由:
- 最初から共通化すると、公式 FastWAM との差分が増える
- いまの目的は StarVLA 既存 module 配下で FastWAM を再現すること
- 公式挙動の確認が済むまで helper 共通化はリスクが高い

### `mot.py` の配置判断

`mot.py` は world model そのものではなく、video/action experts を結合する framework-level orchestration。

第一候補:
- `starVLA/model/framework/WM4A/WanFastWAM.py` へ `MoT` class を吸収

ただし、`MoT` が大きくなりすぎる場合は、既存 module 配下の追加 file として次を許容する。

- `starVLA/model/framework/WM4A/FastWAM_MoT.py`

判断基準:
- `WanFastWAM.py` が読みにくくなるなら分離
- ただし新規 folder は作らない

### 次の実装ステップ

1. `src/fastwam/models/wan22/wan_video_dit.py` を
   `starVLA/model/modules/world_model/FastWAM_WanVideoDiT.py` として配置する
2. import を StarVLA 向けに調整する
   - `fastwam.utils.logging_config.get_logger` を `initialize_overwatch` に置換
   - `.helpers.gradient` 依存を局所化または既存実装に置換
3. `MoT` を `WanFastWAM.py` 内へ吸収できるか試す
4. 難しければ `framework/WM4A/FastWAM_MoT.py` として配置する
5. `WanFastWAM.py` を `Wan2 hidden states -> ActionDiT` ではなく、
   `FastWAM_WanVideoDiT + FastWAM_ActionDiT + MoT` の構成へ置き換える

### Step 5 実装結果

配置済み:
- `starVLA/model/modules/world_model/FastWAM_WanVideoDiT.py`
- `starVLA/model/framework/WM4A/FastWAM_MoT.py`

`FastWAM_WanVideoDiT.py`:
- FastWAM 公式 `src/fastwam/models/wan22/wan_video_dit.py` を配置
- StarVLA 向けに logger と gradient checkpoint helper import だけ調整
- 公式との差分は import 周辺のみ
- `python -m py_compile starVLA/model/modules/world_model/FastWAM_WanVideoDiT.py` 成功

`FastWAM_MoT.py`:
- FastWAM 公式 `src/fastwam/models/wan22/mot.py` を `framework/WM4A` 直下に配置
- `WanFastWAM.py` へ直接吸収するには 556 行と大きく、framework 入口の見通しが悪くなるため分離した
- 新規 folder は作っていない
- StarVLA 向けに helper import と logger だけ調整
- 公式との差分は import 周辺のみ
- `python -m py_compile starVLA/model/framework/WM4A/FastWAM_MoT.py` 成功

次に必要な作業:
- フル FastWAM 用の framework 入口を `starVLA/model/framework/WM4A/FastWAM.py` として新設する
- `FastWAM.py` を `FastWAM_WanVideoDiT + FastWAM_ActionDiT + FastWAM_MoT` で構築する形へ実装する
- そのために `FastWAM.training_loss()` の StarVLA 版を `FastWAMFramework.forward()` へ移植する
- VAE/text encoder はまず既存 `Wan2.py` のものを再利用するか、FastWAM 公式 loader が必要かを判断する

## Step 6. フル FastWAM framework 入口

配置方針:
- `starVLA/model/framework/WM4A/FastWAM.py`

理由:
- `WanFastWAM.py` は既存 `Wan2.py` hidden states と FastWAM ActionDiT を接続する初期 prototype として残す
- 論文相当の FastWAM joint training は別 framework 名 `FastWAM` として扱う
- StarVLA の framework auto-discovery は `WM4A/*.py` を自動 import するため、新規 folder は不要

実装済み:
- `FastWAMDefaultConfig`
- `@FRAMEWORK_REGISTRY.register("FastWAM")`
- `FastWAMFramework`
- `FastWAM_WanVideoDiT`, `ActionDiT`, `MoT` の構築
- MoT が要求する `num_heads`, `attn_head_dim`, `num_layers` の整合性チェック

確認済み:
- `python -m py_compile starVLA/model/framework/WM4A/FastWAM.py`
- framework auto-discovery で `FastWAM_registered=True`

未実装:
- VAE / text encoder / tokenizer loading
- StarVLA batch から FastWAM `sample` 形式への変換
- `FastWAM.training_loss()` の `FastWAMFramework.forward()` への移植
- `FastWAM.infer_action()` / `infer_joint()` の移植

次の作業:
- `FastWAM.training_loss()` を `FastWAM.py` に移植する前に、VAE/text encoder を既存 `Wan2.py` から再利用するか、FastWAM 公式 loader を配置するかを決める
