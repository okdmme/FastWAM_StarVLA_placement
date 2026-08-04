# FastWAM ABCI Cautious Verification Fallback Plan

## この文書の位置づけ

この文書は、`FastWAM_ABCI_checkpoint_conversion_and_inference_plan.md` の本番診断を優先しつつ、必要になった場合に実装する慎重案をまとめる。

現在の進め方は以下。

```text
まず本番診断を実行する
↓
通れば、その結果を使って変換済みcheckpoint作成へ進む
↓
落ちる、または公式再現性を証明できない場合は、この慎重案へ戻る
```

つまり、この文書は最初に必ず全て実装する計画ではない。
ABCI本番診断で得られたログを見て、必要になった部分から実装するための戻り先である。

## なぜ慎重案を残すのか

本番診断は、最短で以下を確認するためには有効。

```text
- full-size StarVLA FastWAMをABCI GPU上で構築できるか
- 公式FastWAM checkpointを読めるか
- proprio_encoder込みでロードできるか
- latent入力からActionが出るか
- raw画像経路へ進めるか
```

しかし、本番診断だけでは次は証明できない。

```text
- checkpoint全体の何%が本当にロードされたか
- 変換済みcheckpointが直接ロード結果と完全に同じか
- 公式FastWAM実装とStarVLA実装が同じ計算をしているか
- raw画像、2カメラ、state正規化、action正規化が公式と一致しているか
- scheduler各stepの中間値が一致しているか
```

したがって、本番診断で「動いた」としても、公式再現性の判断には慎重案が必要になる可能性が高い。

## 本番診断を先に行ってよい条件

次の条件を満たすなら、本番診断を先に実行してよい。

```text
- ABCIログインノードで必要ファイルが配置済み
- .venv-py311 が存在する
- Wan2.2 diffusers encoder一式がある
- FastWAM公式checkpointがある
- qsubでH200等のGPU計算ノードへ投げられる
- 本番診断の結果を「成功確定」ではなく「診断ログ」として扱う
```

重要なのは最後。

本番診断が通っても、公式再現が成功したとはまだ判断しない。
判断できるのは、StarVLA上の推論経路が公式checkpoint由来の重みで動き始めた、という段階まで。

## 本番診断で見るもの

本番診断では以下を確認する。

```text
qsub docs/abci_fastwam_official_infer_smoke.pbs
```

確認ログ:

```text
abci_fastwam_official_infer.log
```

最低限見る値:

```text
loaded
step
proprio_dim
missing_keys_count
unexpected_keys_count
actions_shape
finite
min
max
mean
GPU memory
traceback
```

期待する値:

```text
loaded: ["mot", "proprio_encoder"]
actions_shape: [1, 32, 7]
finite: true
```

`missing_keys_count` と `unexpected_keys_count` はゼロが理想。
ただし、ゼロでも慎重案が不要になるわけではない。

## 慎重案へ戻る条件

以下のどれかに当てはまる場合、この慎重案へ戻る。

```text
- loadで失敗する
- shape mismatchが出る
- missing_keys_count / unexpected_keys_count がゼロでない
- latent推論でNaN/Infが出る
- actions_shapeが期待値と違う
- raw推論でencoderまたは前処理が落ちる
- 変換済みcheckpointを作る必要が出る
- 公式FastWAM実装とAction比較する段階へ進む
```

## 慎重案 Task 1. Checkpoint Inventory

### 目的

公式checkpointそのものに何が入っているかを、model構築なしで調べる。

### 出力

```text
docs/fastwam_checkpoint_inventory_libero.json
```

含める内容:

```text
top_level_keys
state_dict_keys
tensor_shapes
tensor_dtypes
tensor_numel
prefix別tensor count
prefix別parameter count
checkpoint_parameter_count
proprio_encoder shape
step
torch_dtype
```

### なぜ必要か

`missing_keys_count=0` だけでは、checkpoint全体を意味通りに使えているか判断できない。
どのprefixに何parameterあるかを先に把握すれば、ロード率や変換対象を定量的に判断できる。

## 慎重案 Task 2. Load Report強化

### 目的

公式checkpointをStarVLAへロードしたとき、どれだけの重みが実際に使われたかを数値で出す。

### 追加する値

```text
checkpoint_tensor_count
checkpoint_parameter_count
loaded_tensor_count
loaded_parameter_count
parameter_coverage_ratio
shape_matched_count
shape_mismatch
skipped_keys
dtype_converted_keys
duplicated_mapping
```

### 成功条件

初期基準:

```text
parameter_coverage_ratio >= 0.999
shape_mismatch = []
```

ただし、VAE/text encoder/tokenizerのようにcheckpoint外部から読むものは、除外理由をreportへ明記する。

### なぜ必要か

ランダム初期化moduleが残っていても、Action shapeだけは正しく出る可能性がある。
ロード率を見ないと、公式checkpointを本当に使えているとは言えない。

## 慎重案 Task 3. StarVLA Checkpoint Package保存

### 目的

公式形式checkpointを、StarVLAで再利用しやすい形式へ変換して保存する。

### 保存先案

```text
checkpoints/fastwam_starvla/libero_uncond_2cam224/
├── fastwam_weights.pt
├── config.yaml
├── dataset_stats.json
├── manifest.json
└── conversion_report.json
```

### manifestに入れるもの

```text
source_checkpoint
source_checkpoint_sha256
source_format
source_step
official_fastwam_commit
starvla_commit
wan_model_id
wan_revision
dtype
action_dim
proprio_dim
action_horizon
camera_order
image_size
conversion_version
```

### conversion_reportに入れるもの

```text
missing_keys
unexpected_keys
remapped_keys
skipped_keys
shape_mismatch
dtype_converted_keys
checkpoint_parameter_count
loaded_parameter_count
parameter_coverage_ratio
```

### なぜ単一ptだけにしないのか

公式再現性をあとで追うには、重みだけでは不足する。
どの公式checkpointから、どの変換規則で、どのconfigと統計を使って作ったかを残す必要がある。

## 慎重案 Task 4. Round-trip Parity

### 目的

変換済みStarVLA checkpointが、公式形式を直接ロードしたStarVLA modelと同じ出力を返すか確認する。

比較する2系統:

```text
B: StarVLA + 公式形式checkpoint直接ロード
C: StarVLA + 変換済みStarVLA形式checkpoint
```

固定する入力:

```text
first_frame_latents
context
context_mask
state/proprio
initial_action_noise
timesteps
scheduler設定
num_inference_steps
dtype
device
seed
```

比較対象:

```text
各denoise stepのaction latent
最終normalized_actions
```

### なぜ必要か

変換済みcheckpointを作っても、変換によって出力が変わってしまえば意味がない。
公式実装との比較より前に、StarVLA内部で直接ロードと変換ロードが一致することを確認する。

## 慎重案 Task 5. 公式実装とのLatent Parity

### 目的

公式FastWAM実装とStarVLA実装が、FastWAM本体で同じ計算をしているか確認する。

比較する2系統:

```text
A: 公式FastWAM実装
B: StarVLA + 公式形式checkpoint直接ロード
```

最初はraw画像ではなく、precomputed latent/context/stateで比較する。

### なぜraw画像から始めないのか

raw画像から始めると、以下の差が混ざる。

```text
VAE
tokenizer
text encoder
resize
camera order
state正規化
prompt処理
```

latent/context/stateを固定すれば、FastWAM本体の差分だけを見られる。

## 慎重案 Task 6. 推論前処理Parity

### 目的

raw画像からlatent/context/stateを作る過程が公式実装と一致しているか確認する。

確認対象:

```text
front_image_raw
wrist_image_raw
camera_order
resize
camera_concat
image_normalization
vae_latent
prompt
token_ids
context
context_mask
state_raw
state_normalized
```

### 特に注意する点

LIBERO checkpoint名は以下。

```text
libero_uncond_2cam224.pt
```

`2cam224` なので、単一画像の灰色ダミーだけでは公式入力とは言えない。
本番診断のraw modeは経路確認用であり、公式再現性確認用ではない。

## 慎重案 Task 7. raw parity

### 目的

固定した実LIBEROサンプルで、公式実装とStarVLA実装の最終Actionを比較する。

必要な固定サンプル:

```text
tests/fixtures/fastwam/libero_fixed_sample.npz
```

含めるもの:

```text
front_image
wrist_image
state
prompt
metadata
```

比較:

```text
normalized_actions
unnormalized_actions
各denoise stepのaction latent
```

## 許容誤差

最初はFP32で比較する。

推奨初期基準:

```text
state/action正規化: atol <= 1e-7
VAE latent: rtol=1e-5, atol=1e-6
text context: rtol=1e-5, atol=1e-6
Transformer中間出力: rtol=1e-5, atol=1e-6
FP32 final action: rtol=1e-5, atol=1e-6
BF16 final action: rtol=1e-2, atol=1e-2
```

本番診断はABCIメモリ都合で `bfloat16` を使う。
公式再現性比較では、可能ならFP32の少数stepから始める。

## ログインノードで行うこと

ログインノードで行ってよいこと:

```text
git status確認
必要ファイルの有無確認
venvの存在確認
pip freeze / import確認
小さいJSON/manifest作成
checkpoint inventoryの軽量実行
qsub投入
ログ確認
```

ログインノードで避けること:

```text
full-size model構築
12GB checkpointの通常torch.load
Wan2.2 text encoder実ロード
VAE実ロード
推論
長時間GPU前提処理
```

これらはGPU計算ノードで行う。

## GPU計算ノードで行うこと

```text
full-size model構築
公式checkpointロード
latent推論
raw smoke
変換済みcheckpoint作成
Round-trip parity
公式実装との比較
```

## 本番診断との関係

本番診断で通れば、次は慎重案Task 3以降へ進む。

```text
本番診断 load/latent 成功
↓
StarVLA checkpoint package保存
↓
Round-trip parity
↓
raw smoke
↓
公式実装とのlatent parity
↓
raw parity
```

本番診断で落ちた場合は、慎重案Task 1とTask 2へ戻る。

```text
本番診断失敗
↓
checkpoint inventory
↓
load report強化
↓
差分修正
↓
再度本番診断
```

## この慎重案を実装する判断基準

すぐに実装する必要がある場合:

```text
- 本番診断がloadで落ちた
- missing/unexpectedが出た
- latent推論が落ちた
- 変換済みcheckpointを作る段階に入った
- 公式実装と比較する段階に入った
```

まだ実装しなくてよい場合:

```text
- まず本番診断のload/latentが通るかだけ見たい
- 公式再現性ではなく、推論入口の成立だけを確認したい
```

## まとめ

本番診断を先に実行する方針は許容できる。

ただし、本番診断の成功は「推論入口が動いた」という意味であり、「公式FastWAMをStarVLAで完全再現できた」という意味ではない。

公式再現性を主張するには、この慎重案のうち少なくとも以下が必要。

```text
1. checkpoint inventory
2. parameter coverage付きload report
3. StarVLA checkpoint package保存
4. Round-trip parity
5. 公式実装とのlatent parity
6. raw前処理parity
```
