# FastWAM ABCI Checkpoint Conversion and Inference Plan 批評

## 1. 文書の目的

この文書は、以下の計画を、これまでのチャット履歴と現在の実装状況に照らして批評したものです。

対象計画:

```text
FastWAM_ABCI_checkpoint_conversion_and_inference_plan.md
```

現在の主目的は、学習ではなく、次の2点です。

```text
1. StarVLA上で公式Fast-WAM checkpoint由来の重みを使い、Action推論を実行する
2. 同一入力に対する公式Fast-WAM実装とStarVLA実装のActionを比較し、実装再現性を確認する
```

本批評では、以下を区別します。

- **事実**: これまでのチャット履歴、既存のSmoke Test、対象計画に明記されている内容
- **未確認事項**: 現時点ではコードやABCI実行結果を見なければ判断できない事項
- **提案**: 計画をより検証可能にするために追加・修正すべき内容

---

# 2. 総評

## 結論

対象計画の方向性は妥当です。

特に、以前の計画が「Fast-WAMをStarVLAへ学習可能な形で完全移植する」ことを重視していたのに対し、今回の計画は、現在の目的である「公式checkpointを使ったAction推論」に適切に絞られています。

次の順序も合理的です。

```text
公式checkpointをStarVLAへ直接ロード
→ latent入力で推論経路を確認
→ 差分を見て変換規則を決定
→ StarVLA形式のcheckpointを保存
→ raw画像入力で推論
→ 公式実装と比較
```

一方で、現状の計画は、次の目的には十分です。

```text
公式checkpointをロードし、有限値のActionを出す
```

しかし、次の目的を証明するには検証工程が不足しています。

```text
StarVLA実装が公式Fast-WAM実装を正しく再現している
```

特に不足しているのは、以下の4点です。

```text
1. checkpoint全体のロード率確認
2. 変換前後のRound-trip Parity
3. 推論前処理Parity
4. scheduler各stepのParity
```

---

# 3. これまでのチャット履歴との比較

## 3.1 初期の目標

初期の計画では、主に以下を目標としていました。

```text
- Fast-WAMのVideo DiTをStarVLAへ移植
- Action DiTを移植
- MoTを移植
- 動画Flow MatchingとAction Flow Matchingを再現
- Dataset Parityを確認
- Trainerへ統合
- 分散学習とResumeを確認
- 将来的にLIBEROやRoboTwinを再学習する
```

これは、Fast-WAMをStarVLA上で再学習可能な状態まで持っていく計画でした。

## 3.2 現在の目標

その後、チャットで目的が次のように明確化されました。

```text
学習は当面行わない
↓
公式Fast-WAM checkpointをStarVLAへ読み込む
↓
Action推論を動かす
↓
公式Fast-WAM実装と出力を比較する
```

今回の計画は、この変更を正しく反映しています。

## 3.3 評価

| 項目 | 以前の方針 | 今回の計画 | 評価 |
|---|---|---|---|
| 学習対応 | 主目的 | 対象外 | 現在の目的に適合 |
| 公式checkpoint | 後半工程 | 最優先 | 改善 |
| Action推論 | 後半工程 | 中心目標 | 改善 |
| ABCI利用 | 学習・分散検証 | full model診断 | 妥当 |
| Dataset Parity | 最優先 | ほぼ省略 | 推論前処理分は戻すべき |
| 公式実装比較 | 1-step loss中心 | Action中心 | 現在の目的に適合 |
| raw画像経路 | 学習データ経由 | 推論入力経路 | 妥当 |

---

# 4. これまでに確認済みのSmoke Testとの関係

## 4.1 確認済みの事項

これまでのSmoke Testでは、少なくとも以下が確認されています。

```text
- FastWAM frameworkを構築できる
- tiny構成のVideo Expert、Action Expert、MoTがforwardできる
- action_lossを返せる
- backwardできる
- precomputed latentからpredict_action()を呼べる
- first_frame_causal条件を検査できる
- raw exampleをencoder adapterへ流せる
- encoder未ロード時のエラー経路を確認できる
```

## 4.2 Smoke Testが証明していること

これらが証明しているのは、主に次の点です。

```text
StarVLA内に追加したFastWAMコードの接続経路が動く
```

## 4.3 まだ証明していないこと

一方で、以下はまだ証明されていません。

```text
- 公式サイズのモデル構成が一致している
- 公式checkpointの全重みが対応している
- 公式checkpointが十分な割合でロードされている
- 公式schedulerと同じ更新式になっている
- 公式前処理と同じ入力が作られている
- 公式実装と同じActionが得られる
```

したがって、成功段階を次のように分けると整理しやすくなります。

```text
Level 0: tiny smoke成功
Level 1: full-size model構築成功
Level 2: checkpointロード成功
Level 3: StarVLA latent推論成功
Level 4: 変換前後Parity成功
Level 5: 公式実装とのlatent parity成功
Level 6: raw画像推論成功
Level 7: raw画像 parity成功
```

---

# 5. 対象計画の良い点

## 5.1 変換より先に直接ロード診断を行う

これは妥当です。

公式checkpointの実際のkeyとshapeを確認せずに変換スクリプトを作ると、以下の問題が起こり得ます。

```text
- 不要なprefix変換
- 誤ったmodule対応
- 一部重みの未ロード
- 同じ重みの二重保存
- 公式形式との差分を誤認する
```

ABCI上でfull-size modelを構築し、実物のcheckpointに対して、

```text
missing_keys
unexpected_keys
shape mismatch
```

を確認してから変換規則を決める方針は堅実です。

## 5.2 latent推論をraw画像推論より先に行う

これも妥当です。

latent入力から始めれば、Fast-WAM本体の問題と、次のencoder・前処理問題を分離できます。

```text
- Wan2.2 VAE
- text encoder
- tokenizer
- resize
- camera結合
- 画像正規化
```

latent modeで失敗した場合、原因を次の範囲へ絞り込めます。

```text
- MoT
- Action DiT
- first_frame_causal mask
- proprio/state token
- scheduler
- denoise loop
```

## 5.3 「Actionが出る」と「公式再現」を分けている

次の2つを別の成功条件としている点は正しいです。

```text
1. Actionが出る
2. 公式実装と一致する
```

正しいshapeの有限値Actionが出ても、公式実装を再現したことにはなりません。

## 5.4 最終的にStarVLA形式へ保存する方針

公式形式を毎回解釈するより、StarVLA側の読み込み形式へ変換する方針は運用上合理的です。

ただし、後述するように、保存形式は単一の巨大な`framework state_dict`よりも、config・統計・変換レポートを含むcheckpoint packageの方が適しています。

---

# 6. 修正が必要な点

# 6.1 missing_keysとunexpected_keysだけでは不十分

## 現在の計画

現在の計画では、次のように判断しています。

```text
missing_keys_count=0
unexpected_keys_count=0
```

であれば、変換はほぼ再保存だけで済む。

## 問題

この条件だけでは、十分なロードが行われたことを証明できません。

以下のケースが残ります。

```text
- load対象がcheckpointの一部だけになっている
- 公式checkpointに含まれないmoduleがランダム初期化のまま残る
- strict=Falseで一部moduleが検査対象外になる
- 同名keyだが意味の異なるparameterへロードしている
- dtype変換が意図せず行われている
- 一部keyが重複して別moduleへマッピングされている
```

## 提案

ロードレポートへ以下を追加してください。

```text
checkpoint_tensor_count
checkpoint_parameter_count
loaded_tensor_count
loaded_parameter_count
parameter_coverage_ratio
shape_matched_count
dtype_converted_count
skipped_keys
duplicated_mapping
```

最も重要なのは次です。

```text
parameter_coverage_ratio
= loaded_parameter_count / checkpoint_parameter_count
```

接続成功の条件に、ロード率を含めるべきです。

例:

```text
parameter_coverage_ratio >= 0.999
```

ただし、意図的に外部Wan2.2からロードするmoduleがある場合は、その除外理由をconversion reportへ記録します。

---

# 6.2 checkpoint以外の重みの取得元が不明確

## 事実

確認済みの公式checkpoint top-level keysは、計画上では次です。

```text
mot
proprio_encoder
step
torch_dtype
```

## 未確認事項

以下が現時点では不明です。

```text
- Video DiT全体がmot内に含まれているか
- Action DiT全体がmot内に含まれているか
- Action encoder／decoderがmot内に含まれているか
- VAEは公式checkpointに含まれず、Wan2.2から読むのか
- text encoderは公式checkpointに含まれず、Wan2.2から読むのか
- tokenizerのrevisionは何か
```

## 提案

最初に「重み構成表」を作成してください。

| モジュール | 重みの取得元 | checkpoint key | 推論時必須 | 状態 |
|---|---|---|---|---|
| Video DiT | 公式Fast-WAM checkpointまたはWan2.2 | 要確認 | 必須 | 未確認 |
| Action DiT | 公式Fast-WAM checkpoint | `mot.*`内を確認 | 必須 | 未確認 |
| Shared Attention / MoT | 公式Fast-WAM checkpoint | `mot.*` | 必須 | 確認対象 |
| Proprio Encoder | 公式Fast-WAM checkpoint | `proprio_encoder` | state使用時必須 | 確認対象 |
| VAE | Wan2.2 | 外部パス | raw推論時必須 | 未確認 |
| Text Encoder | Wan2.2 | 外部パス | 言語条件使用時必須 | 未確認 |
| Tokenizer | Wan2.2 | 外部パス | 言語条件使用時必須 | 未確認 |

この表を埋めるまでは、「公式checkpoint全体をロードできた」とは判定しない方が安全です。

---

# 6.3 推論前処理Parityが不足している

## 問題

学習を行わないため、学習Dataset全体のParityは不要です。

しかし、raw画像から公式実装と同じActionを得るためには、推論用前処理のParityが必要です。

現在の計画では、

```text
precomputed latent比較
→ raw画像比較
```

と進みますが、その間に以下の比較工程が必要です。

```text
raw画像
→ camera order
→ resize
→ camera concat
→ 画像正規化
→ VAE latent
→ prompt生成
→ tokenize
→ text context
→ context mask
→ state正規化
```

## 特に確認すべき点

LIBERO checkpoint名は次です。

```text
libero_uncond_2cam224.pt
```

したがって、少なくとも以下を明確にする必要があります。

```text
- 2カメラを別々に渡すのか
- 結合済み画像を渡すのか
- front cameraとwrist cameraの順序
- 結合方向
- resize前後のサイズ
```

## 提案

raw modeのCLIを、単一`--image`ではなく次のように変更する方が明確です。

```bash
--front-image front.png \
--wrist-image wrist.png
```

または、再現用固定サンプルを1ファイルへまとめます。

```bash
--sample-npz fixed_libero_sample.npz
```

`fixed_libero_sample.npz`には最低限以下を入れます。

```text
front_image
wrist_image
state
prompt
metadata
```

---

# 6.4 灰色ダミー画像と実画像Parityを分離する

## 現在の計画

`--image`を省略した場合、灰色ダミー画像を使うとされています。

## 評価

これはencoderを含む経路の接続確認には有効です。

ただし、以下は検証できません。

```text
- camera順
- resize方法
- RGB値の扱い
- 実際の入力分布
- 公式画像前処理との一致
```

## 提案

modeまたは成功条件を分けてください。

```text
raw_smoke:
灰色画像でVAE・text encoderを含む経路だけ確認

raw_parity:
固定した実画像・実stateで公式実装と比較
```

---

# 6.5 num_inference_steps=1はSmoke Test専用

## 評価

1 stepは次の確認には適しています。

```text
- shape
- NaN/Inf
- mask
- API
- メモリ
```

ただし、公式再現には不十分です。

## 公式再現時に固定すべきもの

```text
- num_inference_steps
- timestep sequence
- noise schedule
- scheduler更新式
- initial action noise
- CFG scale
- dtype
- device
```

## 提案

次の順序で確認してください。

```text
1 step:
経路確認

2 step:
step間更新確認

公式step数:
最終Parity確認
```

また、最終Actionだけではなく、各denoise stepの中間値を比較してください。

```text
action_latent_step_0
action_latent_step_1
...
action_latent_step_N
```

差が最初に発生したstepを特定できます。

---

# 6.6 `uncond`の意味が未確認

## 未確認事項

checkpoint名に含まれる`uncond`が具体的に何を意味するかは、対象計画だけでは判断できません。

可能性としては以下がありますが、推測で決めるべきではありません。

```text
- 言語条件を使用しないmodel variant
- classifier-free guidance用のunconditional設定
- 空promptで学習したvariant
- release上の命名
```

## 問題

未確認のまま次を公式入力とみなすのは危険です。

```bash
--prompt ""
```

## 提案

Step 0へ次を追加してください。

```text
- `uncond`の定義を公式config／コードで確認
- text contextを推論で使用するか確認
- 空promptが正しいか確認
- CFG branchがあるか確認
- text encoderをraw推論でロードする必要があるか確認
```

---

# 6.7 state正規化とAction後処理が不足している

## 事実

checkpointとともにdataset stats JSONが存在します。

```text
libero_uncond_2cam224_dataset_stats.json
robotwin_uncond_3cam_384_dataset_stats.json
```

これらは通常、次の処理に関係します。

```text
- state/proprio正規化
- action正規化
- 推論後のaction unnormalization
```

## 問題

現在の成功条件は、次までです。

```text
actions_shape
finite
min
max
mean
```

しかし、返却値が以下のどちらか不明です。

```text
normalized_actions
実ロボット単位へ復元済みのactions
```

## 提案

返却値を明示的に分けてください。

```python
{
    "normalized_actions": ...,
    "actions": ...,
}
```

比較も次の2段階で行います。

```text
1. normalized action parity
2. unnormalized action parity
```

また、stateについても次を保存します。

```text
state_raw
state_normalized
```

---

# 6.8 StarVLA形式checkpointの保存方法

## 現在の案

```python
{
    "framework": model.state_dict(),
    ...
}
```

## 問題

full modelのstate dictに以下を全て含めると、非常に大きなファイルになる可能性があります。

```text
- Video DiT
- Action DiT
- VAE
- text encoder
- tokenizer関連情報
- MoT
```

## 提案

単一ファイルよりcheckpoint packageを推奨します。

```text
checkpoints/fastwam_starvla/libero_uncond_2cam224/
├── fastwam_weights.pt
├── config.yaml
├── dataset_stats.json
├── manifest.json
└── conversion_report.json
```

### `manifest.json`に含める内容

```text
source_checkpoint
source_checkpoint_sha256
source_format
source_step
official_fastwam_commit
starvla_commit
wan_model_id
wan_revision
tokenizer_revision
dtype
action_dim
proprio_dim
action_horizon
camera_order
image_size
conversion_version
```

### `conversion_report.json`に含める内容

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

---

# 6.9 変換前後のRound-trip Testが必要

## 問題

変換済みcheckpointを保存した後、変換によって出力が変化していないことを確認する工程がありません。

## 比較すべき3系統

```text
A: 公式Fast-WAM実装
B: StarVLA + 公式形式checkpoint直接ロード
C: StarVLA + 変換済みStarVLA形式checkpoint
```

## 必要な比較

```text
B ≈ C:
変換処理の正しさ

A ≈ B:
StarVLA移植実装の正しさ
```

## 提案するRound-trip Test

```text
1. 公式形式checkpointをStarVLAへ直接ロード
2. 固定入力でActionと中間値を保存
3. StarVLA形式へ変換
4. 新規processで変換済みcheckpointをロード
5. 同じ固定入力でActionと中間値を保存
6. 直接ロード結果と比較
```

このテストは、公式実装との比較より前に実施してください。

---

# 6.10 許容誤差を事前に定義する

## 問題

計画には「許容誤差内」とありますが、数値が定義されていません。

## 提案

最初はFP32で比較してください。

推奨する初期基準:

| 対象 | 初期基準 |
|---|---|
| raw uint8画像 | 完全一致 |
| CPU float32前処理 | `atol <= 1e-7` |
| state/action正規化 | `atol <= 1e-7` |
| VAE latent | `rtol=1e-5, atol=1e-6` |
| text context | `rtol=1e-5, atol=1e-6` |
| Transformer中間出力 | `rtol=1e-5, atol=1e-6` |
| FP32最終Action | `rtol=1e-5, atol=1e-6` |
| BF16最終Action | 初期値として`rtol=1e-2, atol=1e-2` |

推奨順序:

```text
FP32・少数step
→ FP32・公式step数
→ BF16・公式step数
```

BF16のみで始めると、実装差と丸め誤差を区別しにくくなります。

---

# 7. 現在の計画で最も大きな論理上の問題

## 現在のStep 2

```text
ダミーlatent
→ actions_shape=[1,32,7]
→ finite=true
```

## 問題

これは推論経路が壊れていないことしか示しません。

checkpointがほとんどロードされていないランダム初期化modelでも、正しいshapeの有限値Actionが出る可能性があります。

## 提案する成功判定

### 接続成功

```text
- full-size modelが構築できる
- checkpoint parameter coverageが規定値以上
- shape mismatchがない
- latent推論がfinite
```

### 重み利用成功

```text
- checkpointロード前後で固定入力の出力が変化する
- 同じcheckpointを再ロードすると同じ出力になる
- 直接ロードと変換済みcheckpointの出力が一致する
```

### 実装再現成功

```text
- 公式実装とStarVLA実装の各denoise stepが一致する
```

### raw入力再現成功

```text
- 画像前処理が一致する
- VAE latentが一致する
- text contextが一致する
- state正規化が一致する
- 最終Actionが一致する
```

---

# 8. 推奨する修正版の実行順序

# Step 0. 再現条件を固定する

以下を`reproduction_manifest.yaml`へ記録します。

```text
official_fastwam_commit
starvla_commit
checkpoint_path
checkpoint_sha256
dataset_stats_path
dataset_stats_sha256
wan_model_id
wan_revision
vae_revision
text_encoder_revision
tokenizer_revision
prompt_template
camera_order
image_size
action_dim
proprio_dim
action_horizon
scheduler_type
scheduler_parameters
num_inference_steps
dtype
device
seed
uncond_definition
```

# Step 1. checkpoint inventoryを出力する

モデルへロードする前に、checkpoint自体を調査します。

出力:

```text
top_level_keys
state_dict_keys
tensor_shapes
tensor_dtypes
tensor_count
parameter_count
module_prefix_summary
```

# Step 2. full-size modelをcheckpointなしで構築する

確認:

```text
total_parameter_count
trainable_parameter_count
module別parameter count
layer数
hidden dimension
attention head数
action_dim
proprio_dim
action_horizon
```

# Step 3. 公式checkpointを直接ロードする

確認:

```text
missing_keys
unexpected_keys
shape_mismatch
skipped_keys
dtype_converted_keys
loaded_parameter_count
parameter_coverage_ratio
```

# Step 4. StarVLA latent smokeを実行する

条件:

```text
1 denoise step
固定seed
固定shape
```

確認:

```text
actions_shape
finite
min
max
mean
peak_gpu_memory
```

# Step 5. StarVLA形式へ変換しRound-trip Parityを確認する

比較:

```text
StarVLA + 公式形式直接ロード
vs
StarVLA + 変換済み形式ロード
```

確認:

```text
parameter checksum
initial action noise
各denoise step
最終normalized_actions
```

# Step 6. 公式実装とのlatent parityを行う

両実装で固定するもの:

```text
latent
context
context_mask
state
state_normalized
initial_action_noise
timesteps
scheduler_parameters
num_inference_steps
dtype
device
```

比較対象:

```text
proprio token
attention mask
Video DiT出力
Action DiT各層出力
各denoise stepのaction latent
最終normalized_actions
```

# Step 7. 推論前処理Parityを行う

固定LIBEROサンプルについて比較します。

```text
front_image_raw
wrist_image_raw
front_resized
wrist_resized
camera_concat
video_normalized
vae_latent
prompt
token_ids
context
context_mask
state_raw
state_normalized
```

# Step 8. raw smokeを行う

灰色画像を用い、経路だけ確認します。

確認:

```text
VAE load
text encoder load
tokenizer load
raw predict_action完走
finite
```

# Step 9. raw parityを行う

固定した実LIBEROサンプルで比較します。

```text
公式Fast-WAM実装
vs
StarVLA実装
```

比較:

```text
VAE latent
text context
state_normalized
各denoise step
normalized_actions
unnormalized actions
```

---

# 9. 計画へ追加すべき成果物

```text
docs/
├── fastwam_reproduction_manifest.yaml
├── fastwam_checkpoint_inventory.json
├── fastwam_direct_load_report.json
├── fastwam_conversion_report.json
├── fastwam_roundtrip_parity_report.json
├── fastwam_latent_parity_report.json
└── fastwam_raw_parity_report.json
```

固定入力:

```text
tests/fixtures/fastwam/
├── libero_fixed_sample.npz
├── latent_fixed_input.npz
├── initial_action_noise.pt
├── timesteps.pt
└── metadata.json
```

---

# 10. Codex CLIへの具体的な実装依頼

Codex CLIには、以下の順序で依頼するのが安全です。

## Task 1: 既存コード調査

```text
現在のFastWAM実装について、以下を調査してください。

1. 公式checkpointのpayload["mot"]がどのmoduleへロードされるか
2. payload["proprio_encoder"]がどのmoduleへロードされるか
3. load_state_dictでstrictがどう設定されているか
4. missing_keysとunexpected_keys以外にshape mismatchをどのように扱うか
5. Video DiT、Action DiT、VAE、text encoderの重みがそれぞれどこからロードされるか
6. predict_action()がnormalized actionを返すか、unnormalized actionを返すか
7. schedulerのclass、timestep生成、step更新式
8. libero_uncond_2cam224の2カメラ入力形式
9. `uncond`の意味
10. dataset_stats JSONがどこで使われるか

コード変更はまだ行わず、ファイルパス・class名・関数名・該当行を含む調査報告をMarkdownで作成してください。
```

## Task 2: checkpoint inventory

```text
公式Fast-WAM checkpointを読み取り、以下をJSONへ出力するスクリプトを作成してください。

- top-level keys
- state dict keys
- tensor shape
- tensor dtype
- tensor numel
- prefix別tensor count
- prefix別parameter count
- checkpoint全体parameter count

12GB級checkpointを扱うため、不要なmodel構築は行わず、CPUメモリ使用量に注意してください。
```

## Task 3: ロードレポート強化

```text
現在のofficial checkpoint load smokeへ、以下を追加してください。

- checkpoint_parameter_count
- loaded_parameter_count
- parameter_coverage_ratio
- skipped_keys
- shape_mismatch
- dtype_converted_keys
- duplicated_mapping

レポートは標準出力に加えJSONでも保存してください。
```

## Task 4: Round-trip Parity

```text
公式形式checkpointを直接ロードしたStarVLA modelと、変換済みStarVLA checkpointをロードしたmodelの出力を比較するテストを作成してください。

固定するもの:
- latent
- context
- context_mask
- state
- initial action noise
- timesteps
- scheduler設定
- dtype
- device

比較するもの:
- 各denoise stepのaction latent
- 最終normalized_actions

最初はFP32、1 stepで実装し、その後公式step数に拡張できる構成にしてください。
```

## Task 5: raw前処理ダンプ

```text
raw画像からpredict_actionへ至る前処理の各段階を保存できるdebug exportを実装してください。

保存対象:
- front raw image
- wrist raw image
- resized images
- concatenated image
- normalized image
- VAE latent
- prompt
- token ids
- context
- context mask
- raw state
- normalized state

保存形式はnpzとJSON metadataにしてください。
```

---

# 11. 最終評価

## 事実

今回の計画は、以前の「学習可能な完全移植」から、「公式checkpointを用いた推論再現」へ適切に目的を絞っています。

大枠の順序も妥当です。

```text
直接ロード
→ latent推論
→ 変換
→ raw推論
→ 公式比較
```

## 現時点で未確認の事項

```text
- `mot`内にどこまで重みが含まれるか
- `uncond`が何を意味するか
- StarVLA実装と公式実装のschedulerが一致するか
- raw modeが2カメラ入力を正しく扱うか
- dataset statsをどこで適用するか
- checkpoint全体の何%が実際にロードされるか
```

## 判断

この計画は、

```text
公式checkpointをロードして有限値Actionを出す
```

ための計画としては良好です。

ただし、

```text
公式Fast-WAMをStarVLA上で正しく再現したと証明する
```

ためには、以下を必須工程として追加する必要があります。

```text
1. checkpoint parameter coverage
2. 変換前後Round-trip Parity
3. 推論前処理Parity
4. scheduler各stepのParity
```

これらを追加すれば、これまでのSmoke Test成果を活かしつつ、現在の目的へ最短経路で進める、検証可能な計画になります。
