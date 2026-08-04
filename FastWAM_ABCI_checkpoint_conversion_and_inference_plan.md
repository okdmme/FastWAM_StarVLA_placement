# FastWAM ABCI Checkpoint Conversion and Inference Plan

## 目的

StarVLA上に実装したFastWAMで、公式FastWAM checkpointを使ってAction推論できる状態にする。

最終的な成功条件は、以下の2段階に分ける。

```text
1. StarVLA上で公式checkpoint由来の重みを使い、Action推論が実行できる
2. 同一入力に対して、公式FastWAM実装とStarVLA実装のActionが一致、または許容誤差内に入る
```

ここでいう同一入力は、同じ画像、同じ言語条件、同じproprio/state、同じseed、同じscheduler step数、同じdtype/device条件を指す。

## 現状

公式FastWAM checkpointはローカルに配置済み。

```text
checkpoints/fastwam_release/
├── libero_uncond_2cam224.pt
├── libero_uncond_2cam224_dataset_stats.json
├── robotwin_uncond_3cam_384.pt
└── robotwin_uncond_3cam_384_dataset_stats.json
```

確認済みのcheckpoint構造:

```text
libero_uncond_2cam224.pt
- top-level keys: mot, proprio_encoder, step, torch_dtype
- action_dim: 7
- proprio_dim: 8
- action_horizon: 32

robotwin_uncond_3cam_384.pt
- top-level keys: mot, proprio_encoder, step, torch_dtype
- action_dim: 14
- proprio_dim: 14
- action_horizon: 32
```

StarVLA側には以下を実装済み。

```text
- FastWAM framework entry
- FastWAM_WanVideoDiT
- FastWAM_ActionDiT
- MoT
- payload["mot"] のロード入口
- payload["proprio_encoder"] のロード入口
- proprio/state を context 末尾へ1トークン追加する経路
- predict_action()
- ABCI用 official infer config
- ABCI用 smoke script / PBS
```

ただし、現時点ではまだ「StarVLA形式へ変換済みのcheckpoint」は作っていない。

## なぜ最初から変換済みcheckpointを作らないのか

公式checkpointをStarVLA形式へ変換すること自体は必要。

ただし、変換前にABCI上で公式checkpointを直接ロードして診断する。

理由は、ローカル環境では公式サイズのfull modelを安全に構築できないため。

```text
ローカルRAM: 約7.4GiB
公式checkpoint: 1本あたり約12GB
Wan2.2 encoder類: 約14GB
```

この状態でローカルでfull model loadやraw推論を行うと、メモリ不足で落ちる可能性が高い。

そのためABCIのH200環境で以下を先に確認する。

```text
- 公式checkpointのkeyがStarVLA側moduleに対応しているか
- shape mismatchがあるか
- missing_keys / unexpected_keys があるか
- proprio_encoderが正しく読めるか
- predict_action() が有限値のActionを返すか
```

この診断結果を見てから変換規則を確定する。

もし直接ロードで `missing_keys_count=0` かつ `unexpected_keys_count=0` なら、変換はほぼ「StarVLA形式で再保存」だけで済む。

もし差分が出るなら、その差分に対してkey変換、除外、shape対応を行う。

## 全体の流れ

### Step 1. ABCIで公式checkpointを直接ロードする

ABCI上のStarVLA rootで以下を実行する。

```bash
qsub docs/abci_fastwam_official_infer_smoke.pbs
```

このPBSは最初に以下を実行する。

```bash
python examples/simBenchmarks/LIBERO/eval_files/fastwam_official_infer_smoke.py \
  --config starVLA/config/training/starvla_fastwam_libero_official_infer.yaml \
  --mode load \
  --device cuda \
  --dtype bfloat16
```

確認する出力:

```text
loaded: ["mot", "proprio_encoder"]
missing_keys_count
unexpected_keys_count
missing_keys_sample
unexpected_keys_sample
```

なぜこのStepを最初に行うのか:

公式checkpointとStarVLA full-size modelのkey/shapeが本当に対応しているかを、実物で確認するため。
ここを飛ばして変換済みcheckpointを作ると、間違ったkey変換や不要な変換を入れるリスクがある。

### Step 2. ダミーlatentでStarVLA推論を確認する

同じPBS内で次に以下が実行される。

```bash
python examples/simBenchmarks/LIBERO/eval_files/fastwam_official_infer_smoke.py \
  --config starVLA/config/training/starvla_fastwam_libero_official_infer.yaml \
  --mode latent \
  --device cuda \
  --dtype bfloat16 \
  --num-inference-steps 1 \
  --seed 0
```

確認する出力:

```text
actions_shape: [1, 32, 7]
finite: true
min
max
mean
```

なぜraw imageではなくlatentから始めるのか:

Wan2.2 VAE/text encoderをまだ使わず、FastWAM本体だけを切り分けて確認するため。

ここで落ちる場合、原因は主に以下に絞れる。

```text
- MoT action inference
- first_frame_causal attention mask
- ActionDiT denoise loop
- proprio/state context追加
- scheduler step
```

raw image経路まで含めてしまうと、VAE、text encoder、画像前処理、tokenizerの問題も混ざり、原因の切り分けが難しくなる。

### Step 3. 診断結果に基づいて変換ルールを決める

Step 1の結果で分岐する。

#### A. missing/unexpectedがない場合

公式checkpointの構造がStarVLA側とそのまま対応している。

この場合、変換処理は主にStarVLA形式への再保存になる。

保存先案:

```text
checkpoints/fastwam_starvla/
└── libero_uncond_2cam224_starvla.pt
```

保存内容案:

```python
{
    "framework": model.state_dict(),
    "source_format": "official_fastwam",
    "source_checkpoint": "checkpoints/fastwam_release/libero_uncond_2cam224.pt",
    "step": 21700,
    "action_dim": 7,
    "proprio_dim": 8,
    "action_horizon": 32,
    "conversion_report": {
        "missing_keys": [],
        "unexpected_keys": [],
    },
}
```

#### B. missing/unexpectedがある場合

StarVLA側と公式checkpointのmodule名、保存対象、またはshapeに差分がある。

この場合は、差分を見て変換処理を実装する。

必要になる可能性がある処理:

```text
- key rename
- 公式側にだけあるkeyの除外
- StarVLA側にだけあるkeyの初期化維持
- proprio_encoderの別保存からframework state_dictへの統合
- dtype変換
- metadata保存
```

この段階で初めて、公式 `helpers/state_dict_converters.py` 相当の処理をStarVLA側へ最小実装するか判断する。

### Step 4. StarVLA形式checkpointを保存する

Step 3で決めた変換規則に基づいて、変換済みcheckpointを作る。

目的:

```text
- 推論時に毎回公式形式を解釈しなくてよくする
- StarVLA側の読み込みを単純化する
- 公式checkpoint由来であることをmetadataに残す
- 公式実装との比較時に、どの変換を使ったか追跡できるようにする
```

なぜ変換済みcheckpointを作るのか:

公式形式を毎回直接読むだけでも推論は可能だが、長期的には不安定。
StarVLA形式に保存しておけば、以後のABCI推論、比較、実験、再学習の入口が統一される。

### Step 5. 変換済みcheckpointでStarVLA推論を行う

変換済みcheckpointを使って、再度StarVLA側で推論する。

確認順:

```text
1. load
2. latent
3. raw
```

raw推論の確認コマンド:

```bash
python examples/simBenchmarks/LIBERO/eval_files/fastwam_official_infer_smoke.py \
  --config starVLA/config/training/starvla_fastwam_libero_official_infer.yaml \
  --mode raw \
  --device cuda \
  --dtype bfloat16 \
  --num-inference-steps 1 \
  --prompt "" \
  --state "0,0,0,0,0,0,0,0"
```

`--image` を省略した場合は灰色ダミー画像を使う。
これは意味のあるActionを見るためではなく、Wan2.2 VAE/text encoderを含む経路が最後まで動くかを確認するため。

### Step 6. 公式実装と同一入力で比較する

StarVLA側の推論が通った後、公式FastWAM実装でも同じ入力を使ってActionを出す。

一致比較に必要な条件:

```text
- 同じcheckpoint
- 同じ画像
- 同じpromptまたは同じprecomputed context
- 同じproprio/state
- 同じaction_horizon
- 同じnum_inference_steps
- 同じseed
- 同じscheduler設定
- 同じdtype/device
- 同じ前処理
```

最初はraw imageではなく、precomputed latent/context/stateで比較する。

理由:

raw imageから比較すると、VAE、tokenizer、text encoder、画像resizeの微差が混ざる。
precomputed latent/context/stateで比較すれば、FastWAM本体の差分だけを見られる。

precomputed比較で一致した後に、raw image比較へ進む。

## 成功判定

### 接続成功

```text
- 公式checkpointをロードできる
- missing/unexpected keyがゼロ、または理由付きで許容できる
- latent modeで actions_shape=[1,32,7]
- finite=true
```

### StarVLA推論成功

```text
- raw modeで画像、言語、stateからActionが出る
- NaN/Infがない
- Action shapeが期待通り
```

### 公式再現成功

```text
- 同一precomputed latent/context/stateで公式実装とStarVLA実装のActionが一致する
- raw image経路でも許容誤差内に入る
```

## 失敗した場合の見方

### loadで失敗

見るもの:

```text
missing_keys_count
unexpected_keys_count
shape mismatch traceback
```

対応:

```text
- 変換規則を追加する
- configのaction_dim/proprio_dim/text_dim/num_layers等を修正する
- 公式checkpointに含まれないStarVLA側moduleを初期化維持するか判断する
```

### latentで失敗

見るもの:

```text
MoT traceback
attention mask shape
context/context_mask shape
proprio token追加後のcontext length
ActionDiT output shape
```

対応:

```text
- first_frame_causal maskを修正
- predict_actionの入力shapeを修正
- proprio/stateの扱いを公式実装にさらに寄せる
```

### rawで失敗

見るもの:

```text
Wan2.2 VAE load
Wan2.2 text_encoder load
tokenizer load
image preprocessing
latent shape
context shape
```

対応:

```text
- encoder pathを修正する
- 画像サイズを公式checkpointに合わせる
- text encoder dtype/deviceを調整する
```

## 次にやること

1. ABCIで `qsub docs/abci_fastwam_official_infer_smoke.pbs` を実行する。
2. `abci_fastwam_official_infer.log` を確認する。
3. `load` の結果から変換規則が必要か判断する。
4. 必要なら変換スクリプトを実装する。
5. StarVLA形式checkpointを保存する。
6. 変換済みcheckpointでlatent推論、raw推論を確認する。
7. 公式実装とのprecomputed入力比較スクリプトを作る。
8. 公式実装とStarVLA実装のAction一致を確認する。
