# FastWAM MoT Module Design Explanation

## Question

MoT は今後 `FastWAM`、`FastWAMJoint`、`FastWAMIDM` などで共有される可能性が高い。
その場合、`FastWAM.py` に吸収したままでよいのか、または `model/modules` 配下へ分離すべきかを整理する。

あわせて、以下の関係も整理する。

- `FastWAM.py`
- `FastWAM_WanVideoDiT.py`
- `FastWAM_ActionDiT.py`
- `MoT`

## Current State

現在の配置は以下。

```text
starVLA/model/framework/WM4A/FastWAM.py
  - FastWAMFramework
  - WanContinuousFlowMatchScheduler
  - MoT

starVLA/model/modules/world_model/FastWAM_WanVideoDiT.py
  - WanVideoDiT

starVLA/model/modules/action_model/FastWAM_ActionDiT.py
  - ActionDiT
  - FastWAMActionDiTHead
```

`MoT` は一度 `FastWAM.py` に吸収した。
理由は、`framework/WM4A/FastWAM_MoT.py` として独立させると、StarVLA の既存設計から見て「framework 配下に中核 network module がある」状態になっていたため。

ただし、MoT を今後の FastWAM variant でも共有するなら、`FastWAM.py` 内に置くより `model/modules` 配下へ分離する方が自然である。

## Recommendation

共有前提なら、MoT は `framework` ではなく `model/modules` 配下へ置くべき。

最も良い候補は以下。

```text
starVLA/model/modules/mixer/FastWAM_MoT.py
```

理由:

- MoT は world model 単体ではない。
- MoT は action head 単体でもない。
- MoT は video expert と action expert を層ごとに混ぜる module である。
- `framework` は組み立て役に戻せる。
- 将来 `FastWAMJoint` / `FastWAMIDM` などから同じ MoT を import できる。

代替候補:

```text
starVLA/model/modules/fusion_model/FastWAM_MoT.py
```

これも意味は近い。
ただし `fusion_model` という名前は少し広く、projector や cross-modal fusion module も入れたくなる可能性がある。
MoT は「複数 expert の token を attention で混ぜる」ものなので、`mixer` の方が狭く、役割が明確。

既存フォルダだけで済ませる候補:

```text
starVLA/model/modules/world_model/FastWAM_MoT.py
```

これは新規フォルダを作らない利点がある。
しかし MoT は action expert も直接扱うため、`world_model` 配下に置くと意味が少しずれる。
短期的には使えるが、共有前提では `mixer` の方がよい。

## Existing Folder Only Recommendation

新しいフォルダを作らず、既存フォルダのどこかに置くなら、最も現実的なのは以下。

```text
starVLA/model/modules/world_model/FastWAM_MoT.py
```

これは完璧な配置ではない。
MoT は action expert も扱うため、純粋な world model ではない。
それでも既存フォルダの中では `world_model` が一番ましである。

理由:

- FastWAM 全体は「world model for action」として扱える。
- MoT の action-only inference でも、まず video token / video K/V cache を作る。
- `MoT` は `FastWAM_WanVideoDiT.py` の `flash_attention`、`modulate`、`rope_apply` と強く結びついている。
- `FastWAM_WanVideoDiT.py` と同じ `world_model` 配下に置くと、FastWAM の video-side 実装を追いやすい。
- `action_model` 配下へ置くより、video/action 両方を扱う違和感が小さい。

既存フォルダ縛りでの候補順位:

```text
1. starVLA/model/modules/world_model/FastWAM_MoT.py
2. starVLA/model/modules/action_model/FastWAM_MoT.py
3. starVLA/model/modules/projector/FastWAM_MoT.py
4. starVLA/model/modules/dino_model/FastWAM_MoT.py
```

`action_model` は次点。
MoT は action token を更新し、`ActionDiT` と強く関係するため置けなくはない。
しかし video expert の K/V cache を作り、video/action mixed attention を管理するため、action head 単体の配下に置くと責務が重く見える。

`projector` は非推奨。
projector は通常、ある hidden state を別の次元や別の表現へ変換する薄い層である。
MoT は単なる変換ではなく、複数 expert の Transformer block を層ごとに混ぜる本体計算なので、projector に置くと意味がずれる。

`dino_model` は非推奨。
DINO は vision feature extractor 系の置き場であり、MoT の video/action token mixer とは役割が違う。

したがって、新規フォルダを作らない制約を優先するなら、次に実装する配置は以下がよい。

```text
starVLA/model/modules/world_model/FastWAM_MoT.py
```

この場合、`FastWAM.py` は以下の import に戻す。

```python
from starVLA.model.modules.world_model.FastWAM_MoT import MoT
```

この配置は「理想的な分類」ではなく「既存フォルダだけでStarVLA設計から大きく外れない妥協案」である。
将来 FastWAM 系 module が増えた段階で、必要なら `modules/mixer` や `modules/fastwam` のような専用整理を再検討する。

## Why Not Framework

StarVLA の `framework/WM4A` は、基本的に以下を担当している。

- world model を作る。
- action model を作る。
- dataloader の examples を受ける。
- `forward()` で loss を返す。
- `predict_action()` で action を返す。
- framework registry に登録する。

つまり framework は「部品を組み立てて学習・推論の入口を作る場所」である。

一方、MoT は中核 network module である。
`MoT.forward()` は実際に attention を計算し、video token と action token を更新する。
これは framework の薄い統合処理ではなく、モデル本体の処理である。

そのため、共有前提なら `framework` から外して `model/modules/mixer` へ置く方が設計上は自然。

## Relationship Between Files

### FastWAM.py

`FastWAM.py` は StarVLA から見た FastWAM の入口。

主な役割:

- config を読む。
- `WanVideoDiT` を作る。
- `ActionDiT` を作る。
- `MoT` を作る。
- scheduler を作る。
- training loss を計算する。
- `predict_action()` を提供する。

中学生向けに言うと、`FastWAM.py` は「監督」。
動画担当、行動担当、混ぜる係を集めて、学習や推論の流れを決める。

### FastWAM_WanVideoDiT.py

`FastWAM_WanVideoDiT.py` は video expert。

主な役割:

- video latent を patch token に変換する。
- video 用の position embedding / RoPE を作る。
- video timestep embedding を作る。
- video Transformer block を持つ。
- 最後に video noise/velocity を予測する。

FastWAM では、動画そのものを pixel で直接扱うのではなく、VAE で圧縮された latent を扱う。
`WanVideoDiT` はその latent 空間で「次にどんな動画になるか」を考える担当。

中学生向けに言うと、`WanVideoDiT` は「動画担当の生徒」。
画像や動画のメモを読んで、場面がどう動くかを考える。

### FastWAM_ActionDiT.py

`FastWAM_ActionDiT.py` は action expert。

主な役割:

- action を token に変換する。
- action timestep embedding を作る。
- action Transformer block を持つ。
- 最後に action noise/velocity を予測する。
- StarVLA 既存 action head 互換の wrapper も持つ。

通常の action head は、world model や VLM の出力を受け取って action を直接予測する。
しかし FastWAM の `ActionDiT` は、MoT を通じて video expert と層ごとに混ざる前提を持つ。

中学生向けに言うと、`ActionDiT` は「行動担当の生徒」。
ロボットが次にどう動くべきかを、ノイズから少しずつ直しながら考える。

### MoT

MoT は Mixture of Transformers。
FastWAM では video expert と action expert を層ごとに接続する。

主な役割:

- video token と action token の query/key/value を作る。
- それらを連結して mixed attention を行う。
- attention mask で、どの token がどの token を見てよいかを制御する。
- video branch と action branch に結果を戻す。
- action-only 推論では video K/V cache を作り、action denoise 中に再利用する。

中学生向けに言うと、MoT は「班活動の時間」。
動画担当と行動担当が、それぞれのノートを見せ合いながら考えを更新する。

普通の構成:

```text
動画担当が考える
  -> 結果を行動担当に渡す
  -> 行動担当が答える
```

MoT の構成:

```text
動画担当が少し考える
行動担当も少し考える
お互いのノートを見る
また少し考える
これを何層も繰り返す
```

この違いにより、action は最後に video の結果を受け取るだけではなく、途中の reasoning 段階から video token を参照できる。

## Why ActionDiT And VideoDiT Are Hard To Merge

`ActionDiT` と `WanVideoDiT` を1ファイル、または1つの class に統合することは技術的には可能。
しかし、現時点では推奨しない。

理由は、それぞれが扱うデータの形と責務が違うため。

### VideoDiT side

`WanVideoDiT` が扱うもの:

```text
[B, C, T, H, W]
```

これは video latent。
時間、縦、横、latent channel を持つ。
patchify / unpatchify が必要。
video frame ごとの attention mask が必要。

### ActionDiT side

`ActionDiT` が扱うもの:

```text
[B, T_action, action_dim]
```

これは robot action sequence。
空間方向の H/W はない。
action horizon に沿った 1D sequence として扱う。

### Shared Part

両者に似ている部分はある。

- Transformer block
- self-attention
- cross-attention
- timestep modulation
- RoPE
- MLP
- gradient checkpointing

ただし、似ているからといってすぐ統合すると危険。
video 側は 3D latent patch、action 側は 1D action token なので、入力と出力の意味が違う。

統合してよい可能性があるのは、以下のような小さい共通部品。

- `RMSNorm`
- `flash_attention`
- `modulate`
- `rope_apply`
- `sinusoidal_embedding_1d`
- `DiTBlock` の共通化

逆に、今すぐ統合しない方がよい部分:

- `WanVideoDiT` class 全体
- `ActionDiT` class 全体
- video patchify / unpatchify
- action token embedding / action head
- video-specific attention mask
- action-specific scheduler wrapper

## Recommended Next Placement

共有前提で次に直すなら、以下がよい。

```text
starVLA/model/modules/mixer/
  ├── __init__.py
  └── FastWAM_MoT.py
```

そして `FastWAM.py` は以下のように import する。

```python
from starVLA.model.modules.mixer.FastWAM_MoT import MoT
```

この場合、`FastWAM.py` は再び framework entry として軽くなる。
MoT は `framework` ではなく module として扱える。
将来 `FastWAMJoint.py` や `FastWAMIDM.py` を StarVLA に追加した場合も、同じ MoT を共有できる。

## Migration Risk

MoT を `FastWAM.py` から `model/modules/mixer/FastWAM_MoT.py` へ出すリスクは低い。

理由:

- MoT は registry 登録対象ではない。
- config からファイル名で指定されていない。
- 現在の参照元は `FastWAM.py` のみ。
- class 本体を移動し、import を差し替えるだけで済む。
- 既存 smoke test で `build_framework()`、training loss、action inference を確認できる。

注意点:

- `FastWAM_MoT.py` 内で `flash_attention`、`modulate`、`rope_apply` を import する必要がある。
- `__init__.py` を作る場合、既存 import discovery に副作用がないよう空に近い内容にする。
- 公式 FastWAM の `mot.py` との差分追跡のため、MoT class の内部ロジックは移動時に変えない。

## Final Position

現在の判断:

- MoT を共有する可能性が低いなら、`FastWAM.py` 内でも許容できる。
- MoT を共有する可能性が高いなら、`model/modules/mixer/FastWAM_MoT.py` が最も自然。
- `world_model` 配下は短期的には可能だが、action expert も扱うため意味がずれる。
- `action_model` 配下も同様に、video expert を扱うため意味がずれる。
- `framework` 配下に独立ファイルとして置くのは、StarVLA の既存設計から見ると避けたい。

次に実装するなら、`model/modules/mixer/FastWAM_MoT.py` へ MoT を移し、`FastWAM.py` は import して使う形に戻す。
