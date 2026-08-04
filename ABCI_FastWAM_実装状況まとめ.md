# ABCI上でのFast-WAM／StarVLA実装状況

更新日: 2026-08-02

## 1. 目的

現在の目的は、**StarVLA上に実装したFast-WAMへ公式Fast-WAMチェックポイントを読み込み、実画像と言語指示からActionを推論できる状態にすること**です。

現段階では、本格的な再学習よりも、次の推論経路を成立させることを優先します。

```text
実画像・言語指示
    ↓
Wan2.2 VAE／Text Encoder
    ↓
Video latent／Text embedding
    ↓
Fast-WAM Video Expert・MoT・Action Expert
    ↓
Action Chunk
```

---

## 2. やったこと

### 2.1 ABCI接続・ファイル配置

- ABCI用SSH公開鍵を登録
- ABCIへSSHログイン
- GitHubのSSH認証を確認
- StarVLAリポジトリをABCIへclone
- Wan2.2関連のエンコーダーファイル約14.20 GBをABCIへ転送
- 転送元と転送先の差分を確認

### 2.2 GPU計算環境

- `rt_HG`の対話ジョブを取得
- H200 GPUが認識されることを確認
- ログインノードと計算ノードの役割を整理

```text
ログインノード:
- Git操作
- ファイル確認
- 軽いコマンド

GPU計算ノード:
- PyTorch実行
- モデルロード
- Smoke Test
- 推論
```

### 2.3 Python環境

- Python 3.12.9の仮想環境を作成
- StarVLA本体をeditable install
- `python -m pip check`を実行
- `No broken requirements found.`を確認

使用した仮想環境:

```text
.venv-py312
```

### 2.4 PyTorch関連

外部のPyTorch配布サーバーへ接続できず、以下のネットワークエラーが一度発生しました。

```text
Failed to connect to download-r2.pytorch.org
[Errno 101] Network is unreachable
```

その後、Smoke Testを実行できる環境には到達しています。

ただし、**最終的にPyTorchをどの経路で導入したかは、現在の記録だけでは確定できていません**。

### 2.5 Fast-WAM Smoke Test

以下のテストを確認しました。

```text
test_scheduler_shapes
test_build_framework_uses_tiny_fastwam_config
test_tiny_experts_and_mot_forward
test_forward_returns_action_loss_for_precomputed_latents
test_compute_loss_routes_vla_batch
test_raw_examples_route_through_encoder_adapter
test_raw_examples_require_encoder_loading_or_precomputed_latents
test_encoder_smoke_config_is_wired_for_fastwam
test_optional_wan2_encoder_load_smoke
test_predict_action_returns_normalized_actions_for_precomputed_inputs
test_predict_action_requires_first_frame_causal_mode
```

最終的にテスト結果は`OK`になりました。

### 2.6 Smoke Testで確認できた内容

#### 小型Fast-WAM構成の生成

本番のWan2.2-5Bを直接使うのではなく、層数やhidden sizeを縮小したテスト用構成を生成できました。

#### Video Expert・Action Expert・MoTのforward

小型構成で以下の経路が動作しました。

```text
Video側の表現
    ＋
Action側の表現
    ↓
MoT
    ↓
Action側の出力
```

#### Action Lossの計算

事前計算済みlatentを入力した場合に、スカラーの`action_loss`が返ることを確認しました。

#### backwardと勾配

```python
out["action_loss"].backward()
```

を実行し、モデル内のパラメータへ勾配が付くことを確認しました。

これは本格的な学習を完了したという意味ではなく、**計算グラフがつながっていることの確認**です。

#### StarVLAバッチのルーティング

StarVLA形式のVLAバッチが、Fast-WAMの損失計算へ渡されることを確認しました。

#### 生画像入力とencoder adapter

生画像・言語入力はencoder adapterへ送られる設計になっていることを確認しました。

また、encoderが未ロードで、事前計算済みlatentも存在しない場合には、誤って処理を続行せずエラーにすることを確認しました。

#### `predict_action()`の出力

事前計算済み入力から、`normalized_actions`を返せることを確認しました。

#### Fast-WAM baseの因果条件

Fast-WAM baseでは、推論時に未来動画を生成せず、最初の観測フレームを基準にActionを生成します。

Smoke Testで、このfirst-frame causal modeが要求されることを確認しました。

---

## 3. 現在できること

現時点で、次のコード経路は確認できています。

- StarVLA内でFast-WAM Frameworkを構築する
- 小型Video Expertを実行する
- 小型Action Expertを実行する
- MoTを通してforwardする
- 事前計算済みlatentからAction Lossを計算する
- backwardで勾配を計算する
- StarVLA形式のバッチをFast-WAMへ渡す
- 生画像入力をencoder adapterへ振り分ける
- 事前計算済み入力から`predict_action()`を実行する
- 正規化されたAction Chunkを返す
- first-frame causal modeの条件を検査する

現在地を一文で表すと、次のとおりです。

> ABCIのH200環境上で、StarVLAへ追加したFast-WAMの小型実装について、Video Expert、Action Expert、MoT、Action Loss、勾配、入力ルーティング、推論インターフェースまでのSmoke Testを通過した段階です。

---

## 4. これからやること

### 4.1 公式Fast-WAMチェックポイントの構造確認

最初に、公式チェックポイント内のstate dict key、Tensor shape、保存対象モジュールを確認します。

確認対象の例:

```text
video_expert
action_expert
mot
proprio_encoder
vae
text_encoder
```

実際のキー名はチェックポイントを開くまで確定しません。

2026-08-02 追記:

公式Fast-WAM実装 `/home/anpan/WM/FastWAM/src/fastwam/models/wan22/fastwam.py` を確認したところ、公式 `save_checkpoint()` / `load_checkpoint()` の主経路は以下でした。

```python
payload = {
    "mot": self.mot.state_dict(),
    "step": step,
    "torch_dtype": str(self.torch_dtype),
}
```

読込時は `payload["mot"]` を `self.mot.load_state_dict(..., strict=False)` で読み込む。
古い Wan-only 形式では `payload["dit"]` を video expert だけへ読み込む分岐もある。

このため、StarVLA側にも公式互換の `load_checkpoint()` を追加する方針が妥当。
実装方針:

- `framework.checkpoint.path` が指定されていれば `FastWAMFramework.__init__()` 内で自動ロードする。
- `payload["mot"]` があれば MoT 全体へ読み込む。
- `payload["dit"]` だけの場合は legacy 扱いとして video expert のみに読み込む。
- `missing_keys` / `unexpected_keys` / `step` を `loaded_checkpoint` に記録する。
- 実checkpointのshape/keyが一致するかは、ABCI上で公式checkpoint実物を配置してから確認する。

### 4.2 チェックポイント変換・マッピング

公式Fast-WAM側のキーを、StarVLA側のモジュール名へ変換する処理を作成します。

必要な処理:

- Wrapper prefixの除去
- モジュール名の変換
- Tensor shapeの厳密チェック
- Missing keyの記録
- Unexpected keyの記録
- 変換したキー一覧の保存
- 読み込めなかった重みの明示

### 4.3 公式重みのロード

変換後の重みを、少なくとも以下へ読み込みます。

```text
Video Expert
Action Expert
MoT
Proprio関連モジュール
```

ロード後は、ランダム初期値のまま残っているパラメータがないか確認します。

### 4.4 Wan2.2 encoderの実ロード

転送済みのWan2.2関連ファイルを使い、以下を実際にロードします。

- Video VAE
- Text Encoder
- 必要なTokenizer
- Fast-WAMが使用するVideo DiT関連重み

現在は設定の接続やoptional smokeの確認段階であり、**本番サイズのencoderを使用した完全な入力処理は未確認**です。

### 4.5 実画像からlatentを生成

実画像を入力し、以下を確認します。

- 画像のshape
- カメラ順
- resize
- 値域
- VAE latentのshape
- dtype
- device
- NaN／Infの有無

### 4.6 言語指示からembeddingを生成

言語指示をText Encoderへ入力し、以下を確認します。

- Prompt template
- Tokenizer
- 最大長
- Text embeddingのshape
- Attention mask
- dtype
- NaN／Infの有無

### 4.7 End-to-End Action推論

最終的に、次の経路を実行します。

```text
実画像
＋
言語指示
＋
必要ならstate
    ↓
VAE／Text Encoder
    ↓
Fast-WAM
    ↓
normalized_actions
    ↓
必要なら逆正規化
    ↓
実行可能なAction Chunk
```

### 4.8 実データでの確認

最初は1サンプルで確認し、その後にデータセットを使った評価へ進みます。

候補:

- LIBERO
- StarVLA側ですでに扱えるLeRobot形式データ
- 公式Fast-WAMと同じ前処理条件を再現できるデータ

最初からclosed-loop成功率を測るのではなく、まず以下を確認します。

```text
入力が正しい
→ latentが正しい
→ checkpointが正しくロードされる
→ Actionが有限値で出る
→ Action shapeが正しい
```

---

## 5. 検討中のこと

### 5.1 `Wan2.py`をどこまで再利用するか

検討点:

- 既存`Wan2.py`をencoder wrapperとして使うか
- Fast-WAM専用Video DiTを別実装として使うか
- Diffusers版Wanと公式Fast-WAM版Wanのstate dictが互換か
- 既存のWan2実装へ公式Fast-WAM重みを直接入れられるか

現時点の判断:

> VAEやText Encoderなど周辺部は再利用できる可能性があります。一方、Video DiT本体はMoTとの結合方法やblock構造が異なる可能性があるため、公式Fast-WAM実装との対応確認が必要です。

### 5.2 公式チェックポイントを直接読むか、変換して保存するか

候補は2つあります。

#### 毎回ロード時に変換

長所:

- 元の公式チェックポイントを保持できる
- 変換規則をコードで追跡できる

短所:

- 毎回変換が必要
- 読み込み処理が複雑になる

#### 一度変換してStarVLA形式で保存

長所:

- 2回目以降のロードが簡単
- 推論スクリプトを単純化できる

短所:

- 変換済みファイルのバージョン管理が必要
- 変換元と変換コードの記録が必要

現時点では、**最初はロード時変換で検証し、成功後に変換済みチェックポイントを保存する方法**が安全だと考えています。

### 5.3 事前計算済みlatentと生画像入力の使い分け

#### 事前計算済みlatent

用途:

- Action側の実装確認
- MoTの確認
- checkpointロード確認
- 問題箇所の切り分け

#### 生画像入力

用途:

- 実運用に近いEnd-to-End推論
- VAEとText Encoderを含む確認
- データ前処理の確認

当面は、事前計算済みlatentでcheckpointロードを確認した後、生画像入力へ進む方針です。

### 5.4 使用する最初のデータセット

候補としてLIBEROがありますが、最初の確認では大規模評価よりも、1サンプルまたは少数サンプルでの決定論的確認が適しています。

確認する必要がある項目:

- カメラ数
- カメラ順
- 観測フレーム数
- Action Horizon
- Action次元
- state次元
- 正規化統計
- Prompt template

### 5.5 学習を行うか

現在の目的は推論です。

そのため、当面は次を行いません。

- 本格的な再学習
- 大規模Fine-tuning
- 動画・Actionの共同学習
- 公式ベンチマークの完全再現

ただし、ロードした重みが正しく機能しているか確認するため、必要に応じて以下は行う可能性があります。

- 1 batchのforward
- Loss計算
- backward確認
- 1サンプルへのoverfit確認

### 5.6 embeddingの入力位置

確認が必要な点:

- Text embeddingがどのモジュールへ渡るか
- Video tokenとAction tokenの両方にcross-attentionされるか
- 現在のコードで事前計算済みembeddingを直接受け取れるか
- 生の文字列をencoder adapterでembeddingへ変換できるか
- Attention maskが正しく渡っているか

これは、実装ファイルとforward経路を追跡して確定する必要があります。

---

## 6. 現時点で未確定のこと

以下は、まだ断定できません。

- 公式Fast-WAMチェックポイントの正確なキー構造
- 公式Fast-WAMと現在のStarVLA側Video DiTの完全互換性
- 変換だけで公式重みを100%読み込めるか
- 本番Wan2.2-5Bをロードした際のGPUメモリ使用量
- 実画像からのEnd-to-End推論がそのまま成功するか
- Text EncoderとVAEの現在のパス指定が完全に正しいか
- 公式実装と同じAction正規化・逆正規化が再現できているか
- PyTorchを最終的にどの方法で導入したか

---

## 7. 直近の作業順序

推奨する次の順序は以下です。

### Step 1

公式READMEに従い、Hugging Faceの公開checkpointとdataset statsを配置する。

StarVLAリポジトリでは、READMEと同じ相対パスを維持して以下へ置く。

```text
checkpoints/fastwam_release/
├── libero_uncond_2cam224.pt
├── libero_uncond_2cam224_dataset_stats.json
├── robotwin_uncond_3cam_384.pt
└── robotwin_uncond_3cam_384_dataset_stats.json
```

`checkpoints/` は大容量の外部成果物置き場なので、`.gitignore` に追加してgit管理から外す。

READMEの `huggingface-cli download ...` は、`huggingface-cli` がPATHにある環境ではそのまま使える。今回のStarVLAローカル環境ではCLIが見つからなかったため、既に入っている `huggingface_hub` Python APIで同じファイルを取得する。

2026-08-02時点のローカル配置結果:

```text
checkpoints/fastwam_release/libero_uncond_2cam224.pt                  12,041,735,140 bytes
checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json      40,939 bytes
checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt              12,041,813,092 bytes
checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json   88,715 bytes
```

`checkpoints/fastwam_release/` 全体は約23GB。`.gitignore` に `checkpoints/` を追加済みなので、これらの大容量ファイルはgitには入らない。

公式checkpointを `torch.load(..., mmap=True)` で軽く確認した結果:

```text
libero_uncond_2cam224.pt
- top-level keys: mot, proprio_encoder, step, torch_dtype
- step: 21700
- torch_dtype: torch.bfloat16
- mot state_dict keys: 1649
- action head: mixtures.action.head.weight = [7, 1024]
- proprio_encoder: weight = [4096, 8], bias = [4096]

robotwin_uncond_3cam_384.pt
- top-level keys: mot, proprio_encoder, step, torch_dtype
- step: 29355
- torch_dtype: torch.bfloat16
- mot state_dict keys: 1649
- action head: mixtures.action.head.weight = [14, 1024]
- proprio_encoder: weight = [4096, 14], bias = [4096]
```

dataset statsの要点:

```text
libero_uncond_2cam224_dataset_stats.json
- action_dim: 7
- action_horizon: 32
- state_dim: 8
- num_episodes: 1712
- num_transition: 277713

robotwin_uncond_3cam_384_dataset_stats.json
- action_dim: 14
- action_horizon: 32
- state_dim: 14
- num_episodes: 27500
- num_transition: 6075103
```

この確認により、公式checkpointはStarVLA側に追加済みの `payload["mot"]` ロード方針と合っている。一方で、公式checkpointは `proprio_encoder` も持つため、公式推論を完全に再現するには、StarVLA側にもproprio/stateをcontextへ追加する経路が必要だった。

2026-08-04更新:

StarVLA FastWAMへ公式 `proprio_encoder` 相当の経路を実装した。

- `framework.action_model.proprio_dim` を追加した。通常は `None` で無効。
- 公式checkpointに `proprio_encoder` が入っている場合は、`proprio_encoder.weight` のshapeから `proprio_dim` を自動判定し、StarVLA側に `nn.Linear(proprio_dim, text_dim)` を作る。
- `payload["proprio_encoder"]` をstrictにロードする。
- training時は `sample["proprio"]` または `sample["state"]` を受け取り、先頭時刻のstate/proprioをtext context末尾へ1トークン追加する。
- `predict_action()` 時も `proprio` または `state` を受け取り、同じくcontext末尾へ1トークン追加する。
- `proprio_encoder` が有効なのに `proprio/state` が無い場合は明確に失敗させる。これは公式checkpointの条件を欠いたまま意味の薄い推論を実行しないため。
- `proprio_encoder` が無効なのに `proprio/state` が渡された場合も失敗させる。configの不一致を早く検出するため。

小型構成での確認:

```text
.venv-py311/bin/python -m unittest tests.test_fastwam_smoke
Ran 17 tests in 1.375s
OK (skipped=1)
```

追加確認済み:

- 公式形式 `payload["mot"]` のロード
- 公式形式 `payload["proprio_encoder"]` のロード
- checkpoint shapeからの `proprio_dim` 自動構築
- training inputでcontext長が `+1` されること
- `predict_action()` inputでcontext長が `+1` されること

ABCIで公式checkpointを確認するため、既存 `starVLA/config/training/` 配下に以下のconfigを追加した。

```text
starVLA/config/training/starvla_fastwam_libero_official_infer.yaml
- checkpoint: ./checkpoints/fastwam_release/libero_uncond_2cam224.pt
- action_dim: 7
- proprio_dim: 8
- action_horizon: 32
- image size: 224 x 224
- video_attention_mask_mode: first_frame_causal

starVLA/config/training/starvla_fastwam_robotwin_official_infer.yaml
- checkpoint: ./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
- action_dim: 14
- proprio_dim: 14
- action_horizon: 32
- image size: 384 x 384
- video_attention_mask_mode: first_frame_causal
```

まずABCIではLIBERO側configを使うのがよい。RoboTwinはaction/stateが14次元で、評価環境の準備もLIBEROより重くなりやすいため。

ABCI実行入口として以下を追加した。

```text
examples/simBenchmarks/LIBERO/eval_files/fastwam_official_infer_smoke.py
```

このスクリプトは3段階で使う。

```text
--mode load
- StarVLA FastWAM full-size modelを構築する
- cuda + bfloat16へ移す
- 公式checkpointの mot/proprio_encoder をロードする
- missing/unexpected key数を表示する
- Wan2.2 VAE/text encoderは読まない

--mode latent
- loadに加えて、ランダムな first_frame_latents/context とゼロstateで predict_action() を実行する
- Action shape、finite、min/max/meanを確認する
- Wan2.2 VAE/text encoderは読まない

--mode raw
- loadに加えて、Wan2.2 VAE/text encoderを読む
- 実画像または灰色のダミー画像、言語prompt、stateから predict_action() を実行する
```

ABCI用PBSテンプレート:

```text
docs/abci_fastwam_official_infer_smoke.pbs
```

最初にABCIで実行する順序:

```bash
qsub docs/abci_fastwam_official_infer_smoke.pbs
```

このPBSはまず以下を実行する。

```bash
python examples/simBenchmarks/LIBERO/eval_files/fastwam_official_infer_smoke.py \
  --config starVLA/config/training/starvla_fastwam_libero_official_infer.yaml \
  --mode load \
  --device cuda \
  --dtype bfloat16

python examples/simBenchmarks/LIBERO/eval_files/fastwam_official_infer_smoke.py \
  --config starVLA/config/training/starvla_fastwam_libero_official_infer.yaml \
  --mode latent \
  --device cuda \
  --dtype bfloat16 \
  --num-inference-steps 1 \
  --seed 0
```

`load` が通らない場合は、StarVLA実装と公式checkpointのkey/shape対応にまだ問題がある。`load` が通り `latent` が落ちる場合は、MoT action inference、attention mask、ActionDiT post処理、またはproprio context追加のどこかに問題が残っている。

`latent` まで通った後に初めて、Wan2.2 encoder込みのraw推論へ進む。

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

raw modeで `--image` を省略すると、224x224の灰色ダミー画像を使う。これは意味のある行動を期待するものではなく、VAE/text encoderを含めた経路確認用。

### Step 2

公式Fast-WAMチェックポイントの中身を表示する。

```text
state dict key
Tensor shape
dtype
保存モジュール
```

### Step 3

StarVLA側Fast-WAMモデルの`state_dict()`を表示し、対応表を作る。

### Step 4

変換規則を実装し、以下のレポートを出す。

```text
loaded
remapped
missing
unexpected
shape mismatch
```

### Step 5

事前計算済みlatentを使い、公式重みで`predict_action()`を実行する。

### Step 6

Wan2.2 VAEとText Encoderを実ロードする。

### Step 7

実画像・言語指示からEnd-to-EndでActionを出す。

### Step 8

少数の実データまたはLIBEROサンプルで出力を検証する。

---

## 8. 完了条件

今回の推論実装について、最低限の完了条件は次のとおりです。

```text
[x] 公式Fast-WAM checkpointのキー構造を確認
[ ] StarVLA側とのキー対応表を作成
[x] checkpointロード入口を実装
[x] proprio_encoderロード経路を実装
[ ] shape mismatchを解消
[ ] 公式重みをロード
[ ] 事前計算済みlatentからActionを生成
[ ] Wan2.2 VAEを実ロード
[ ] Wan2.2 Text Encoderを実ロード
[ ] 実画像と言語指示からActionを生成
[ ] Actionのshape・値域・有限値を確認
[ ] 正規化と逆正規化を確認
```

補足: 公式checkpointは12GB級のため、ローカルの約7.4GiB RAM環境でフルモデルを組んでロードするのは危険。キー構造確認はmmapで可能だったが、公式サイズの実ロードと推論はABCIのH200ノードで行うのが妥当。

---

## 9. ミーティング用要約

ABCIへのSSH接続、GitHub認証、StarVLAのclone、Wan2.2関連ファイルの転送、H200計算ノードの取得、Python 3.12環境の構築まで完了しました。

Fast-WAMについては、小型構成を使ったSmoke Testを実施し、Video Expert、Action Expert、MoTのforward、Action Loss、backward、StarVLAバッチのルーティング、encoder adapter、`predict_action()`による正規化Action出力まで確認できています。

現在は、**Fast-WAMのコード経路に加えて、公式checkpointの `mot` と `proprio_encoder` をStarVLA側へ読む入口が成立した段階**です。

次は、ABCI上で公式サイズのStarVLA FastWAMを構築し、`libero_uncond_2cam224.pt` を実ロードします。その後、事前計算済みlatent/context/proprioで `predict_action()` を確認し、Wan2.2のVAEとText Encoderを実ロードして、実画像・言語指示・proprio/stateからActionを出すEnd-to-End推論へ進みます。

検討中の主な項目は、既存`Wan2.py`をどこまで再利用できるか、公式Fast-WAMのVideo DiTと現在の実装に互換性があるか、チェックポイントをロード時に変換するか変換済み形式で保存するか、最初の実データとして何を使うか、という点です。
