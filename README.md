# ambe2-neurodec

AMBE+2 half-rate（2450 bps、49 bit/20 ms。DMR、P25 Phase 2、NXDN などの音声フレーム）の互換デコーダーです。
従来のデコーダーより元の音声に近い音で復号でき、ビット誤りやフレームの欠落にも強くなっています。

## 試聴

合成音声（Qwen3-TTS で生成、CC0）を [ambe2-c](https://github.com/tallcat4/ambe2-c) のエンコーダーで符号化し、従来のデコーダー（ambe2-c。AMBE-3000R とほぼ同じ音を出します）と、このデコーダーで復号しました。
誤りのある条件では、両方のデコーダーに同じフレームを渡しています。

| 声 | 読み上げた文 | 元の音声 |
|---|---|---|
| 女性 | このソフトウェアは、すべての人のために、常に自由であり続けます。 | [original.wav](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/female/original.wav) |
| 男性 | GPL-3.0は、研究、改変、共有する自由を保証するオープンなライセンスです。 | [original.wav](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/male/original.wav) |

| 受信の条件 | 女性: 従来のデコーダー | 女性: `ambe49_causal_v1` | 男性: 従来のデコーダー | 男性: `ambe49_causal_v1` |
|---|---|---|---|---|
| 誤りなし | [ambe2c](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/female/decoded_ambe2c_clean.wav) | [neurodec](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/female/decoded_ambe49_causal_v1_clean.wav) | [ambe2c](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/male/decoded_ambe2c_clean.wav) | [neurodec](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/male/decoded_ambe49_causal_v1_clean.wav) |
| ビット誤り 3 % | [ambe2c](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/female/decoded_ambe2c_ber3.wav) | [neurodec](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/female/decoded_ambe49_causal_v1_ber3.wav) | [ambe2c](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/male/decoded_ambe2c_ber3.wav) | [neurodec](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/male/decoded_ambe49_causal_v1_ber3.wav) |
| ビット誤り 8 % | [ambe2c](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/female/decoded_ambe2c_ber8.wav) | [neurodec](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/female/decoded_ambe49_causal_v1_ber8.wav) | [ambe2c](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/male/decoded_ambe2c_ber8.wav) | [neurodec](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/male/decoded_ambe49_causal_v1_ber8.wav) |
| フレームの欠落 約 17 % | [ambe2c](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/female/decoded_ambe2c_loss17.wav) | [neurodec](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/female/decoded_ambe49_causal_v1_loss17.wav) | [ambe2c](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/male/decoded_ambe2c_loss17.wav) | [neurodec](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/male/decoded_ambe49_causal_v1_loss17.wav) |
| ビット誤り 3 % ＋ 欠落 約 17 % | [ambe2c](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/female/decoded_ambe2c_ber3_loss17.wav) | [neurodec](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/female/decoded_ambe49_causal_v1_ber3_loss17.wav) | [ambe2c](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/male/decoded_ambe2c_ber3_loss17.wav) | [neurodec](https://github.com/tallcat4/ambe2-neurodec/raw/main/examples/qwen3-tts/male/decoded_ambe49_causal_v1_ber3_loss17.wav) |

- **フレーム**: 復号に使ったフレームも同梱しています。手元で復号し直せます（[examples/qwen3-tts](examples/qwen3-tts/README.md)）。
- **元の音声との比較**: 元の音声を 8 kHz に落として比べると、PESQ は誤りなしで女性 3.13 → 3.36・男性 2.80 → 3.08、ビット誤り 3 % で女性 1.57 → 2.26・男性 1.48 → 2.40 です（従来のデコーダー → このデコーダー）。

## 中身

| 部分 | 内容 |
|---|---|
| デコーダー | 49 bit/20 ms → 8 kHz 音声。受信ビットと、従来のデコーダーの出力の log-mel を入力に、FocalNet（focal modulation）でクリーン音声の log-mel を予測し、iSTFT ボコーダー（Vocos 型）で波形にします（[しくみ](docs/method.md)） |
| 従来のデコーダー | `ambe2_neurodec/csrc/`: ambe2-c のデコーダー（blip25-vocoder の C 移植、MIT）。ニューラルデコーダーの入力の一部として使います。初回の実行時に C コンパイラーでビルドします |
| 学習済みモデル | `models/ambe49_causal_v1.pt`（重み、float16 で保存）と `models/ambe49_causal_v1.json`（構造・学習条件・評価値） |
| 入力形式 | AMBE-3000R の rate 34（7 バイト/フレーム）、mbelib / DSD の ambe_d 順（7 バイト）、rate 33（3600x2450、FEC 付き 9 バイト）、16 進テキスト（欠落したフレームを書ける） |
| 試聴用の例 | `examples/qwen3-tts/`: 元の音声、フレーム、両方のデコーダーの復号音（[上の「試聴」](#試聴)） |

ボコーダー部分の事前学習には [neurodec](https://github.com/tallcat4/neurodec) を使いました。

## モデル

| | `ambe49_causal_v1` |
|---|---|
| 構造 | 因果的。フレーム t の復号に使うのは、フレーム t+1 までのビット。FocalNet は幅 192 × 3 段 |
| 遅延（従来のデコーダーの出力に対して） | 20 ms（1 フレーム）の先読み ＋ iSTFT の重ね合わせ 32 ms |
| パラメータ数（重み） | 約 340 万（6.5 MiB、うちボコーダー 170 万） |
| CPU 1 スレッドでの速さ | 音声の長さの約 1/70 |

未来のフレームを 1 つしか使わないので、受信しながら逐次に復号する用途に向いています（このリポジトリの `decode` は、まとめて復号する実装です）。

## 品質

LibriSpeech test-clean の未知話者 40 名 × 5 発話と、JSUT の 50 発話（日本語）の、計 250 発話での評価です。
両方のデコーダーに同じフレームを渡し、元の音声を 8 kHz に落としたものと比べています。

| 受信の条件 | PESQ（NB） | STOI | log-mel 距離 |
|---|---|---|---|
| 誤りなし | 2.86 → **3.06** | 0.907 → **0.919** | 5.00 → **4.24 dB** |
| ビット誤り 1 % | 2.26 → **2.78** | 0.850 → **0.899** | 5.57 → **4.52 dB** |
| ビット誤り 3 % | 1.73 → **2.36** | 0.765 → **0.864** | 6.57 → **5.00 dB** |
| フレームの欠落 約 5 % | 2.46 → **2.80** | 0.878 → **0.905** | 5.37 → **4.43 dB** |
| フレームの欠落 約 10 % | 2.22 → **2.61** | 0.850 → **0.890** | 5.79 → **4.63 dB** |

（従来のデコーダー → `ambe49_causal_v1`）

- **話者別（誤りなし、PESQ）**: 英語の男性 2.93 → 3.01、英語の女性 2.92 → 3.17、日本語（JSUT）2.63 → 2.94。
- **より過酷な条件**: 日本語（JSUT の 6 発話）で、ビット誤り 8 % でも PESQ 1.20 → 1.50、フレームの欠落 約 28 % で 1.47 → 1.82 です（[詳細](docs/method.md#過酷な条件)）。
- **自然さ（UTMOS）**: 誤りなしでは従来のデコーダーとほぼ同じ（2.38 → 2.31）で、誤りのある条件では上回ります（ビット誤り 3 % で 1.32 → 1.80。[詳細](docs/method.md#評価)）。
- **大きいモデルとの比較**: 同じ構成で幅 768（約 2640 万パラメータ）のモデルとの ABX テスト（日本語 40 問、誤りなし・ビット誤り・欠落を混ぜたもの）では、25/40 問の正解で、統計的には聞き分けられませんでした（[詳細](docs/method.md#モデルの大きさ)）。
- **比べている相手**: 正解は、エンコーダーに入れる前のクリーン音声です。フレームは ambe2-c のエンコーダーで作りました。チップ（AMBE-3000R）がエンコードしたフレームや、実際の無線で受信したフレームでは、まだ評価していません。
- **誤りのモデル**: ビット誤りは 49 bit に一様に加えています。実際の無線では、FEC のあとの誤りは偏ります。
- **得意・不得意**:
  - 学習データは音声だけです。トーンや DTMF、背景雑音フレームなどの音声以外の信号は評価していません。正しく再現される保証はありません。
  - 低い声（男性）は、高い声よりやや苦手です。

## 動作環境

- Python 3.10 以上、PyTorch 2.1 以上、numpy、soundfile
- C コンパイラー（`$CC`、または PATH にある cc / gcc / clang）。同梱の従来のデコーダーを初回にビルドします（Windows では MinGW-w64 の gcc で確認）。
- CPU でも動きます。デスクトップ CPU の 1 スレッドで、音声の長さの約 1/70 の時間で復号できます（[モデル](#モデル)）。CUDA があれば自動で使います。

## インストール

```bash
pip install -e .        # ambe2neurodec コマンドが入ります（モデルはリポジトリの models/ を使います）
pip install -e .[test] && pytest
```

## 使い方

```bash
ambe2neurodec frames.ambe out.wav                     # AMBE-3000R rate 34（7 バイト/20 ms、ambe2-c の wav2ambe の出力）
ambe2neurodec frames.ambe out.wav --reference ref.wav # 従来のデコーダーの出力も書き出す
ambe2neurodec frames.bin out.wav --format ambe_d      # mbelib / DSD の ambe_d 順（49 bit を MSB から 7 バイトに詰めたもの）
ambe2neurodec frames.bin out.wav --format rate33      # AMBE-3000R rate 33（3600x2450、FEC 付き 9 バイト）
ambe2neurodec frames.hex out.wav                      # 1 行 1 フレームの 16 進 14 桁（rate 34 の順）。"-" か空行は欠落したフレーム
python -m ambe2_neurodec ...                          # インストールせずに実行
```

### 入力についての注意

- **フレームの順序**: フレームは時間順に並べてください。受信できなかったフレームは、16 進テキストの "-" で同じ位置に入れてください。欠落したフレームは、従来のデコーダーの補間（直前のフレームの繰り返し）と、欠落のフラグの両方でデコーダーに伝わります（学習時と同じ処理です）。
- **rate 33**: 先頭の符号語（c0）が訂正できないフレームは、欠落したフレームとして扱います。
- **出力のタイミング**: 出力は、従来のデコーダーの出力と同じタイミングです（入力の音声から約 45 ms 遅れます）。最後のフレームの先読みの分は、欠落したフレームで補います。
- **出力の再現性**: 出力の波形サンプルは、ボコーダーの励振の雑音の乱数（`--seed`）によってわずかに変わります。

### Python から

```python
from ambe2_neurodec import decode, frames, load
bits, lost = frames.read("frames.ambe")                       # bits [T, 49]（ambe_d 順）、lost [T]
wav = decode(load("models/ambe49_causal_v1.pt"), bits, lost)  # float32、8 kHz
```

## 学習データ

- **音声**: [LibriSpeech](https://www.openslr.org/12) の train-clean-100、train-clean-360、train-other-500（960 時間、CC BY 4.0）。
- **フレーム**: 上の音声を 8 kHz に落とし、アクティブレベルを −26 dBFS にそろえて、ambe2-c のエンコーダー（RATET 34）で符号化しました。学習中は、ビット誤り（0〜3 %）とフレームの欠落（1〜3 フレームの連続）をランダムに加えています。
- **ボコーダー部分**: クリーン音声で事前学習し、デコーダーの学習中は固定しました。事前学習には [VCTK](https://doi.org/10.7488/ds/2645)（CC BY 4.0）、[JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)（個人・非商用の研究目的に限る、再配布不可）、[JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)（配布元の利用規約による）を使っています。
- **評価に使った音声**: LibriSpeech test-clean（学習に使っていない話者）と JSUT。JSUT はボコーダー部分の事前学習には含まれています。

## ライセンス

- コード: GPL-3.0-or-later（GNU GPL バージョン 3 またはそれ以降の任意のバージョン。[LICENSE](LICENSE)）。
- 同梱したほかのコード:
  - `ambe2_neurodec/csrc/`: ambe2-c のデコーダー（MIT、[LICENSE.ambe2-c](ambe2_neurodec/csrc/LICENSE.ambe2-c)）。blip25-vocoder（MIT、Chance Lindsey）の派生物です（[LICENSE.blip25](ambe2_neurodec/csrc/LICENSE.blip25)）。
  - `ambe2_neurodec/focalnet.py`: [FocalCodec](https://github.com/lucadellalib/focalcodec) の focal modulation（Apache-2.0、Luca Della Libera。[LICENSE.focalnet](ambe2_neurodec/LICENSE.focalnet)）。
- `examples/qwen3-tts/` の音声とフレーム: CC0 1.0（[examples/qwen3-tts/README.md](examples/qwen3-tts/README.md)）。
- 学習済みの重み: 下の注記を参照。

<sub>学習データの一部には配布元の利用規約があり、JVS は非商用・研究目的の利用に限られ、JSUT にも配布元の利用規約があります。法的な判断を断言することはできませんが、日本の著作権法第 30 条の 4 第 2 号（情報解析のための利用。著作隣接権にも第 102 条第 1 項で準用）により、これらのコーパスをこのモデルの学習に用いることは著作権法上許されると考えられます。また、学習済みの重みは各コーパスの音声を含まず復元もしないため、重みの配布はコーパスの再配布には当たらないと考えられます。ただし、コーパスの利用規約の解釈や日本国外での扱いについては、利用者ご自身で確認してください。</sub>

<sub>AMBE、AMBE+2 は Digital Voice Systems, Inc.（DVSI）の商標です。このリポジトリは DVSI とは関係がなく、同社の承認・関与を受けたものではありません。「AMBE+2」の名称は、そのフレームとの互換性を示す目的でのみ使用しています。AMBE+2 には特許があります（US 8,359,197 ほか）。</sub>
