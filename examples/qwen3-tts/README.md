# 試聴用の例: 合成音声を AMBE+2 で符号化して、両方のデコーダーで復号

| ファイル | 内容 |
|---|---|
| `<声>/original.wav` | 元の音声（24 kHz） |
| `<声>/text.txt` | 読み上げた文 |
| `<声>/frames_<条件>.ambe` | AMBE-3000R rate 34 のフレーム（7 バイト/20 ms）。誤りなしと、ビット誤りだけの条件 |
| `<声>/frames_<条件>.hex` | フレームの欠落がある条件。1 行 1 フレームの 16 進 14 桁で、欠落したフレームは "-" |
| `<声>/decoded_ambe2c_<条件>.wav` | 上のフレームを、従来のデコーダー（同梱の ambe2-c）で復号した音声（8 kHz） |
| `<声>/decoded_ambe49_causal_v1_<条件>.wav` | 同じフレームを、このリポジトリのモデル（`ambe49_causal_v1`）で復号した音声（8 kHz） |
| `generation.json` | 元の音声の生成条件（モデル、リビジョン、声の指示文、シード） |

## 条件

| 条件 | 内容 | 女性 PESQ（NB） | 男性 PESQ（NB） |
|---|---|---|---|
| `clean` | 誤りなし | 3.13 → 3.36 | 2.80 → 3.08 |
| `ber3` | 49 bit に一様なビット誤り 3 % | 1.57 → 2.26 | 1.48 → 2.40 |
| `ber8` | ビット誤り 8 % | 1.18 → 1.52 | 1.23 → 1.61 |
| `loss17` | 1〜3 フレームの連続した欠落（女性 18 %、男性 16 % のフレーム） | 2.08 → 2.63 | 1.49 → 2.09 |
| `ber3_loss17` | ビット誤り 3 % ＋ 欠落（女性 19 %、男性 19 % のフレーム） | 1.27 → 1.55 | 1.31 → 1.69 |

（従来のデコーダー → `ambe49_causal_v1`。元の音声を 8 kHz に落とし、アクティブレベルを −26 dBFS にそろえたものとの比較）

## 作り方

- **元の音声**: [tokuder-neurodec](https://github.com/tallcat4/tokuder-neurodec) の例と同じ音声です。[Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)（Apache-2.0）で、文章から新しく生成しました。
  - 実在の人物の声の複製（クローン）や、外部の参照音声は使っていません。
  - ピッチの加工や継ぎ接ぎもしていません。
- **フレーム**: 元の音声を 8 kHz に落とし、アクティブレベルを −26 dBFS にそろえて、[ambe2-c](https://github.com/tallcat4/ambe2-c) のエンコーダー（RATET 34）で符号化しました。誤りと欠落は、乱数のシードを条件と声ごとに固定して加えています。
- **復号の再現**: 次のコマンドで作りました。

  ```bash
  ambe2neurodec examples/qwen3-tts/female/frames_ber3.ambe decoded_ambe49_causal_v1_ber3.wav --reference decoded_ambe2c_ber3.wav
  ambe2neurodec examples/qwen3-tts/female/frames_loss17.hex decoded_ambe49_causal_v1_loss17.wav --reference decoded_ambe2c_loss17.wav
  ```

  - `--reference` で、従来のデコーダーの出力も書き出せます。従来のデコーダーの出力は、同梱の WAV とビット単位で一致します。
  - 同梱の `decoded_ambe49_causal_v1_*.wav` は GPU で作りました。CPU で復号した音とは、ごくわずかに違います。

## ライセンス

このディレクトリの音声とフレームは、[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) で提供します。権利を有する範囲で、すべての権利を放棄します。
この宣言は、音声を生成したモデルや第三者の権利を再許諾するものではありません。
