# llama.geta (Gateway Enabling Tool Access)

**llama.geta**は、小型言語モデルがContinueの高度なツール（Function Calling）を利用できるようにするAPIサーバーです。

## 概要

llama-cpp-agentが提供する小型モデルでも呼び出しやすいツール機構を利用して、**小型モデルがContinueの提供するツール（ファイル操作、検索など）を正確に呼び出せる**ようにします。

### アーキテクチャの違い

**従来（大型モデル）:**
```
Continue → 大型モデル (GPT-5など) → ツール呼び出し判断 → Continueのツール実行
```

**llama.geta使用（小型モデル）:**
```
Continue → llama.geta → 小型モデル → ツール呼び出し判断 → llama.getaのツールを実行 → Continueのツール実行に変換
```

## 特徴

- **小型モデルによるツール呼び出し**: ローカルの小型モデルがContinueの豊富なツールを利用可能
- **OpenAI互換API**: `/v1/chat/completions`エンドポイントでOpenAI APIと互換性

## インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/morikoudayo/llama.geta.git
cd llama.geta
```

### 2. 依存関係のインストール

```bash
uv sync
```

### 3. llama.cppサーバーの起動

別のターミナルでllama.cppサーバーを起動してください：

```bash
# 例：llama.cppサーバーをポート8080で起動
./llama-server -m /path/to/model.gguf --port 8080
```

## 使い方

### サーバーの起動

```bash
export LLAMA_CPP_SERVER_URL=http://localhost:8080
uv run python main.py
```

llama.getaは `http://localhost:8000` で起動します。

### Continueとの連携

Continueの設定ファイル（`.continue/config.json`）に以下を追加：

```json
{
  "models": [
    {
      "title": "Local Small Model via llama.geta",
      "provider": "openai",
      "model": "local-model",
      "apiBase": "http://localhost:8000/v1"
    }
  ]
}
```

## 現在サポートされているツール

現在、以下のContinueツールをサポートしています（小型モデルから呼び出し可能）：

- **read_file**: ワークスペース内のファイルの内容を読み取る

追加のツール定義は [geta/tools.py](geta/tools.py) で行います。

## Continueツールの追加方法

小型モデルが新しいContinueツールを呼び出せるようにするには：

1. [geta/tools.py](geta/tools.py) に新しい関数を定義（Continueツールのシグネチャに対応）
2. 関数内で `ToolCallIntercepted` 例外を発生させる
3. [geta/agent.py](geta/agent.py) の `initialize()` 関数で新しいツールを登録

例（write_fileツールの追加）：

```python
# geta/tools.py
def write_file(filepath: str, content: str) -> str:
    """
    Write content to a file.

    Args:
        filepath: File path relative to workspace root
        content: Content to write
    """
    raise ToolCallIntercepted("write_file", {
        "filepath": filepath,
        "content": content
    })

# geta/agent.py
tools = [
    LlamaCppFunctionTool(read_file),
    LlamaCppFunctionTool(write_file)  # 新しいツールを追加
]
```

**重要**: ここで定義する関数は実際にツールを実行するわけではなく、小型モデルがツール呼び出しを判断するためのスキーマ定義です。実際のツール実行はContinueが行います。

## 技術詳細

### ツール呼び出しの仕組み

1. **リクエスト受信**: Continueからユーザーのリクエストをllama.getaが受信
2. **小型モデルで判断**: llama-cpp-agentが小型モデルにリクエストを送信し、ツール呼び出しが必要かを判断
3. **ツールインターセプト**: 小型モデルがツール呼び出しを決定すると、定義されたツール関数が呼ばれ、`ToolCallIntercepted` 例外が発生
4. **ツール呼び出しリクエスト生成**: llama.getaが例外をキャッチし、OpenAI形式のツール呼び出しリクエストをContinueにストリーミング
5. **Continueがツール実行**: Continueが実際のツール（ファイル操作など）を実行
6. **結果受信**: Continueがツール実行結果をllama.getaに送信
7. **最終レスポンス**: llama.getaが小型モデルでツール結果を元に最終レスポンスを生成
8. **結果表示**: Continueがユーザーに結果を表示

### 例外ベースのインターセプト仕組み

llama.getaは、小型モデルがツールを呼び出そうとした瞬間に、Python例外を使ってその呼び出しをインターセプトします。

**なぜ実際にツールとしての動作をせずに、例外を起こすのか:**
- llama.getaがツールを直接実行してしまうと、Continueのセキュリティ機能やUIが活用できない

**llama.getaの解決策:**
```python
def read_file(filepath: str) -> str:
    """ファイルを読み取る"""
    # 実際にファイルを読まず、例外を投げて処理を中断
    raise ToolCallIntercepted("read_file", {"filepath": filepath})
```

この仕組みにより：
1. 小型モデルが「read_fileを呼び出したい」と判断
2. 実際にファイルを読むのではなく、`ToolCallIntercepted`例外を発生させる
3. llama.getaが例外を捕捉し、ツール名とパラメータを取得
4. OpenAI形式でContinueに「read_fileを実行してください」とリクエスト
5. Continueが実際のファイル読み取りを実行
