This is a **strong v6.14 design**, and it is very close to a production tokenizer pipeline. But I would **not run it unchanged yet**. There are a few important correctness issues—some are critical.

## 🔴 1. The pasted code has broken escape sequences

In your message, many strings appear like:

```python
text.replace("
", "
")
```

and:

```python
text.split("
")
```

and:

```python
ch not in "
	"
```

If this is literally the file content, Python will fail with `SyntaxError`.

They must be:

```python
"\r\n"
"\n"
"\r"
"\t"
```

For example:

```python
def normalize_spaces(text: str) -> str:
    return "".join(
        " "
        if (
            unicodedata.category(ch) == "Zs"
            or (ch.isspace() and ch not in "\n\t")
        )
        else ch
        for ch in text
    )
```

And:

```python
text = text.replace("\r\n", "\n").replace("\r", "\n")
```

This may only be a copy/paste rendering issue, but **verify the actual `.py` file first**.

---

# 🔴 2. Critical: deduplication is NOT global

Your header says:

> Global train/benchmark isolation validation

But this is still **not global deduplication during corpus construction**.

You create:

```python
dedup = DedupRegistry()
```

inside:

```python
write_stream_concurrent()
```

That means each category has its own dedup registry:

```text
Telugu       → dedup A
English      → dedup B
Mix          → dedup C
Parallel     → dedup D
Python       → dedup E
```

Therefore:

```text
Telugu document
       │
       ├── same normalized text appears in Parallel
       │
       └── both can survive
```

Your later validation only checks:

```text
TRAIN ∩ BENCH_BAL
TRAIN ∩ BENCH_WTD
```

It does **not check duplicate documents between training categories**.

For example:

```text
corpus_te.txt
corpus_par.txt
corpus_mix_nat.txt
```

can contain the same document.

### Better architecture

Use one global registry:

```python
GLOBAL_DEDUP = DedupRegistry()
```

Then pass it into every category processor:

```python
def write_stream_concurrent(
    ...,
    global_dedup: DedupRegistry,
):
```

Replace:

```python
dedup = DedupRegistry()
```

with the shared registry.

Then every accepted document is checked globally:

```text
Telugu ───────┐
English ──────┤
Mix ──────────┤
Parallel ─────┼──► GLOBAL DEDUP
Python ───────┘
```

This is especially important because your **parallel corpus can overlap heavily with Telugu and English corpora**.

---

# 🔴 3. Your benchmark isolation logic is probabilistic, not quota-controlled

This design:

```python
0–4       → balanced
5–9       → weighted
10–999    → train
```

is deterministic and good for reproducibility.

But it does **not guarantee exact benchmark sizes**.

For example, if you process only a limited number of documents, you could get:

```text
Target balanced: 500,000 chars
Actual:          320,000 chars
```

because the hash distribution is approximate.

Your current code stops only when all three targets are met:

```python
if (
    train_chars >= target_train and
    bench_bal_chars >= target_bench_bal and
    bench_wtd_chars >= target_bench_wtd
):
    break
```

The problem is that if the hash distribution is unlucky, it may scan a huge amount of data.

### Better approach

The current method is acceptable if you deliberately want a **deterministic hash split**.

But I would add a hard safety limit:

```python
max_scanned_chars = int(
    (target_train + target_bench_bal + target_bench_wtd) * 3.0
)
```

Then:

```python
if scanned_chars >= max_scanned_chars:
    break
```

And report:

```text
⚠️ Target not reached after maximum scan budget
```

Otherwise a streaming dataset could theoretically continue for a very long time.

---

# 🔴 4. Your 55/20/10/5/10 ratio is correct only for training

Your training quotas are:

```text
Telugu       55M
English      20M
Natural Mix  10M
Parallel      5M
Python       10M
────────────────
Total       100M
```

Correct:

```text
Telugu       55%
English      20%
Natural Mix  10%
Parallel      5%
Python       10%
```

But your **balanced benchmark** is:

```text
500k + 500k + 500k + 200k + 500k
```

So the benchmark composition is:

```text
Telugu       22.7%
English      22.7%
Mix          22.7%
Parallel      9.1%
Python       22.7%
```

That is genuinely balanced across categories except parallel.

Your weighted benchmark is:

```text
550k + 200k + 100k + 50k + 100k
```

which corresponds exactly to:

```text
55 / 20 / 10 / 5 / 10
```

Excellent design.

---

# 🟠 5. `has_executable_python()` still has one subtle issue

You include:

```python
ast.Expr
```

in:

```python
executable_nodes
```

This means a code sample containing only:

```python
"hello"
```

can be considered executable.

Also:

```python
print("hello")
```

is an `ast.Expr`, so that is legitimate executable code.

The real problem is that a module containing only a docstring is also:

```python
Expr(Constant("docstring"))
```

Therefore:

```python
"""This is only documentation."""
```

can pass.

### Better implementation

Use:

```python
def has_executable_python(tree: ast.AST) -> bool:
    for node in ast.walk(tree):

        if isinstance(node, ast.Expr):
            value = node.value

            # Ignore module/function/class docstrings
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                continue

            return True

        if isinstance(node, (
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Return,
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
            ast.Import,
            ast.ImportFrom,
            ast.Raise,
            ast.Assert,
            ast.Delete,
            ast.Match,
            ast.Global,
            ast.Nonlocal,
        )):
            return True

    return False
```

This is more accurate.

---

# 🟠 6. Python chunking can break syntax after the AST boundary

This is the biggest remaining code-specific weakness.

You correctly try:

```python
preferred.add(end_lineno - 1)
```

But if the full program is longer than:

```python
16,384 characters
```

you split it into chunks.

Those chunks may be:

```python
def function_a():
    ...
```

without the surrounding context, or:

```python
class MyClass:
    def method_a(self):
```

The resulting chunks may not be independently valid Python.

That is not necessarily bad for tokenizer training.

For tokenizer training, preserving:

```text
indentation
newlines
brackets
operators
keywords
```

is more important than every chunk being independently executable.

So I would change the comment from:

```text
AST-aware chunking
```

to:

```text
AST-guided lossless chunking
```

That is more technically accurate.

---

# 🟠 7. Your `safe_chunk_text()` can lose the exact document boundary semantics

You write:

```python
write_document(f, chunks)
```

and:

```python
for chunk in chunks:
    file_handle.write(chunk)

if not chunks[-1].endswith("\n"):
    file_handle.write("\n")

file_handle.write(CONFIG.doc_separator)
file_handle.write("\n")
```

This is mostly correct.

But `safe_chunk_text()` preserves newline endings using:

```python
splitlines(keepends=True)
```

and `split_text_lossless()` may split at a space and retain:

```python
" "
```

at the end of one chunk.

That is fine.

The important thing is:

```text
DOCUMENT
\n
<|DOC_SEP|>
\n
```

Your current design does preserve this.

I would only add a validation test:

```python
def validate_document_boundaries(path):
    with open(path, encoding="utf-8") as f:
        content = f.read()

    docs = content.split(CONFIG.doc_separator)

    for i, doc in enumerate(docs[:-1]):
        assert doc.endswith("\n"), f"Document {i} missing final newline"
```

---

# 🔴 8. Your tokenizer selection score is not yet scientifically correct

This part is the weakest conceptual component:

```python
py_frag = byte_fallback_rate * 100 + unk_rate * 100
```

You call this:

```text
Python fragmentation
```

but it is not actually Python fragmentation.

For example:

```text
Python A → 1.5 tokens/character
Python B → 2.0 tokens/character
```

Both may have:

```text
byte_fallback_rate = 0%
unk_rate = 0%
```

Your score sees them as identical.

### You should evaluate Python separately

Add a real Python benchmark:

```python
PYTHON_BENCHMARK_SNIPPETS = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    ...
]
```

Then calculate:

```python
python_tokens_per_char
python_bytes_per_token
```

and include:

```python
score_weight_python
```

based on actual Python token efficiency.

For your model, this is especially important because you deliberately allocate:

```text
10% Python code
```

to the tokenizer training corpus.

---

# 🟠 9. Your benchmark evaluation includes the document separator problem

You do:

```python
content = content.replace(
    CONFIG.doc_separator + "\n",
    ""
)
```

That is good for evaluation.

However, the benchmark files contain:

```text
document A
<|DOC_SEP|>

document B
<|DOC_SEP|>
```

When you remove the separator, you may accidentally concatenate documents depending on exact boundary formatting.

Your current writer produces:

```text
document A
<|DOC_SEP|>
document B
<|DOC_SEP|>
```

After replacement:

```text
document A
document B
```

Usually okay because the separator is preceded by a newline.

But for more accurate evaluation, encode each document separately:

```python
def encode_documents(path, sp):
    total_ids = 0
    total_chars = 0

    with open(path, encoding="utf-8") as f:
        content = f.read()

    for doc in content.split(CONFIG.doc_separator):
        doc = doc.strip("\n")
        if not doc.strip():
            continue

        ids = sp.encode(doc, out_type=int)
        total_ids += len(ids)
        total_chars += len(doc)

    return total_chars, total_ids
```

This avoids cross-document tokenization effects.

---

# 🟢 10. The 64K tokenizer is probably the strongest candidate

Based on your current target:

```text
Telugu       55%
English      20%
Code-mix     10%
Parallel      5%
Python       10%
```

and your current model:

```text
d_model = 1440
```

I would expect:

### 32K

Best:

```text
embedding parameter efficiency
```

but likely more Telugu agglutinative fragmentation.

### 48K

Probably the best balance.

### 64K

Likely best raw token efficiency, especially for:

```text
Telugu compound words
Telugu-English code-mix
Python identifiers
```

But it costs:

```text
32K → 46.08M tied embedding parameters
48K → 69.12M tied embedding parameters
64K → 92.16M tied embedding parameters
```

At your `d_model=1440`:

```text
64K - 32K = 46.08M additional parameters
```

That is substantial.

Therefore, **do not select purely by the weighted score**.

Your final selection should consider:

```text
Tokenizer quality
        +
Token efficiency
        +
Model parameter cost
        +
Training compute
```

Your current manifest already records:

```python
tied_embedding_parameters_m
```

which is excellent.

---

# 🟢 11. One important correction: `character_coverage` comment

You wrote:

```python
character_coverage=0.9999
```

with comment:

```text
cover 99.99% of chars with vocab, rest via byte fallback
```

That explanation is slightly misleading.

`character_coverage` controls the character coverage used during SentencePiece training. With:

```python
byte_fallback=True
```

unknown characters can be represented through byte pieces.

But this does **not mean exactly 0.01% of characters become byte fallback**.

The actual result should be measured by:

```text
byte_fallback_rate
```

which you already do.

So change the comment to:

```python
character_coverage: float = 0.9999
# SentencePiece training character coverage.
# Byte fallback provides robust representation for unseen characters.
```

---

# 🔴 12. Major data-quality issue: FineWeb-Edu is not necessarily "pure English"

Your filter:

```python
latin_ratio >= 0.90
```

does not guarantee English.

For example, this can pass:

```text
Spanish
French
German
Portuguese
```

because they use Latin script.

So your category name:

```text
English
```

is technically too strong.

You need either:

```text
English language identification
```

or rename the category:

```text
Latin / English
```

For a Telugu-English tokenizer, I recommend an actual language-quality filter.

The same issue applies to:

```python
is_acceptable_english()
```

It validates **Latin script**, not English language.

---

# 🔴 13. Parallel corpus orientation should be consistent

You randomly do:

```python
if hash_val % 2 == 0:
    raw = te_part + "\n" + en_part
else:
    raw = en_part + "\n" + te_part
```

This is a questionable choice for tokenizer training.

You already have:

```text
55% Telugu
20% English
10% code-mix
```

so the tokenizer has plenty of both languages.

For parallel data, I would use a stable format:

```text
English
Telugu
```

or:

```text
Telugu
English
```

but not random ordering.

Random ordering adds no meaningful linguistic benefit for a tokenizer.

If your goal is simply vocabulary co-occurrence, it is okay, but I prefer:

```text
EN
TE
```

consistently.

---

# 🟢 14. The strongest part of your pipeline is the deterministic reproducibility

This is excellent:

```text
same source datasets
        ↓
same normalization
        ↓
same deduplication
        ↓
same split hashing
        ↓
same document shuffle seed
        ↓
same corpus SHA-256
        ↓
32K / 48K / 64K trained on identical corpus
```

That makes the tokenizer comparison scientifically meaningful.

Your:

```python
corpus_sha256
```

is particularly important.

The final experiment should look like:

```text
32K ─┐
48K ─┼── SAME corpus SHA-256
64K ─┘
```

This is exactly how you should compare tokenizer candidates.

---

# My final verdict

### Current status: **8.5/10**

### Architecture: **excellent**

### Reproducibility: **excellent**

### Dataset split design: **very good**

### Benchmark design: **very good**

### Deduplication: **needs global registry**

### Python handling: **good, needs docstring correction**

### Tokenizer selection score: **needs real Python fragmentation metric**

### English purity: **needs actual language filtering**

### Code correctness: **verify the broken newline/tab escape sequences**

---

## Before running v6.14, I would make exactly these 5 final changes

### 1. Global deduplication

```text
ALL CATEGORIES
      ↓
ONE GLOBAL DEDUP REGISTRY
```

### 2. Add real Python benchmark token efficiency

```text
python_tokens_per_char
```

instead of using byte fallback as a proxy.

### 3. Fix the executable Python detector

Ignore pure docstrings.

### 4. Add an actual English-language filter

Latin script ≠ English.

### 5. Verify all literal `\n`, `\r`, and `\t` escapes in the actual Python file

If the code is exactly as pasted, it will not execute.

After these five fixes, I would consider the pipeline **ready to run and generate the final 32K/48K/64K tokenizer comparison**.
