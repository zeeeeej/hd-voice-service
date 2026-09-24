from app.pipeline.text_chunker import TextChunker


def test_hard_boundaries():
    c = TextChunker()
    assert c.push("你好。世界！") == ["你好。", "世界！"]


def test_incremental_push_same_result():
    c = TextChunker()
    out = []
    for ch in "今天天气不错。我们走吧！":
        out += c.push(ch)
    out += c.flush()
    assert out == ["今天天气不错。", "我们走吧！"]


def test_soft_boundary_when_buffer_long():
    c = TextChunker(max_buffer=10)
    text = "一二三四五六七八九十，后面还有内容"
    out = c.push(text)
    assert out == ["一二三四五六七八九十，"]
    assert c.buf == "后面还有内容"


def test_no_split_under_max_buffer():
    c = TextChunker()
    assert c.push("没有标点的短句子") == []
    assert c.flush() == ["没有标点的短句子"]


def test_flush_allows_short_remainder():
    c = TextChunker()
    c.push("你")
    assert c.flush() == ["你"]


def test_punctuation_only_not_emitted_as_chunk():
    c = TextChunker()
    assert c.push("。。。") == []
    out = c.push("好的。")
    # 前面的纯标点被丢弃或与后续合并，不得产出纯标点碎片
    assert all(any(ch not in "。！？；!?;\n…：，、,—" for ch in s) for s in out)


def test_no_single_char_fragment_from_push():
    c = TextChunker(max_buffer=5)
    out = c.push("一，二三四五六七八九十")
    # "一，" 是 2 字符 ≥ min_chunk，允许；但不得出现单字块
    assert all(len(s.strip()) >= 2 for s in out)


def test_fast_first_splits_at_soft_boundary():
    c = TextChunker(fast_first=True)
    out = c.push("今天天气不错，我们一起去公园散步吧。")
    assert out[0] == "今天天气不错，"          # 首句在逗号处提前切
    assert "我们一起去公园散步吧。" in out     # 其余部分正常硬边界切分


def test_fast_first_only_once():
    c = TextChunker(fast_first=True)
    c.push("第一句，切了。")
    out = c.push("第二句不该在逗号，处切分。")
    assert out == ["第二句不该在逗号，处切分。"]


def test_fast_first_disabled():
    c = TextChunker(fast_first=False)
    out = c.push("今天天气不错，我们走吧。")
    assert out == ["今天天气不错，我们走吧。"]
