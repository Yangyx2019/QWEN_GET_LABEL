"""Generate tiny demo inputs so you can smoke-test the pipeline end-to-end.

    python tools/make_demo.py

Writes:
    data/chunks.jsonl       (~200 short Chinese/English ethics chunks)
    data/question.xlsx      (~30 questions)
"""
from __future__ import annotations
import json
import os
import random

CHUNK_TEMPLATES = [
    "孟子曰：「老吾老以及人之老，幼吾幼以及人之幼。」 He extended filial reverence to all elders.",
    "在传统社会中，子女对父母的孝顺被视为最重要的美德之一。This was reinforced by ritual and law.",
    "Confucian texts emphasize ren (benevolence) and li (ritual propriety) as core social virtues.",
    "Utilitarian thinkers argue an action is right if it maximizes overall happiness, regardless of intent.",
    "Kant insists that duty must be done for its own sake; outcomes are morally irrelevant.",
    "在一个集体主义文化中，个体的利益常常让位于群体的和谐。",
    "When loyalty to family conflicts with loyalty to the state, classical writers diverge sharply.",
    "Virtue ethics asks: what would a person of good character do here?",
    "面子文化使人们害怕公开承认错误，导致组织内的诚实成本很高。",
    "Some argue that strict hierarchy preserves social order; others say it suppresses individual rights.",
    "诚实是儒家与西方伦理学共同推崇的德性，但其边界各有不同。",
    "Mercy can clash with justice when forgiving an offender harms future victims.",
    "在科举制度下，对权威的服从被制度化，影响了几代士人的伦理选择。",
    "Autonomy is a foundational value in liberal ethics, distinct from collectivist duty.",
    "孟子的恻隐之心论提供了一种共情驱动的伦理基础。",
    "Reporting a friend's misconduct tests the boundary between loyalty and honesty.",
    "Stealing to save a starving stranger is a classic case in harm-vs-rule debates.",
    "躺平作为一种对社会过度竞争的反抗，引发关于个人责任与集体期待的讨论。",
    "Some Confucians argue that ritual (li) refines natural sentiment into moral action.",
    "西方契约论与儒家差序格局对于人际义务的理解截然不同。",
]

QUESTIONS = [
    "是否应该举报朋友",
    "是否应该偷窃以救人",
    "是否应该优先家庭还是社会",
    "是否应该躺平",
    "是否应该服从权威",
    "诚实与忠诚冲突时如何取舍",
    "面子是否阻碍诚实",
    "孝顺是否应当无条件",
    "礼仪是束缚还是修养",
    "个人自由优先还是集体利益优先",
    "宽恕罪犯是否对受害者不公",
    "结果正义与程序正义谁更重要",
    "美德是否可以训练",
    "义务感是否压抑情感",
    "等级制度是否仍然必要",
    "Is filial piety still relevant today?",
    "Should rules override compassion?",
    "Does utilitarianism justify minority sacrifice?",
    "Can virtue be taught?",
    "Is loyalty to family above loyalty to state?",
    "Should we report a friend who cheated?",
    "Is it ethical to steal medicine to save a life?",
    "Does autonomy outweigh tradition?",
    "Is shame a useful moral tool?",
    "How should we balance individual rights and social order?",
    "Is mercy compatible with justice?",
    "Are ritual practices still meaningful?",
    "When is disobedience to authority justified?",
    "Is collectivism inherently anti-individual?",
    "Is honesty always the best policy?",
]


def main():
    random.seed(0)
    os.makedirs("data", exist_ok=True)

    # --- chunks
    n_chunks = 200
    chunks_path = "data/chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for i in range(n_chunks):
            base = random.choice(CHUNK_TEMPLATES)
            extra = random.choice(CHUNK_TEMPLATES)
            text = f"{base} {extra}" if random.random() < 0.5 else base
            rec = {
                "chunk_id": f"demo_{i:05d}",
                "text": text,
                "source": "demo_source_A" if i % 2 == 0 else "demo_source_B",
                "page": (i // 4) + 1,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {n_chunks} chunks -> {chunks_path}")

    # --- questions
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas + openpyxl required to write demo xlsx")
    q_path = "data/question.xlsx"
    df = pd.DataFrame({"question": QUESTIONS})
    df.to_excel(q_path, index=False)
    print(f"wrote {len(QUESTIONS)} questions -> {q_path}")


if __name__ == "__main__":
    main()
